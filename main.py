import asyncio
import os
import time
import uuid
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.responses import FileResponse
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core.runnables import RunnableLambda
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.langchain import LangchainProcessor
from pipecat.runner.livekit import generate_token, generate_token_with_agent
from pipecat.runner.run import app
from pipecat.runner.types import LiveKitRunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from llm_config import model

load_dotenv()


@app.get("/healthz", include_in_schema=False)
async def health():
    return {"status": "ok"}


_bot_tasks: set[asyncio.Task] = set()


@app.get("/livekit", include_in_schema=False)
async def livekit_client_page():
    return FileResponse(Path(__file__).parent / "livekit_client.html")


@app.post("/livekit/connect", include_in_schema=False)
async def livekit_connect():
    required = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Missing environment variable(s): {', '.join(missing)}",
        )

    room_name = f"pipecatx-{uuid.uuid4().hex[:12]}"
    livekit_url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]

    agent_token = generate_token_with_agent(
        room_name, "Pipecat Agent", api_key, api_secret
    )
    user_token = generate_token(room_name, "User", api_key, api_secret)
    task = asyncio.create_task(
        bot(
            LiveKitRunnerArguments(
                room_name=room_name,
                url=livekit_url,
                token=agent_token,
            )
        )
    )
    _bot_tasks.add(task)

    def clear_finished_task(finished: asyncio.Task) -> None:
        _bot_tasks.discard(finished)
        if not finished.cancelled() and (error := finished.exception()):
            logger.opt(exception=error).error("LiveKit bot task failed")

    task.add_done_callback(clear_finished_task)
    return {"url": livekit_url, "token": user_token, "room": room_name}


@tool
def get_weather(city: str):
    """Get the weather for a city."""
    return f"The weather of {city} is 16 degrees Celsius and rainy."


agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a friendly voice assistant. Keep replies short and plain.",
)


async def ask_deep_agent(data: dict[str, str]) -> str:
    started_at = time.perf_counter()
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=data["input"])]}
        )
        return str(result["messages"][-1].content)
    finally:
        logger.info(
            "DeepAgent request completed in {:.2f}s",
            time.perf_counter() - started_at,
        )


async def bot(runner_args):
    missing = [
        name
        for name in ("DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        logger.error("Missing environment variable(s): {}", ", ".join(missing))
        return

    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True, audio_out_enabled=True
            ),
            "livekit": lambda: LiveKitParams(
                audio_in_enabled=True, audio_out_enabled=True
            ),
        },
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
        ),
    )

    user_context, assistant_context = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=0.4,
                    )
                ]
            ),
        ),
    )
    voice_agent = LangchainProcessor(RunnableLambda(ask_deep_agent))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_context,
            voice_agent,
            tts,
            transport.output(),
            assistant_context,
        ]
    )
    task = PipelineTask(pipeline)

    if isinstance(runner_args, LiveKitRunnerArguments):

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant_id):
            voice_agent.set_participant_id(participant_id)
            await task.queue_frame(
                TTSSpeakFrame("Hi, I'm your voice assistant. What can I do for you?")
            )

        @transport.event_handler("on_participant_disconnected")
        async def on_participant_disconnected(transport, participant_id):
            await task.cancel()

    else:

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            await task.queue_frame(
                TTSSpeakFrame("Hi, I'm your voice assistant. What can I do for you?")
            )

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await task.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


if __name__ == "__main__":
    import sys

    from pipecat.runner.run import main

    port = os.environ.get("PORT")
    if port and "--host" not in sys.argv and "--port" not in sys.argv:
        sys.argv += ["--host", "0.0.0.0", "--port", port]
    main()
