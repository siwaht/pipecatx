import asyncio
import os
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
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from llm_config import model

load_dotenv()


# Render's health check only needs a cheap HTTP 200. Register these on the
# runner's FastAPI app at import time so they answer as soon as uvicorn is
# listening: no Deepgram/ElevenLabs/Cloudflare calls, no Silero download, no
# websocket. The runner's own "/" is a redirect to the prebuilt client for
# webrtc and a webhook handler (POST) for telephony, so /healthz is the route
# to point Render's health check at.
@app.get("/healthz", include_in_schema=False)
@app.get("/health", include_in_schema=False)
async def health():
    """Liveness/readiness probe. Must never block or touch external services."""
    return {"status": "ok"}


# --- LiveKit path: the transport that actually works behind Render ----------
#
# The built-in SmallWebRTC transport is peer-to-peer: the browser has to reach
# this process directly over UDP for media. Render web services are only
# reachable on 443/HTTPS and the container has no public IP, so the SDP
# exchange succeeds (the UI says "connected") and then media never establishes.
#
# With LiveKit, both the browser and this process connect *outbound* to LiveKit
# Cloud, which relays the media. Nothing needs to reach us on UDP, so it works
# unchanged on Render. Requires LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET.

# Keep strong references to in-flight bot tasks so they aren't garbage
# collected mid-call.
_bot_tasks: set[asyncio.Task] = set()


@app.get("/livekit", include_in_schema=False)
async def livekit_client_page():
    """Serve the minimal browser client for the LiveKit path."""
    return FileResponse(Path(__file__).parent / "livekit_client.html")


@app.post("/livekit/connect", include_in_schema=False)
async def livekit_connect():
    """Create a room, start the bot in it, and return client credentials."""
    url = os.environ.get("LIVEKIT_URL")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    missing = _missing_env("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    if missing:
        detail = f"Missing environment variable(s): {', '.join(missing)}"
        logger.error(detail)
        raise HTTPException(status_code=503, detail=detail)

    room_name = f"pipecatx-{uuid.uuid4().hex[:12]}"
    agent_token = generate_token_with_agent(
        room_name, "Pipecat Agent", api_key, api_secret
    )
    user_token = generate_token(room_name, "User", api_key, api_secret)

    runner_args = LiveKitRunnerArguments(
        room_name=room_name, url=url, token=agent_token
    )
    task = asyncio.create_task(bot(runner_args))
    _bot_tasks.add(task)
    task.add_done_callback(_bot_tasks.discard)

    logger.info(f"LiveKit session requested, room={room_name}")
    return {"url": url, "token": user_token, "room": room_name}


@tool
def get_weather(city: str):
    """Get the weather for a city."""
    return f"The weather of {city} is 16 degrees Celsius and rainy."


# Your existing LangChain Deep Agent.
agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a friendly voice assistant. Keep replies short and plain.",
)


# Pipecat gives this function {"input": "what is the weather in New York?"}.
# It returns one final string for Deepgram TTS to speak.
async def ask_deep_agent(data: dict[str, str]) -> str:
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=data["input"])]}
    )
    return str(result["messages"][-1].content)


# This is the small adapter between Pipecat and your Deep Agent.
voice_agent = LangchainProcessor(RunnableLambda(ask_deep_agent))


def _missing_env(*names: str) -> list[str]:
    """Return which of the given environment variables are unset or empty."""
    return [name for name in names if not os.environ.get(name)]


async def bot(runner_args):
    # bot() runs as a FastAPI background task, so an exception in here never
    # reaches the browser: the client just sits there "connected" while nothing
    # happens. .env is gitignored, so on Render these must be set in the
    # dashboard's Environment section. Fail loudly in the log instead.
    logger.info(f"bot task started ({type(runner_args).__name__})")
    missing = _missing_env("DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")
    if missing:
        logger.error(
            "Cannot start the bot: missing environment variable(s) {}. "
            "Set them in the Render dashboard (Environment) and redeploy.",
            ", ".join(missing),
        )
        return

    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True, audio_out_enabled=True
            ),
            # Twilio Media Streams use 8kHz audio. The serializer/add_wav_header
            # are set automatically by create_transport() for telephony.
            # LiveKit relays media through LiveKit Cloud, so this is the
            # transport to use on Render. See /livekit above.
            "livekit": lambda: LiveKitParams(
                audio_in_enabled=True, audio_out_enabled=True
            ),
            # Twilio Media Streams use 8kHz audio. The serializer/add_wav_header
            # are set automatically by create_transport() for telephony.
            "twilio": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        },
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    # Override this in .env with your own ElevenLabs voice ID if you want.
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice=voice_id),
    )


    context = LLMContext()
    user_context, assistant_context = LLMContextAggregatorPair(
        context,
        # Detects when the user has stopped speaking.
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # microphone audio
            stt,  # speech -> text
            user_context,  # waits for the end of the user's turn
            voice_agent,  # text -> your Deep Agent -> reply text
            tts,  # reply text -> speech
            transport.output(),  # speaker audio
            assistant_context,  # save the reply in Pipecat's context
        ]
    )
    task = PipelineTask(pipeline)

    greeting = "Hi, I'm your voice assistant. What can I do for you?"

    # LiveKitTransport and the WebRTC/websocket transports expose different
    # event names, so register against whichever one we actually built.
    if isinstance(runner_args, LiveKitRunnerArguments):

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant_id):
            logger.info(f"media connected: participant {participant_id} joined")
            await task.queue_frame(TTSSpeakFrame(greeting))

        @transport.event_handler("on_participant_disconnected")
        async def on_participant_disconnected(transport, participant_id):
            logger.info("media disconnected: cancelling pipeline task")
            await task.cancel()

    else:
        # If "media connected" never appears after the browser says it
        # connected, the SDP exchange succeeded but the WebRTC media path did
        # not: that is a network/ICE problem, not an application problem.
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("media connected: audio is flowing, pipeline is live")
            await task.queue_frame(TTSSpeakFrame(greeting))

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("media disconnected: cancelling pipeline task")
            await task.cancel()

    logger.info("running pipeline, waiting for media to connect")
    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


if __name__ == "__main__":
    import sys

    from pipecat.runner.run import main

    # Render (and most PaaS hosts) assign the listen port via $PORT and expect
    # the process to bind 0.0.0.0 so their proxy can reach it. The pipecat
    # runner defaults to localhost:7860, which leaves the health check hanging
    # until Render times the deploy out. Inject --host/--port ahead of parsing
    # unless they were passed explicitly on the CLI.
    #
    # Do not remove this block: reverting it is what made the deploy after
    # 3421846 time out.
    host = "0.0.0.0"
    port = os.environ.get("PORT", "7860")
    if "--host" not in sys.argv and "--port" not in sys.argv:
        sys.argv += ["--host", host, "--port", port]
    else:
        host, port = "(from CLI)", "(from CLI)"

    logger.info(f"Starting pipecat runner HTTP server on {host}:{port}")
    main()
    logger.info("Pipecat runner HTTP server stopped")
