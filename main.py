import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core.runnables import RunnableLambda
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.langchain import LangchainProcessor
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from llm_config import model

load_dotenv()


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


async def bot(runner_args):
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
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

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
