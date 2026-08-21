import asyncio
import time

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

from llm_config import model

load_dotenv()


@tool
def get_weather(city: str):
    """Get the weather for a city."""
    return f"The weather of {city} is 16 degrees Celsius and rainy."


# Sub-agent: waits, then does the task.
reminder_agent = create_deep_agent(model=model, tools=[get_weather])


async def remind_me_later(seconds: int, task: str):
    time.sleep(seconds)
    result = await reminder_agent.ainvoke({"messages": [HumanMessage(content=task)]})
    print(result["messages"][-1].content)


# Main agent: just a normal Deep Agent, no reminder tool needed for this demo.
main_agent = create_deep_agent(model=model)


async def main():
    print("Main agent handling the request...")
    reply = await main_agent.ainvoke(
        {"messages": [HumanMessage(content="I'll check the weather for you in 20 seconds.")]}
    )
    print(reply["messages"][-1].content)

    await remind_me_later(20, "What's the weather in New York?")


asyncio.run(main())
