"""
Simplest possible demo: wait, then ask the Deep Agent something.

time.sleep(seconds) just pauses the script. That's it - that's the "schedule".
"""

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


agent = create_deep_agent(model=model, tools=[get_weather])


async def main():
    print("Waiting 60 seconds...")
    time.sleep(60)  # <-- change this to however long you want to wait

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="what is the weather in ny?")]}
    )
    print(result["messages"][-1].content)


asyncio.run(main())


