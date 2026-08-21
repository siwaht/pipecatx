"""
Minimal demo: schedule a Deep Agent to run a task later (or every morning).

This reuses your existing Deep Agent setup (llm_config.model + a weather tool)
from main.py, just without the Pipecat voice pipeline attached. It's meant to
be run directly so you can see scheduling behavior in this window.

Two modes, pick one at the bottom:
  1. run_once_at(...)  -> fires a single time at a specific future datetime
  2. run_daily_at(...) -> fires every day at a fixed HH:MM, forever

Note: this only works while the script is running (it's a local `while True`
loop). For scheduling that survives your machine being off, you'd deploy the
agent to LangGraph Platform and use a server-side cron job instead
(see: https://docs.langchain.com/langsmith/cron-jobs).
"""

import asyncio
import time
from datetime import datetime, timedelta

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain.tools import tool

from llm_config import model

load_dotenv()


@tool
def get_weather(city: str):
    """Get the weather for a city."""
    return f"The weather of {city} is 16 degrees Celsius and rainy."


agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a friendly assistant. Keep replies short and plain.",
)


async def ask_agent(task: str) -> str:
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return str(result["messages"][-1].content)


def run_agent_task(task: str) -> None:
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Running task: {task!r}")
    reply = asyncio.run(ask_agent(task))
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Agent says: {reply}\n")


def run_once_at(target_time: datetime, task: str, poll_seconds: int = 15) -> None:
    """Block until target_time, then run the agent once."""
    print(f"Waiting until {target_time:%Y-%m-%d %H:%M:%S} to run: {task!r}")
    while datetime.now() < target_time:
        time.sleep(poll_seconds)
    run_agent_task(task)


def run_daily_at(hour: int, minute: int, task: str, poll_seconds: int = 30) -> None:
    """Run the agent every day at hour:minute, until you stop the script (Ctrl+C)."""
    print(f"Scheduling '{task}' every day at {hour:02d}:{minute:02d}. Press Ctrl+C to stop.")

    def next_run_from(now: datetime) -> datetime:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    next_run = next_run_from(datetime.now())
    print(f"Next run: {next_run:%Y-%m-%d %H:%M:%S}")

    while True:
        now = datetime.now()
        if now >= next_run:
            run_agent_task(task)
            next_run = next_run_from(datetime.now())
            print(f"Next run: {next_run:%Y-%m-%d %H:%M:%S}")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    # --- Mode 1: run once, ~1 minute from now (so you can see it fire quickly) ---
    target = datetime.now() + timedelta(minutes=1)
    run_once_at(target, "What's the weather like in New York?")

    # --- Mode 2: run every morning at 07:00 (uncomment to use instead) ---
    # run_daily_at(7, 0, "Check the weather in New York and give me a short summary.")
