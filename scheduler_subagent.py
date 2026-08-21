import asyncio
import threading
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


# The reminder/to-do sub-agent. It just carries out one task's text.
reminder_agent = create_deep_agent(model=model, tools=[get_weather])


def run_task(task: str) -> None:
    """Actually run one task through the sub-agent and print the reply."""
    result = asyncio.run(
        reminder_agent.ainvoke({"messages": [HumanMessage(content=task)]})
    )
    reply = result["messages"][-1].content
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {reply}")


def next_run_time(hour: int, minute: int, weekday: int | None) -> datetime:
    """Work out the next datetime matching hour:minute (and weekday, if given)."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if weekday is None:
        # daily / once: today at hour:minute, or tomorrow if that's already passed
        if target <= now:
            target += timedelta(days=1)
    else:
        # weekly: next occurrence of that weekday (0=Monday ... 6=Sunday)
        days_ahead = (weekday - target.weekday()) % 7
        target += timedelta(days=days_ahead)
        if target <= now:
            target += timedelta(days=7)

    return target


def scheduler_loop(
    task: str, hour: int, minute: int, repeat: str, weekday: int | None
) -> None:
    """Runs in a background thread: waits, runs the task, reschedules if repeating."""
    while True:
        target = next_run_time(hour, minute, weekday)
        print(f"Next run of {task!r}: {target:%Y-%m-%d %H:%M:%S}")

        while datetime.now() < target:
            time.sleep(15)

        run_task(task)

        if repeat == "once":
            break
        # "daily" or "weekly": loop back around and compute the next target


@tool
def schedule_task(
    task: str, hour: int, minute: int, repeat: str = "once", weekday: int | None = None
) -> str:
    """Schedule a task to run at a given time, optionally repeating.

    Args:
        task: what to ask the reminder sub-agent to do, e.g. "check the weather in NYC".
        hour: hour of day to run at (0-23).
        minute: minute of hour to run at (0-59).
        repeat: "once", "daily", or "weekly".
        weekday: only used when repeat="weekly". 0=Monday ... 6=Sunday.
    """
    thread = threading.Thread(
        target=scheduler_loop, args=(task, hour, minute, repeat, weekday), daemon=True
    )
    thread.start()
    return f"Scheduled '{task}' to run {repeat} at {hour:02d}:{minute:02d}."


# Main agent: the only tool it has is schedule_task. It hands off any
# reminder / to-do request to the scheduler, which runs it via reminder_agent.
main_agent = create_deep_agent(
    model=model,
    tools=[schedule_task],
    system_prompt=(
        "You schedule reminders and to-dos. When the user asks for something "
        "later, daily, or weekly, call schedule_task with the right hour, "
        "minute, and repeat value."
    ),
)


async def main():
    # Ask 2 minutes from now so we can see it fire in this demo. Note:
    # schedule_task only takes hour:minute (no seconds), so give it a
    # couple of minutes of buffer - by the time the LLM decides to call
    # the tool, a few seconds may have already passed.
    soon = datetime.now() + timedelta(minutes=2)
    prompt = (
        f"Check the weather in New York once, at {soon.hour}:{soon.minute:02d} "
        f"today (repeat='once')."
    )
    result = await main_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    print(result["messages"][-1].content)

    # Keep the script alive long enough to see the scheduled task fire.
    time.sleep(150)


if __name__ == "__main__":
    asyncio.run(main())
