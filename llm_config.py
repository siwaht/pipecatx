from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from loguru import logger
import os

# This module reads os.environ at import time, and it gets imported before the
# load_dotenv() calls in raga.py / agent.py run. So load .env here rather than
# relying on an importer to have done it first.
load_dotenv()

#####################################################
# Cloudflare Workers AI — OpenAI-compatible endpoint.
# Credentials live in .env (gitignored); never hardcode them here.
#
# Missing credentials must not kill the process at import time: on Render that
# turns into a failed/timed-out deploy with no HTTP server ever listening. Warn
# loudly instead and let the health endpoint stay up; the first actual LLM call
# is what will fail.
ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")

_missing = [
    name
    for name, value in (
        ("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID),
        ("CLOUDFLARE_API_TOKEN", API_TOKEN),
    )
    if not value
]
if _missing:
    logger.warning(
        "Missing environment variable(s): {}. The web server will still start, "
        "but LLM calls will fail until these are set.",
        ", ".join(_missing),
    )


model = ChatOpenAI(
    model="@cf/deepseek-ai/deepseek-v4-pro-0813",
    base_url=(
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{ACCOUNT_ID or 'missing-account-id'}/ai/v1"
    ),
    api_key=API_TOKEN or "missing-api-token",
)
######################################################################