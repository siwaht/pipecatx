# PipecatX — Pipecat voice agent

Real-time voice agent: Deepgram STT → LangChain DeepAgent on Cloudflare Workers AI (DeepSeek v4 Pro) → ElevenLabs TTS, with Pipecat over LiveKit as the browser media transport.

## Stack
- Python >= 3.13 (system Python via the `python-base-3.13` module), managed with `uv`
- FastAPI server from `pipecat.runner.run` (`app` object)
- Dependencies: `pyproject.toml`, `requirements.txt`, `uv.lock` — `pipecat-ai[deepgram,elevenlabs,langchain,livekit,runner,silero,webrtc]`

## Running
- Workflow `Start application`: `PORT=5000 uv run python main.py` (webview, port 5000)
- `main.py` binds `0.0.0.0:$PORT` when `PORT` is set
- Endpoints: `GET /healthz`, `GET /livekit` (browser client), `POST /livekit/connect` (creates a LiveKit room + tokens and starts the bot task)

## Secrets (Replit Secrets, never hardcode)
Required for a full voice session: `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`, `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`. The server starts without them; `/livekit/connect` returns 503 until LiveKit secrets are set.

## Notes
- User turn stop uses `SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.4)` — do NOT switch to SmartTurn (it added ~3s of end-of-turn delay).
- Do NOT use Pipecat's Small WebRTC browser transport in Replit Preview; ICE/media does not work behind the proxy. LiveKit only.
- `llm_config.py` reads Cloudflare credentials at import time and only warns when missing, so the health endpoint stays up.
- Environment quirk: uv is configured system-wide with `UV_PYTHON_DOWNLOADS=never` / `UV_PYTHON_PREFERENCE=only-system`, and uv's managed CPython 3.13.9 is broken here (missing `_struct`). The `.pythonlibs` venv must be built on the system Python (`uv sync --python "$(which python3)"` after removing `.pythonlibs`).
