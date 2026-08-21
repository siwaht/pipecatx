# DigitalOcean App Platform builds this from the connected GitHub repo.
#
# A Dockerfile rather than App Platform's Python buildpack, because this project
# requires Python >= 3.13 (see pyproject.toml) and aiortc links against system
# media libraries. Pinning both here keeps the build reproducible.

FROM python:3.13-slim

# libopus / libvpx / libsrtp are what aiortc links against for WebRTC media.
# They are also needed for the LiveKit transport's audio handling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libopus0 \
        libvpx-dev \
        libsrtp2-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so App Platform can reuse the layer when only code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# App Platform injects PORT and expects the process to bind 0.0.0.0 on it.
# main.py reads PORT and injects --host 0.0.0.0 --port $PORT for the runner.
ENV PORT=8080
EXPOSE 8080

CMD ["python", "main.py"]
