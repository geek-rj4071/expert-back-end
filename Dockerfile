FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8000 \
    SERVE_WEB_UI=false \
    AI_PROVIDER=deterministic \
    TTS_PROVIDER=mock \
    AVATAR_DB_PATH=/data/avatar_ai.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        poppler-utils \
        tesseract-ocr \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Reliable TLS CA bundle for OpenAI/Gemini calls.
RUN python -m pip install --no-cache-dir --upgrade pip certifi

COPY src ./src

RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "avatar_ai.server"]
