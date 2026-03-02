# Expert Back-End (SP Sir)

Python WSGI backend for SP Sir avatar service.

## Prerequisites

- Python 3.11+ (3.12 recommended)
- `pip`
- Optional for OCR-heavy PDFs/images:
  - `tesseract`
  - `poppler-utils` (`pdftotext`, `pdftoppm`)

## Project Structure

- `src/avatar_ai/` - backend source code
- `tests/` - unit/integration tests
- `Dockerfile` - container image definition
- `.env.example` - environment variable template

## Local Setup

1. Create virtual environment and install runtime deps (if you use provider extras, install those too):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip certifi
```

2. Create env file:

```bash
cp .env.example .env
```

3. (Optional) add Gemini key via secret file:

```bash
mkdir -p .secrets
echo "YOUR_GEMINI_API_KEY" > .secrets/gemini_api_key
```

4. Run server:

```bash
PYTHONPATH=src python3 -m avatar_ai.server
```

Server default: `http://127.0.0.1:8000`

## Running Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
```

## Docker

Build image:

```bash
docker build -t expert-back-end:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e AVATAR_DB_PATH=/data/avatar_ai.db \
  -v $(pwd)/.secrets:/app/.secrets:ro \
  -v expert_backend_data:/data \
  expert-back-end:latest
```

## Auto Deploy On Every Push

This repo includes GitHub Actions workflow:

- `.github/workflows/docker-cicd.yml`

On each push to `main`:

1. Build Docker image
2. Push image to GHCR:
   - `ghcr.io/<github-user-or-org>/expert-back-end:latest`
   - `ghcr.io/<github-user-or-org>/expert-back-end:sha-<commit>`
3. Optional server deploy over SSH (if secrets are configured)

### Required GitHub Secrets (for auto-deploy job)

- `DEPLOY_HOST`
- `DEPLOY_USER`
- `DEPLOY_SSH_KEY`
- `DEPLOY_PATH` (directory on server containing your `docker-compose.yml`)

If these are not set, image build/push still runs, deploy job is skipped.

## Key Environment Variables

- `AI_PROVIDER`: `deterministic` | `openai` | `gemini` | `ollama`
- `TTS_PROVIDER`: `mock` | `system`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY` or `GEMINI_API_KEY_FILE`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `MAX_TRAINING_FILE_BYTES`
- `MAX_STUDENT_IMAGE_FILE_BYTES`
- `OCR_ENABLED`, `OCR_MAX_PAGES`, `OCR_LANGUAGE`
- `CORS_ALLOW_ORIGIN`

## Main API Prefix

All routes are under:

`/avatar-service`

Examples:

- `GET /avatar-service/system/status`
- `GET /avatar-service/ai/health`
- `POST /avatar-service/training/upload`
- `POST /avatar-service/conversations/{conversationId}/messages`
- `POST /avatar-service/conversations/{conversationId}/image`

## Common Issues

- `413 Request Entity Too Large`: increase proxy upload limit (nginx) and/or backend file-size limits.
- `insufficient_readable_text`: OCR could not extract text; verify file quality and OCR tools.
- `image_ocr_unavailable`: install OCR tools or configure Gemini API key for vision fallback.
- `OPENAI_API_KEY is required`: set `AI_PROVIDER` and matching API key correctly.
