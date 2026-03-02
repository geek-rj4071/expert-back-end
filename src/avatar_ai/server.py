"""Local server entrypoint."""
from __future__ import annotations

import os
from pathlib import Path
from wsgiref.simple_server import make_server

from .api import AvatarAPI
from .models import Avatar
from .persistent_service import PersistentChatService, PersistentServiceConfig
from .persistence import SQLiteRepository
from .providers import (
    DeterministicLLMProvider,
    GeminiChatProvider,
    MockTTSProvider,
    OllamaChatProvider,
    OpenAIChatProvider,
    SystemTTSProvider,
)


def _configure_ssl_cert_file() -> None:
    if str(os.getenv("SSL_CERT_FILE", "")).strip():
        return
    try:
        import certifi  # type: ignore

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except Exception:
        # Keep startup resilient if certifi is unavailable.
        pass


def _load_backend_secret_env(var_name: str, secret_filename: str) -> None:
    existing = str(os.getenv(var_name, "")).strip()
    if existing:
        return

    secret_file_env = str(os.getenv(f"{var_name}_FILE", "")).strip()
    candidate_paths: list[Path] = []
    if secret_file_env:
        candidate_paths.append(Path(secret_file_env))
    candidate_paths.append(Path(__file__).resolve().parents[2] / ".secrets" / secret_filename)
    candidate_paths.append(Path("/app/secrets") / secret_filename)

    for path in candidate_paths:
        try:
            if not path.exists() or not path.is_file():
                continue
            value = path.read_text(encoding="utf-8", errors="ignore").strip()
            if value:
                os.environ[var_name] = value
                return
        except Exception:
            continue


def create_app(db_path: str | None = None):
    _configure_ssl_cert_file()
    _load_backend_secret_env("GEMINI_API_KEY", "gemini_api_key")
    resolved_db_path = db_path or os.getenv("AVATAR_DB_PATH", "/tmp/avatar_ai.db")
    repo = SQLiteRepository(resolved_db_path)
    provider_name = os.getenv("AI_PROVIDER", "openai").strip().lower()
    tts_provider_name = os.getenv("TTS_PROVIDER", "system").strip().lower()
    llm_fallback_enabled = os.getenv("LLM_FALLBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    try:
        system_tts_rate_wpm = int(os.getenv("SYSTEM_TTS_RATE_WPM", "150"))
    except ValueError:
        system_tts_rate_wpm = 150
    try:
        embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "256"))
    except ValueError:
        embedding_dimension = 256
    try:
        chunk_chars = int(os.getenv("TRAINING_CHUNK_CHARS", "900"))
    except ValueError:
        chunk_chars = 900
    try:
        chunk_overlap_chars = int(os.getenv("TRAINING_CHUNK_OVERLAP", "130"))
    except ValueError:
        chunk_overlap_chars = 130
    try:
        max_message_chars = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))
    except ValueError:
        max_message_chars = 4000
    try:
        max_tts_chars = int(os.getenv("MAX_TTS_CHARS", "12000"))
    except ValueError:
        max_tts_chars = 12000
    try:
        max_training_file_bytes = int(os.getenv("MAX_TRAINING_FILE_BYTES", "8000000"))
    except ValueError:
        max_training_file_bytes = 8000000
    try:
        max_training_docs_per_avatar = int(os.getenv("MAX_TRAINING_DOCS_PER_AVATAR", "40"))
    except ValueError:
        max_training_docs_per_avatar = 40
    try:
        max_student_image_file_bytes = int(os.getenv("MAX_STUDENT_IMAGE_FILE_BYTES", "6000000"))
    except ValueError:
        max_student_image_file_bytes = 6000000
    try:
        ocr_max_pages = int(os.getenv("OCR_MAX_PAGES", "15"))
    except ValueError:
        ocr_max_pages = 15
    internet_lookup_enabled = os.getenv("INTERNET_LOOKUP_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
    strict_book_only_mode = os.getenv("STRICT_BOOK_ONLY_MODE", "true").strip().lower() in {"1", "true", "yes"}
    ocr_enabled = os.getenv("OCR_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    ocr_language = os.getenv("OCR_LANGUAGE", "eng")

    service_config = PersistentServiceConfig(
        max_message_chars=max(500, max_message_chars),
        max_tts_chars=max(1000, max_tts_chars),
        max_training_file_bytes=max(1_000_000, max_training_file_bytes),
        max_training_docs_per_avatar=max(1, max_training_docs_per_avatar),
        max_student_image_file_bytes=max(300_000, max_student_image_file_bytes),
        internet_lookup_enabled=internet_lookup_enabled,
        strict_book_only_mode=strict_book_only_mode,
        chunk_chars=max(300, chunk_chars),
        chunk_overlap_chars=max(40, chunk_overlap_chars),
        embedding_dimension=max(64, embedding_dimension),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "auto"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
        ocr_enabled=ocr_enabled,
        ocr_max_pages=max(1, ocr_max_pages),
        ocr_language=ocr_language.strip() or "eng",
    )

    if provider_name == "openai":
        llm_provider = OpenAIChatProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    elif provider_name == "gemini":
        llm_provider = GeminiChatProvider(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
        )
    elif provider_name == "ollama":
        llm_provider = OllamaChatProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        )
    else:
        llm_provider = DeterministicLLMProvider()

    if tts_provider_name == "mock":
        tts_provider = MockTTSProvider()
    else:
        tts_provider = SystemTTSProvider(rate_wpm=system_tts_rate_wpm)

    service = PersistentChatService(
        repository=repo,
        config=service_config,
        llm_provider=llm_provider,
        tts_provider=tts_provider,
        llm_fallback_enabled=llm_fallback_enabled,
        ai_provider_name=provider_name,
        tts_provider_name=tts_provider_name,
    )
    service.seed_avatars(
        [
            Avatar(id="av_coach", name="Coach Ava", persona_prompt="Coach Ava", voice_id="alloy"),
            Avatar(id="av_tutor", name="Tutor Leo", persona_prompt="Tutor Leo", voice_id="sage"),
            Avatar(id="av_friend", name="Friend Nia", persona_prompt="Friend Nia", voice_id="verse"),
        ]
    )
    return AvatarAPI(service).app


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"
    try:
        port = int(os.getenv("PORT", "8000"))
    except ValueError:
        port = 8000
    app = create_app()
    with make_server(host, port, app) as server:
        provider_name = os.getenv("AI_PROVIDER", "openai").strip().lower()
        tts_provider_name = os.getenv("TTS_PROVIDER", "system").strip().lower()
        llm_fallback_enabled = os.getenv("LLM_FALLBACK_ENABLED", "true").strip().lower()
        print(
            f"Serving on http://{host}:{port} (AI_PROVIDER={provider_name}, "
            f"TTS_PROVIDER={tts_provider_name}, SYSTEM_TTS_RATE_WPM={os.getenv('SYSTEM_TTS_RATE_WPM', '150')}, "
            f"MAX_MESSAGE_CHARS={os.getenv('MAX_MESSAGE_CHARS', '4000')}, "
            f"MAX_TTS_CHARS={os.getenv('MAX_TTS_CHARS', '12000')}, "
            f"MAX_TRAINING_FILE_BYTES={os.getenv('MAX_TRAINING_FILE_BYTES', '8000000')}, "
            f"MAX_TRAINING_DOCS_PER_AVATAR={os.getenv('MAX_TRAINING_DOCS_PER_AVATAR', '40')}, "
            f"MAX_STUDENT_IMAGE_FILE_BYTES={os.getenv('MAX_STUDENT_IMAGE_FILE_BYTES', '6000000')}, "
            f"OCR_ENABLED={os.getenv('OCR_ENABLED', 'true')}, "
            f"OCR_MAX_PAGES={os.getenv('OCR_MAX_PAGES', '15')}, "
            f"SERVE_WEB_UI={os.getenv('SERVE_WEB_UI', 'true')}, "
            f"CORS_ALLOW_ORIGIN={os.getenv('CORS_ALLOW_ORIGIN', '*')}, "
            f"INTERNET_LOOKUP_ENABLED={os.getenv('INTERNET_LOOKUP_ENABLED', 'false')}, "
            f"STRICT_BOOK_ONLY_MODE={os.getenv('STRICT_BOOK_ONLY_MODE', 'true')}, "
            f"LLM_FALLBACK_ENABLED={llm_fallback_enabled})"
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
