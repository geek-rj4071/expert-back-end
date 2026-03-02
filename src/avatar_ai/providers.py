"""Provider interfaces and adapters for LLM, STT, and TTS."""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResult:
    text: str
    emotion: str


@dataclass(frozen=True)
class STTResult:
    text: str


@dataclass(frozen=True)
class TTSResult:
    audio_bytes: bytes
    mime_type: str


class LLMProvider(Protocol):
    def complete(self, *, persona: str, user_text: str) -> LLMResult: ...


class STTProvider(Protocol):
    def transcribe(self, *, audio_bytes: bytes, mime_type: str) -> STTResult: ...


class TTSProvider(Protocol):
    def synthesize(self, *, text: str, voice_id: str) -> TTSResult: ...


class DeterministicLLMProvider:
    """Rule-based provider for local and test usage."""

    def complete(self, *, persona: str, user_text: str) -> LLMResult:
        lowered = user_text.lower()
        if "sad" in lowered or "stressed" in lowered:
            return LLMResult(
                text=f"{persona}: I hear you. Let's take one small next step together.",
                emotion="empathetic",
            )
        if user_text.strip().endswith("?"):
            return LLMResult(
                text=f"{persona}: Great question. Here's a practical way to think about it.",
                emotion="curious",
            )
        return LLMResult(
            text=f"{persona}: Thanks for sharing. Want to go one level deeper?",
            emotion="neutral",
        )


class MockSTTProvider:
    """Simple STT mock that decodes UTF-8 text from provided bytes."""

    def transcribe(self, *, audio_bytes: bytes, mime_type: str) -> STTResult:
        del mime_type
        if not audio_bytes:
            return STTResult(text="")
        try:
            return STTResult(text=audio_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            return STTResult(text="[unintelligible audio]")


class MockTTSProvider:
    """Simple TTS mock that returns a deterministic byte payload."""

    def synthesize(self, *, text: str, voice_id: str) -> TTSResult:
        payload = f"VOICE={voice_id};TEXT={text}".encode("utf-8")
        return TTSResult(audio_bytes=payload, mime_type="audio/wav")


class SystemTTSProvider:
    """Offline system TTS using macOS `say` command."""

    _VOICE_MAP = {
        "alloy": "Alex",
        "sage": "Alex",
        "verse": "Alex",
    }

    def __init__(self, *, rate_wpm: int = 150) -> None:
        # macOS `say` is usually around 175 wpm by default; 150 is easier to follow.
        self.rate_wpm = max(90, min(260, int(rate_wpm)))

    def synthesize(self, *, text: str, voice_id: str) -> TTSResult:
        voice = self._VOICE_MAP.get(voice_id, "Alex")
        smooth_text = self._smooth_text(text)
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp_aiff:
            aiff_path = tmp_aiff.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        try:
            say_cmd = ["say", "-v", voice, "-r", str(self.rate_wpm), "-o", aiff_path, smooth_text]
            subprocess.run(
                say_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            mime = "audio/aiff"
            audio_path = aiff_path

            # Convert to WAV for best browser compatibility.
            if shutil.which("afconvert"):
                subprocess.run(
                    ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff_path, wav_path],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                mime = "audio/wav"
                audio_path = wav_path

            with open(audio_path, "rb") as f:
                audio = f.read()
        except FileNotFoundError as exc:
            raise RuntimeError("system_tts_error: say_not_found") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"system_tts_error: {exc.stderr.strip()}") from exc
        finally:
            try:
                os.remove(aiff_path)
            except OSError:
                pass
            try:
                os.remove(wav_path)
            except OSError:
                pass

        return TTSResult(audio_bytes=audio, mime_type=mime)

    def _smooth_text(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        clean = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", clean)
        clean = re.sub(r"\s*,\s*", ", ", clean)
        return clean


class OpenAIChatProvider:
    """Basic OpenAI-compatible chat provider over HTTP."""

    def __init__(self, *, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model

    def complete(self, *, persona: str, user_text: str) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"You are {persona}. Reply concisely."},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.4,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llm_provider_error: {exc}") from exc

        text = payload["choices"][0]["message"]["content"].strip()
        return LLMResult(text=text, emotion="neutral")


class GeminiChatProvider:
    """Google Gemini chat provider over HTTP."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-flash-latest",
        base_url: str = "https://generativelanguage.googleapis.com",
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.ssl_context = self._build_ssl_context()

    def complete(self, *, persona: str, user_text: str) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required")

        body = {
            "systemInstruction": {
                "parts": [
                    {"text": f"You are {persona}. Reply concisely."},
                ]
            },
            "contents": [
                {
                    "parts": [
                        {"text": user_text},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.4},
        }
        model_path = urllib.parse.quote(self.model, safe=":-._")
        req = urllib.request.Request(
            f"{self.base_url}/v1beta/models/{model_path}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "X-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30, context=self.ssl_context) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = ""
            fp = exc.fp
            try:
                if fp is not None:
                    body_text = fp.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                body_text = ""
            finally:
                try:
                    if fp is not None:
                        fp.close()
                except Exception:
                    pass
                try:
                    exc.close()
                except Exception:
                    pass
            detail = body_text or str(exc)
            raise RuntimeError(f"gemini_provider_error: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"gemini_provider_error: {exc}") from exc

        text = self._extract_text(payload)
        if not text:
            raise RuntimeError("gemini_provider_error: empty_response")
        return LLMResult(text=text, emotion="neutral")

    def _extract_text(self, payload: dict) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        collected: list[str] = []
        for part in parts:
            text = str(part.get("text", "")).strip()
            if text:
                collected.append(text)
        return "\n".join(collected).strip()

    def _build_ssl_context(self) -> ssl.SSLContext:
        ca_bundle = os.getenv("GEMINI_CA_BUNDLE", "").strip() or os.getenv("SSL_CERT_FILE", "").strip()
        if ca_bundle:
            try:
                return ssl.create_default_context(cafile=ca_bundle)
            except Exception as exc:
                raise RuntimeError(f"gemini_provider_error: invalid_ca_bundle: {ca_bundle}: {exc}") from exc

        try:
            import certifi  # type: ignore

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()


class OllamaChatProvider:
    """Local Ollama chat provider for free, self-hosted responses."""

    def __init__(self, *, base_url: str = "http://127.0.0.1:11434", model: str = "llama3.2:3b") -> None:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/api"):
            normalized = normalized[:-4]
        elif normalized.endswith("/v1"):
            normalized = normalized[:-3]
        self.base_url = normalized
        self.model = model

    def complete(self, *, persona: str, user_text: str) -> LLMResult:
        chat_body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": f"You are {persona}. Reply naturally and briefly."},
                {"role": "user", "content": user_text},
            ],
        }
        prompt = f"System: You are {persona}. Reply naturally and briefly.\nUser: {user_text}\nAssistant:"
        generate_body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        openai_body = {
            "model": self.model,
            "messages": chat_body["messages"],
            "temperature": 0.4,
        }

        # Try Ollama native chat, then native generate, then OpenAI-compatible endpoint.
        attempts = [
            ("/api/chat", chat_body, self._extract_chat_text),
            ("/api/generate", generate_body, self._extract_generate_text),
            ("/v1/chat/completions", openai_body, self._extract_openai_text),
        ]

        last_error: str | None = None
        last_path: str | None = None
        for path, body, extractor in attempts:
            try:
                last_path = path
                payload = self._post_json(path, body)
                text = extractor(payload)
                if text:
                    return LLMResult(text=text, emotion="neutral")
                last_error = "empty_response"
            except urllib.error.HTTPError as exc:
                body_text = ""
                fp = exc.fp
                try:
                    if fp is not None:
                        body_text = fp.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    body_text = ""
                finally:
                    try:
                        if fp is not None:
                            fp.close()
                    except Exception:
                        pass
                    try:
                        exc.close()
                    except Exception:
                        pass

                # If endpoint exists but model is missing, surface the model error directly.
                lowered = body_text.lower()
                if "model" in lowered and "not found" in lowered:
                    raise RuntimeError(f"ollama_provider_error: {body_text}") from exc

                # Try the next compatible endpoint on 404/405.
                if exc.code in (404, 405):
                    suffix = f" body={body_text}" if body_text else ""
                    last_error = f"http_{exc.code}{suffix}"
                    continue
                raise RuntimeError(f"ollama_provider_error: {exc}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"ollama_provider_error: {exc}") from exc

        if last_error:
            where = f" endpoint={last_path}" if last_path else ""
            raise RuntimeError(f"ollama_provider_error: {last_error}{where}")
        raise RuntimeError("ollama_provider_error: unsupported_endpoint")

    def _post_json(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _extract_chat_text(self, payload: dict) -> str:
        return str(payload.get("message", {}).get("content", "")).strip()

    def _extract_generate_text(self, payload: dict) -> str:
        return str(payload.get("response", "")).strip()

    def _extract_openai_text(self, payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content", "")).strip()


class OpenAITTSProvider:
    """Basic OpenAI-compatible TTS provider over HTTP."""

    def __init__(self, *, api_key: str | None = None, model: str = "gpt-4o-mini-tts") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model

    def synthesize(self, *, text: str, voice_id: str) -> TTSResult:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        body = {
            "model": self.model,
            "voice": voice_id,
            "input": text,
            "format": "wav",
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                audio = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"tts_provider_error: {exc}") from exc

        return TTSResult(audio_bytes=audio, mime_type="audio/wav")


class OpenAIWhisperSTTProvider:
    """STT adapter placeholder that expects base64 audio text payload."""

    def __init__(self, *, api_key: str | None = None, model: str = "gpt-4o-mini-transcribe") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model

    def transcribe(self, *, audio_bytes: bytes, mime_type: str) -> STTResult:
        del mime_type
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        # Uses JSON-based fallback to avoid multipart complexity in stdlib-only environment.
        body = {
            "model": self.model,
            "audio": base64.b64encode(audio_bytes).decode("ascii"),
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"stt_provider_error: {exc}") from exc

        return STTResult(text=str(payload.get("text", "")).strip())
