import json
import os
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from avatar_ai.providers import (
    FallbackTTSProvider,
    GeminiChatProvider,
    GeminiTTSProvider,
    MockTTSProvider,
    OllamaChatProvider,
    SystemTTSProvider,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class OllamaChatProviderTests(unittest.TestCase):
    def test_complete_returns_text(self) -> None:
        provider = OllamaChatProvider(base_url="http://localhost:11434", model="llama3.2:3b")

        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse({"message": {"content": "hello there"}})):
            out = provider.complete(persona="Coach Ava", user_text="Hi")

        self.assertEqual(out.text, "hello there")
        self.assertEqual(out.emotion, "neutral")

    def test_complete_raises_on_empty_payload(self) -> None:
        provider = OllamaChatProvider()

        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse({"message": {"content": ""}})):
            with self.assertRaises(RuntimeError):
                provider.complete(persona="Coach Ava", user_text="Hi")

    def test_complete_falls_back_to_generate_when_chat_404(self) -> None:
        provider = OllamaChatProvider(base_url="http://localhost:11434", model="llama3.2:3b")

        def fake_urlopen(req, timeout):
            del timeout
            url = req.full_url
            if url.endswith("/api/chat"):
                raise HTTPError(url=url, code=404, msg="Not Found", hdrs=None, fp=None)
            if url.endswith("/api/generate"):
                return _FakeHTTPResponse({"response": "fallback generate reply"})
            raise AssertionError(f"unexpected url {url}")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = provider.complete(persona="Coach Ava", user_text="Hi")

        self.assertEqual(out.text, "fallback generate reply")
        self.assertEqual(out.emotion, "neutral")

    def test_complete_falls_back_to_openai_compatible_when_native_missing(self) -> None:
        provider = OllamaChatProvider(base_url="http://localhost:11434", model="llama3.2:3b")

        def fake_urlopen(req, timeout):
            del timeout
            url = req.full_url
            if url.endswith("/api/chat") or url.endswith("/api/generate"):
                raise HTTPError(url=url, code=404, msg="Not Found", hdrs=None, fp=None)
            if url.endswith("/v1/chat/completions"):
                return _FakeHTTPResponse({"choices": [{"message": {"content": "openai compat reply"}}]})
            raise AssertionError(f"unexpected url {url}")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = provider.complete(persona="Coach Ava", user_text="Hi")

        self.assertEqual(out.text, "openai compat reply")
        self.assertEqual(out.emotion, "neutral")

    def test_base_url_normalization_strips_api_and_v1_suffix(self) -> None:
        self.assertEqual(OllamaChatProvider(base_url="http://localhost:11434/api").base_url, "http://localhost:11434")
        self.assertEqual(OllamaChatProvider(base_url="http://localhost:11434/v1").base_url, "http://localhost:11434")


class GeminiChatProviderTests(unittest.TestCase):
    def test_complete_returns_text(self) -> None:
        provider = GeminiChatProvider(api_key="k", model="gemini-flash-latest")

        with patch(
            "urllib.request.urlopen",
            return_value=_FakeHTTPResponse({"candidates": [{"content": {"parts": [{"text": "hello from gemini"}]}}]}),
        ):
            out = provider.complete(persona="SP Sir", user_text="Hi")

        self.assertEqual(out.text, "hello from gemini")
        self.assertEqual(out.emotion, "neutral")

    def test_complete_joins_multiple_parts(self) -> None:
        provider = GeminiChatProvider(api_key="k", model="gemini-flash-latest")

        with patch(
            "urllib.request.urlopen",
            return_value=_FakeHTTPResponse(
                {"candidates": [{"content": {"parts": [{"text": "line one"}, {"text": "line two"}]}}]}
            ),
        ):
            out = provider.complete(persona="SP Sir", user_text="Hi")

        self.assertEqual(out.text, "line one\nline two")

    def test_complete_requires_api_key(self) -> None:
        provider = GeminiChatProvider(api_key="")
        with self.assertRaises(RuntimeError):
            provider.complete(persona="SP Sir", user_text="Hi")

    def test_complete_surfaces_http_error_body(self) -> None:
        provider = GeminiChatProvider(api_key="k", model="gemini-flash-latest")

        def fake_urlopen(req, timeout, context=None):
            del req, timeout, context
            raise HTTPError(
                url="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
                code=404,
                msg="model not found",
                hdrs=None,
                fp=None,
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError) as err:
                provider.complete(persona="SP Sir", user_text="Hi")

        self.assertIn("gemini_provider_error", str(err.exception))
        self.assertIn("model not found", str(err.exception))

    def test_invalid_ca_bundle_fails_fast(self) -> None:
        with patch.dict(os.environ, {"GEMINI_CA_BUNDLE": "/tmp/missing-ca-bundle.pem"}, clear=False):
            with self.assertRaises(RuntimeError) as err:
                GeminiChatProvider(api_key="k", model="gemini-flash-latest")
        self.assertIn("invalid_ca_bundle", str(err.exception))


class SystemTTSProviderTests(unittest.TestCase):
    def test_synthesize_returns_audio_bytes(self) -> None:
        provider = SystemTTSProvider()

        with tempfile.TemporaryDirectory() as tmp:
            out_aiff = os.path.join(tmp, "tts.aiff")
            out_wav = os.path.join(tmp, "tts.wav")

            def fake_run(cmd, check, capture_output, text):
                del check, capture_output, text
                if cmd[0] == "say":
                    self.assertIn("-r", cmd)
                    out_idx = cmd.index("-o") + 1
                    path = cmd[out_idx]
                    with open(path, "wb") as f:
                        f.write(b"FAKE_AIFF")
                elif cmd[0] == "afconvert":
                    path = cmd[-1]
                    with open(path, "wb") as f:
                        f.write(b"FAKE_WAV")

            with patch("subprocess.run", side_effect=fake_run):
                with patch("tempfile.NamedTemporaryFile") as named_tmp:
                    class _Tmp:
                        def __init__(self, name):
                            self.name = name
                        def __enter__(self):
                            return self
                        def __exit__(self, exc_type, exc, tb):
                            return False
                    named_tmp.side_effect = [_Tmp(out_aiff), _Tmp(out_wav)]
                    with patch("shutil.which", return_value="/usr/bin/afconvert"):
                        out = provider.synthesize(text="hello", voice_id="alloy")

        self.assertEqual(out.mime_type, "audio/wav")
        self.assertEqual(out.audio_bytes, b"FAKE_WAV")


class GeminiTTSProviderTests(unittest.TestCase):
    def test_synthesize_requires_api_key(self) -> None:
        provider = GeminiTTSProvider(api_key="")
        with self.assertRaises(RuntimeError):
            provider.synthesize(text="hello", voice_id="alloy")

    def test_synthesize_returns_audio(self) -> None:
        provider = GeminiTTSProvider(api_key="k", model="gemini-2.5-flash-preview-tts", voice_name="Kore")
        audio_b64 = "RkFLRV9BVURJTw=="
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/wav",
                                    "data": audio_b64,
                                }
                            }
                        ]
                    }
                }
            ]
        }
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(payload)):
            out = provider.synthesize(text="speak this", voice_id="alloy")
        self.assertEqual(out.mime_type, "audio/wav")
        self.assertEqual(out.audio_bytes, b"FAKE_AUDIO")

    def test_synthesize_surfaces_http_errors(self) -> None:
        provider = GeminiTTSProvider(api_key="k", model="gemini-2.5-flash-preview-tts")

        def fake_urlopen(req, timeout, context=None):
            del req, timeout, context
            raise HTTPError(
                url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent",
                code=404,
                msg="not found",
                hdrs=None,
                fp=None,
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError) as err:
                provider.synthesize(text="hello", voice_id="alloy")
        self.assertIn("gemini_tts_provider_error", str(err.exception))


class FallbackTTSProviderTests(unittest.TestCase):
    def test_fallback_uses_secondary_when_primary_fails(self) -> None:
        class _FailingTTS:
            def synthesize(self, *, text: str, voice_id: str):
                del text, voice_id
                raise RuntimeError("primary_failed")

        fallback = MockTTSProvider()
        provider = FallbackTTSProvider(primary=_FailingTTS(), fallback=fallback)
        out = provider.synthesize(text="hello", voice_id="alloy")
        self.assertEqual(out.mime_type, "audio/wav")
        self.assertIn(b"VOICE=alloy", out.audio_bytes)


if __name__ == "__main__":
    unittest.main()
