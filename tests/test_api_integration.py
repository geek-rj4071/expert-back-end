import base64
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from avatar_ai.api import AvatarAPI
from avatar_ai.models import Avatar
from avatar_ai.persistent_service import PersistentChatService
from avatar_ai.persistence import SQLiteRepository


class APIRoutesIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        db_path = str(Path(cls.tmp.name) / "api.db")
        repo = SQLiteRepository(db_path)
        service = PersistentChatService(repository=repo)
        service.seed_avatars([
            Avatar(id="av_coach", name="Coach Ava", persona_prompt="Coach Ava", voice_id="alloy"),
        ])
        cls.service = service
        cls.app = AvatarAPI(service).app

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _request(self, method: str, path: str, *, payload=None, headers=None):
        body_bytes = json.dumps(payload).encode("utf-8") if payload is not None else b""
        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "CONTENT_LENGTH": str(len(body_bytes)),
            "CONTENT_TYPE": "application/json",
            "wsgi.input": io.BytesIO(body_bytes),
        }
        for key, value in (headers or {}).items():
            env_key = f"HTTP_{key.upper().replace('-', '_')}"
            environ[env_key] = value

        response_state = {}

        def start_response(status, response_headers):
            response_state["status"] = status
            response_state["headers"] = response_headers

        chunks = self.app(environ, start_response)
        raw = b"".join(chunks).decode("utf-8")
        status_code = int(response_state["status"].split()[0])
        return status_code, json.loads(raw)

    def _request_bytes(self, method: str, path: str, *, body_bytes: bytes, content_type: str, headers=None):
        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "CONTENT_LENGTH": str(len(body_bytes)),
            "CONTENT_TYPE": content_type,
            "wsgi.input": io.BytesIO(body_bytes),
        }
        for key, value in (headers or {}).items():
            env_key = f"HTTP_{key.upper().replace('-', '_')}"
            environ[env_key] = value

        response_state = {}

        def start_response(status, response_headers):
            response_state["status"] = status
            response_state["headers"] = response_headers

        chunks = self.app(environ, start_response)
        raw = b"".join(chunks).decode("utf-8")
        status_code = int(response_state["status"].split()[0])
        return status_code, json.loads(raw)

    def _request_raw(self, method: str, path: str, *, query: str = "", headers=None):
        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "QUERY_STRING": query,
            "CONTENT_LENGTH": "0",
            "wsgi.input": io.BytesIO(b""),
        }
        for key, value in (headers or {}).items():
            env_key = f"HTTP_{key.upper().replace('-', '_')}"
            environ[env_key] = value
        response_state = {}

        def start_response(status, response_headers):
            response_state["status"] = status
            response_state["headers"] = response_headers

        chunks = self.app(environ, start_response)
        raw = b"".join(chunks)
        status_code = int(response_state["status"].split()[0])
        return status_code, raw, dict(response_state["headers"])

    def test_full_chat_flow_and_voice_routes(self) -> None:
        status, user = self._request("POST", "/avatar-service/auth/signup", payload={"email": "api@example.com"})
        self.assertEqual(status, 201)
        user_id = user["id"]

        status, avatars = self._request("GET", "/avatar-service/avatars")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(avatars), 1)

        status, convo = self._request(
            "POST",
            "/avatar-service/conversations",
            payload={"avatarId": "av_coach"},
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 201)
        conversation_id = convo["id"]

        status, turn = self._request(
            "POST",
            f"/avatar-service/conversations/{conversation_id}/messages",
            payload={"text": "Can you help me focus?"},
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 201)
        self.assertEqual(turn["assistantMessage"]["role"], "assistant")

        status, convo_loaded = self._request(
            "GET",
            f"/avatar-service/conversations/{conversation_id}",
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(convo_loaded["messages"]), 2)

        audio_source = base64.b64encode(b"hello from audio").decode("ascii")
        status, stt = self._request(
            "POST",
            "/avatar-service/voice/stt",
            payload={"audioBase64": audio_source, "mimeType": "audio/wav"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(stt["text"], "hello from audio")

        status, tts = self._request(
            "POST",
            "/avatar-service/voice/tts",
            payload={"text": "hello world", "voiceId": "alloy"},
        )
        self.assertEqual(status, 200)
        decoded = base64.b64decode(tts["audioBase64"])
        self.assertIn(b"VOICE=alloy", decoded)

    def test_voice_tts_accepts_long_assistant_text(self) -> None:
        long_text = ("This is a detailed explanation from SP Sir. " * 500).strip()
        status, tts = self._request(
            "POST",
            "/avatar-service/voice/tts",
            payload={"text": long_text, "voiceId": "alloy"},
        )
        self.assertEqual(status, 200)
        decoded = base64.b64decode(tts["audioBase64"])
        self.assertIn(b"VOICE=alloy", decoded)

    def test_messages_accept_long_student_question(self) -> None:
        status, user = self._request("POST", "/avatar-service/auth/signup", payload={"email": "long-msg@example.com"})
        self.assertEqual(status, 201)
        user_id = user["id"]

        status, convo = self._request(
            "POST",
            "/avatar-service/conversations",
            payload={"avatarId": "av_coach"},
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 201)
        long_question = ("Please explain the full chapter summary with examples. " * 40).strip()

        status, turn = self._request(
            "POST",
            f"/avatar-service/conversations/{convo['id']}/messages",
            payload={"text": long_question},
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 201)
        self.assertEqual(turn["assistantMessage"]["role"], "assistant")

    def test_static_ui_assets_are_served(self) -> None:
        status, html, headers = self._request_raw("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertTrue(b"AvatarTalk Call" in html or b"SP Sir Call" in html)

        status, css, headers = self._request_raw("GET", "/assets/styles.css")
        if status == 200:
            self.assertEqual(headers["Content-Type"], "text/css; charset=utf-8")
            self.assertIn(b":root", css)
        else:
            css_match = re.search(rb"/assets/[^\"'\s>]+\.css", html)
            self.assertIsNotNone(css_match)
            status, css, headers = self._request_raw("GET", css_match.group(0).decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "text/css; charset=utf-8")
            self.assertGreater(len(css), 20)

        status, js, headers = self._request_raw("GET", "/assets/app.js")
        if status == 200:
            self.assertEqual(headers["Content-Type"], "application/javascript; charset=utf-8")
            self.assertIn(b"const state", js)
        else:
            js_match = re.search(rb"/assets/[^\"'\s>]+\.js", html)
            self.assertIsNotNone(js_match)
            status, js, headers = self._request_raw("GET", js_match.group(0).decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "application/javascript; charset=utf-8")
            self.assertGreater(len(js), 40)

    def test_sse_streaming_route(self) -> None:
        status, user = self._request("POST", "/avatar-service/auth/signup", payload={"email": "sse@example.com"})
        self.assertEqual(status, 201)
        user_id = user["id"]

        status, convo = self._request(
            "POST",
            "/avatar-service/conversations",
            payload={"avatarId": "av_coach"},
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 201)

        text = quote("How should I begin?")
        query = f"userId={quote(user_id)}&conversationId={quote(convo['id'])}&text={text}"
        status, raw, headers = self._request_raw("GET", "/avatar-service/realtime/sse", query=query)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertIn(b"event: assistant.delta", raw)
        self.assertIn(b"event: assistant.final", raw)

    def test_sse_auto_heals_stale_conversation_id(self) -> None:
        status, user = self._request("POST", "/avatar-service/auth/signup", payload={"email": "heal@example.com"})
        self.assertEqual(status, 201)
        user_id = user["id"]

        query = f"userId={quote(user_id)}&conversationId=cnv_missing123&text={quote('hello')}"
        status, raw, headers = self._request_raw("GET", "/avatar-service/realtime/sse", query=query)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertIn(b"event: assistant.final", raw)

    def test_messages_auto_heals_stale_conversation_id(self) -> None:
        status, user = self._request("POST", "/avatar-service/auth/signup", payload={"email": "heal-msg@example.com"})
        self.assertEqual(status, 201)
        user_id = user["id"]

        status, payload = self._request(
            "POST",
            "/avatar-service/conversations/cnv_missing456/messages",
            payload={"text": "hello"},
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 201)
        self.assertIn("conversationId", payload)
        self.assertEqual(payload["assistantMessage"]["role"], "assistant")

    def test_voice_health_endpoint(self) -> None:
        status, payload = self._request("GET", "/avatar-service/voice/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertGreater(payload["bytes"], 0)
        self.assertIn(payload["mimeType"], {"audio/wav", "audio/aiff"})

    def test_ai_health_and_system_status_endpoints(self) -> None:
        status, payload = self._request("GET", "/avatar-service/ai/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertIn("provider", payload)

        status, payload = self._request("GET", "/avatar-service/system/status")
        self.assertEqual(status, 200)
        self.assertIn("aiProvider", payload)
        self.assertIn("ttsProvider", payload)

    def test_api_options_preflight_returns_cors_headers(self) -> None:
        status, raw, headers = self._request_raw("OPTIONS", "/avatar-service/ai/health")
        del raw
        self.assertEqual(status, 204)
        self.assertIn("Access-Control-Allow-Origin", headers)
        self.assertIn("Access-Control-Allow-Methods", headers)
        self.assertIn("Access-Control-Allow-Headers", headers)

    def test_training_routes_upload_status_list_and_clear(self) -> None:
        status, user = self._request("POST", "/avatar-service/auth/signup", payload={"email": "train-api@example.com"})
        self.assertEqual(status, 201)
        user_id = user["id"]

        book_text = (
            b"Quadratic equation has the form ax^2 + bx + c = 0. "
            b"For matric level, solve by factorization or quadratic formula. "
            b"This chapter includes multiple solved examples and exercises."
        )
        status, upload = self._request(
            "POST",
            "/avatar-service/training/upload",
            payload={
                "avatarId": "av_coach",
                "filename": "math-book.txt",
                "fileBase64": base64.b64encode(book_text).decode("ascii"),
            },
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 201)
        self.assertEqual(upload["filename"], "math-book.txt")
        self.assertGreaterEqual(upload["chunksIndexed"], 1)
        self.assertGreaterEqual(upload["embeddingsIndexed"], 1)

        status, raw, _ = self._request_raw(
            "GET",
            "/avatar-service/training/status",
            query="avatarId=av_coach",
        )
        self.assertEqual(status, 200)
        status_payload = json.loads(raw.decode("utf-8"))
        self.assertGreaterEqual(status_payload["documents"], 1)
        self.assertGreaterEqual(status_payload["vectors"], 1)
        self.assertTrue(status_payload["teacherMode"])

        status, raw, _ = self._request_raw(
            "GET",
            "/avatar-service/training/documents",
            query="avatarId=av_coach",
        )
        self.assertEqual(status, 200)
        docs = json.loads(raw.decode("utf-8"))
        self.assertGreaterEqual(len(docs), 1)
        self.assertTrue(any(doc.get("filename") == "math-book.txt" for doc in docs))

        status, raw, _ = self._request_raw(
            "DELETE",
            "/avatar-service/training/documents",
            query="avatarId=av_coach",
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 200)
        cleared = json.loads(raw.decode("utf-8"))
        self.assertGreaterEqual(cleared["deleted"], 1)

    def test_training_upload_accepts_multipart_form_data(self) -> None:
        status, user = self._request(
            "POST",
            "/avatar-service/auth/signup",
            payload={"email": "train-multipart@example.com"},
        )
        self.assertEqual(status, 201)
        user_id = user["id"]

        boundary = "----AvatarBoundary7MA4YWxkTrZu0gW"
        file_bytes = (
            b"Human brain is a central organ of the nervous system. "
            b"This science chapter explains cerebrum, cerebellum, and medulla. "
            b"It also covers neurons, spinal cord coordination, memory processing, and sensory control."
        )
        parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="avatarId"\r\n\r\n',
            b"av_coach\r\n",
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="file"; filename="science.txt"\r\n',
            b"Content-Type: text/plain\r\n\r\n",
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        body = b"".join(parts)
        status, upload = self._request_bytes(
            "POST",
            "/avatar-service/training/upload",
            body_bytes=body,
            content_type=f"multipart/form-data; boundary={boundary}",
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 201)
        self.assertEqual(upload["filename"], "science.txt")
        self.assertGreaterEqual(upload["chunksIndexed"], 1)

    def test_student_image_upload_parses_text_for_next_question(self) -> None:
        status, user = self._request(
            "POST",
            "/avatar-service/auth/signup",
            payload={"email": "image-student@example.com"},
        )
        self.assertEqual(status, 201)
        user_id = user["id"]

        status, convo = self._request(
            "POST",
            "/avatar-service/conversations",
            payload={"avatarId": "av_coach"},
            headers={"X-User-Id": user_id},
        )
        self.assertEqual(status, 201)

        with patch.object(
            self.__class__.service,
            "_extract_image_text",
            return_value="Question in image: Name the parts of neuron including axon and dendrites.",
        ):
            status, payload = self._request(
                "POST",
                f"/avatar-service/conversations/{convo['id']}/image",
                payload={
                    "filename": "neuron-question.png",
                    "fileBase64": base64.b64encode(b"fake-image-bytes").decode("ascii"),
                },
                headers={"X-User-Id": user_id},
            )
        self.assertEqual(status, 201)
        self.assertEqual(payload["filename"], "neuron-question.png")
        self.assertGreaterEqual(payload["extractedChars"], 20)
        self.assertIn("neuron", payload["preview"].lower())

    def test_google_auth_and_role_based_flows(self) -> None:
        status, login = self._request(
            "POST",
            "/avatar-service/auth/google",
            payload={"email": "admin@example.com", "name": "Admin User"},
        )
        self.assertEqual(status, 200)
        admin_token = login["token"]
        self.assertEqual(login["user"]["role"], "admin")

        status, teacher = self._request(
            "POST",
            "/avatar-service/admin/users",
            payload={"email": "teacher@example.com", "name": "Teacher One", "role": "teacher"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(teacher["role"], "teacher")

        status, student = self._request(
            "POST",
            "/avatar-service/admin/users",
            payload={"email": "student@example.com", "name": "Student One", "role": "student"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(student["role"], "student")

        status, teacher_login = self._request(
            "POST",
            "/avatar-service/auth/google",
            payload={"email": "teacher@example.com"},
        )
        self.assertEqual(status, 200)
        teacher_token = teacher_login["token"]

        status, student_login = self._request(
            "POST",
            "/avatar-service/auth/google",
            payload={"email": "student@example.com"},
        )
        self.assertEqual(status, 200)
        student_token = student_login["token"]

        book_text = (
            b"Biology chapter: the human brain controls memory, movement, and body coordination in matric science. "
            b"The cerebrum handles thinking and learning, and the cerebellum helps balance and fine motor control."
        )

        status, payload = self._request(
            "POST",
            "/avatar-service/training/upload",
            payload={
                "avatarId": "av_coach",
                "filename": "biology.txt",
                "fileBase64": base64.b64encode(book_text).decode("ascii"),
            },
            headers={"Authorization": f"Bearer {student_token}"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "forbidden_role")

        status, upload = self._request(
            "POST",
            "/avatar-service/training/upload",
            payload={
                "avatarId": "av_coach",
                "filename": "biology.txt",
                "fileBase64": base64.b64encode(book_text).decode("ascii"),
            },
            headers={"Authorization": f"Bearer {teacher_token}"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(upload["filename"], "biology.txt")

        status, payload = self._request(
            "POST",
            "/avatar-service/conversations",
            payload={"avatarId": "av_coach"},
            headers={"Authorization": f"Bearer {teacher_token}"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "forbidden_role")

        status, convo = self._request(
            "POST",
            "/avatar-service/conversations",
            payload={"avatarId": "av_coach"},
            headers={"Authorization": f"Bearer {student_token}"},
        )
        self.assertEqual(status, 201)

        status, turn = self._request(
            "POST",
            f"/avatar-service/conversations/{convo['id']}/messages",
            payload={"text": "Tell me about human brain"},
            headers={"Authorization": f"Bearer {student_token}"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(turn["assistantMessage"]["role"], "assistant")

        status, deleted = self._request(
            "DELETE",
            f"/avatar-service/admin/users/{student['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(deleted["deleted"])

        status, payload = self._request(
            "DELETE",
            f"/avatar-service/admin/users/{login['user']['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "cannot_delete_admin")


if __name__ == "__main__":
    unittest.main()
