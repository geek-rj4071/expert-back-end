"""Dependency-free REST-style WSGI API for avatar chat service."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import re
import tempfile
import time
from http import HTTPStatus
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote_plus
import urllib.error
import urllib.request
from uuid import uuid4
from wsgiref.util import setup_testing_defaults

from .errors import ModerationError, NotFoundError, RateLimitError, ValidationError
from .models import AccountRole, Conversation, Message, User
from .persistent_service import PersistentChatService


API_PREFIX = "/avatar-service"


class AvatarAPI:
    def __init__(self, service: PersistentChatService) -> None:
        self.service = service
        self.web_dir, self.web_assets_dir = self._resolve_web_dirs()
        self.serve_web_ui = os.getenv("SERVE_WEB_UI", "true").strip().lower() in {"1", "true", "yes"}
        self.cors_allow_origin = os.getenv("CORS_ALLOW_ORIGIN", "*").strip() or "*"
        self.cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.cors_allow_headers = os.getenv(
            "CORS_ALLOW_HEADERS",
            "Content-Type, Authorization, X-User-Id",
        ).strip()
        self.cors_allow_methods = os.getenv(
            "CORS_ALLOW_METHODS",
            "GET, POST, DELETE, OPTIONS",
        ).strip()
        self.auth_secret = os.getenv("AUTH_SECRET", "dev-auth-secret-change-me").encode("utf-8")
        self.auth_ttl_seconds = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "43200"))
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
        self.allow_dev_google_login = os.getenv("ALLOW_DEV_GOOGLE_LOGIN", "true").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.allow_legacy_role_bypass = os.getenv("ALLOW_LEGACY_ROLE_BYPASS", "true").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        # Login/session requirement is relaxed for now:
        # if X-User-Id is missing, API falls back to this demo user.
        self.demo_user_id = self.service.register_user(
            f"demo-{uuid4().hex[:8]}@local.dev",
            role=AccountRole.STUDENT.value,
            auth_provider="demo",
            display_name="Demo User",
        ).id

    def app(self, environ, start_response):
        setup_testing_defaults(environ)
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "")

        try:
            if method == "OPTIONS" and path.startswith(API_PREFIX):
                return self._handle_options(start_response)
            if self.serve_web_ui and method == "GET" and path == "/":
                return self._serve_static(start_response, "index.html", "text/html; charset=utf-8")
            if self.serve_web_ui and method == "GET" and path.startswith("/assets/"):
                return self._serve_asset_path(start_response, path)
            if method == "POST" and path == f"{API_PREFIX}/auth/signup":
                return self._handle_signup(environ, start_response)
            if method == "GET" and path == f"{API_PREFIX}/auth/config":
                return self._handle_auth_config(start_response)
            if method == "POST" and path == f"{API_PREFIX}/auth/google":
                return self._handle_google_login(environ, start_response)
            if method == "GET" and path == f"{API_PREFIX}/auth/me":
                return self._handle_auth_me(environ, start_response)
            if method == "POST" and path == f"{API_PREFIX}/auth/logout":
                return self._handle_logout(start_response)
            if method == "GET" and path == f"{API_PREFIX}/admin/users":
                return self._handle_admin_list_users(environ, start_response)
            if method == "POST" and path == f"{API_PREFIX}/admin/users":
                return self._handle_admin_create_user(environ, start_response)
            if method == "GET" and path == f"{API_PREFIX}/avatars":
                return self._handle_list_avatars(start_response)
            if method == "POST" and path == f"{API_PREFIX}/conversations":
                return self._handle_create_conversation(environ, start_response)
            if method == "POST" and path == f"{API_PREFIX}/training/upload":
                return self._handle_training_upload(environ, start_response)
            if method == "GET" and path == f"{API_PREFIX}/training/status":
                return self._handle_training_status(environ, start_response)
            if method == "GET" and path == f"{API_PREFIX}/training/documents":
                return self._handle_training_documents(environ, start_response)
            if method == "DELETE" and path == f"{API_PREFIX}/training/documents":
                return self._handle_clear_training_documents(environ, start_response)
            if method == "POST" and path.startswith(f"{API_PREFIX}/conversations/") and path.endswith("/image"):
                conversation_id = path.split("/")[3]
                return self._handle_student_image_upload(environ, start_response, conversation_id)
            if method == "POST" and path.startswith(f"{API_PREFIX}/conversations/") and path.endswith("/messages"):
                conversation_id = path.split("/")[3]
                return self._handle_send_message(environ, start_response, conversation_id)
            if method == "GET" and path.startswith(f"{API_PREFIX}/conversations/"):
                conversation_id = path.split("/")[3]
                return self._handle_get_conversation(environ, start_response, conversation_id)
            if method == "POST" and path == f"{API_PREFIX}/voice/stt":
                return self._handle_stt(environ, start_response)
            if method == "POST" and path == f"{API_PREFIX}/voice/tts":
                return self._handle_tts(environ, start_response)
            if method == "GET" and path == f"{API_PREFIX}/voice/health":
                return self._handle_voice_health(start_response)
            if method == "GET" and path == f"{API_PREFIX}/ai/health":
                return self._handle_ai_health(start_response)
            if method == "GET" and path == f"{API_PREFIX}/system/status":
                return self._json(start_response, HTTPStatus.OK, self.service.system_status())
            if method == "GET" and path == f"{API_PREFIX}/realtime/sse":
                return self._handle_sse(environ, start_response)
            return self._json(start_response, HTTPStatus.NOT_FOUND, {"error": "route_not_found"})
        except ValidationError as exc:
            return self._json(start_response, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except NotFoundError as exc:
            return self._json(start_response, HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except RateLimitError as exc:
            return self._json(start_response, HTTPStatus.TOO_MANY_REQUESTS, {"error": str(exc)})
        except ModerationError as exc:
            return self._json(start_response, HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
        except FileNotFoundError:
            return self._json(start_response, HTTPStatus.NOT_FOUND, {"error": "asset_not_found"})
        except RuntimeError as exc:
            return self._json(start_response, HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except Exception:
            return self._json(start_response, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_server_error"})

    def _handle_signup(self, environ, start_response):
        body = self._read_json(environ)
        user = self.service.register_user(body.get("email", ""), role=AccountRole.STUDENT.value, auth_provider="legacy")
        return self._json(start_response, HTTPStatus.CREATED, self._serialize_user(user))

    def _handle_auth_config(self, start_response):
        return self._json(start_response, HTTPStatus.OK, {
            "googleClientIdConfigured": bool(self.google_client_id),
            "googleClientId": self.google_client_id,
            "allowDevGoogleLogin": self.allow_dev_google_login,
        })

    def _handle_google_login(self, environ, start_response):
        body = self._read_json(environ)
        credential = str(body.get("credential", "")).strip()
        email = str(body.get("email", "")).strip().lower()
        display_name = str(body.get("name", "")).strip()

        if credential:
            verified = self._verify_google_credential(credential)
            email = str(verified.get("email", "")).strip().lower()
            display_name = str(verified.get("name", "")).strip() or display_name
        elif not self.allow_dev_google_login:
            raise ValidationError("google_credential_required")

        if "@" not in email:
            raise ValidationError("invalid_email")

        existing = self.service.get_user_by_email(email)
        bootstrap_admin = False
        if existing is None:
            if not self.service.has_admin_users():
                existing = self.service.register_user(
                    email,
                    role=AccountRole.ADMIN.value,
                    display_name=display_name or "Admin",
                    auth_provider="google",
                )
                bootstrap_admin = True
            else:
                raise ValidationError("user_not_provisioned")

        token = self._issue_session_token(existing.id)
        return self._json(start_response, HTTPStatus.OK, {
            "token": token,
            "user": self._serialize_user(existing),
            "bootstrapAdmin": bootstrap_admin,
        })

    def _handle_auth_me(self, environ, start_response):
        user = self._require_authenticated_user(environ)
        return self._json(start_response, HTTPStatus.OK, {"user": self._serialize_user(user)})

    def _handle_logout(self, start_response):
        return self._json(start_response, HTTPStatus.OK, {"ok": True})

    def _handle_admin_list_users(self, environ, start_response):
        self._require_role(environ, AccountRole.ADMIN.value)
        users = [self._serialize_user(u) for u in self.service.list_users()]
        return self._json(start_response, HTTPStatus.OK, users)

    def _handle_admin_create_user(self, environ, start_response):
        self._require_role(environ, AccountRole.ADMIN.value)
        body = self._read_json(environ)
        role = str(body.get("role", "")).strip().lower()
        if role not in {AccountRole.TEACHER.value, AccountRole.STUDENT.value}:
            raise ValidationError("invalid_role")
        user = self.service.create_managed_user(
            email=str(body.get("email", "")),
            role=role,
            display_name=str(body.get("name", "")).strip() or None,
        )
        return self._json(start_response, HTTPStatus.CREATED, self._serialize_user(user))

    def _handle_list_avatars(self, start_response):
        avatars = self.service.list_avatars()
        payload = [
            {
                "id": a.id,
                "name": a.name,
                "personaPrompt": a.persona_prompt,
                "voiceId": a.voice_id,
            }
            for a in avatars
        ]
        return self._json(start_response, HTTPStatus.OK, payload)

    def _handle_create_conversation(self, environ, start_response):
        user_id = self._require_role(environ, AccountRole.STUDENT.value).id
        body = self._read_json(environ)
        avatar_id = self._resolve_avatar_id(str(body.get("avatarId", "")))
        convo = self.service.create_conversation(user_id=user_id, avatar_id=avatar_id)
        return self._json(start_response, HTTPStatus.CREATED, self._serialize_conversation(convo))

    def _handle_training_upload(self, environ, start_response):
        user_id = self._require_role(environ, AccountRole.TEACHER.value).id
        upload = self._read_training_upload(environ)
        avatar_id = self._resolve_avatar_id(upload["avatar_id"])
        filename = upload["filename"]
        raw = upload["file_bytes"]

        result = self.service.upload_training_material(
            user_id=user_id,
            avatar_id=avatar_id,
            filename=filename,
            file_bytes=raw,
        )
        return self._json(start_response, HTTPStatus.CREATED, {
            "documentId": result.document_id,
            "filename": result.filename,
            "chunksIndexed": result.chunks_indexed,
            "extractedChars": result.extracted_chars,
            "embeddingsIndexed": result.embeddings_indexed,
        })

    def _handle_training_status(self, environ, start_response):
        avatar_id = self._resolve_avatar_id(self._optional_query_param(environ, "avatarId"))
        payload = self.service.training_status(avatar_id=avatar_id)
        return self._json(start_response, HTTPStatus.OK, payload)

    def _handle_training_documents(self, environ, start_response):
        avatar_id = self._resolve_avatar_id(self._optional_query_param(environ, "avatarId"))
        docs = self.service.list_training_documents(avatar_id=avatar_id)
        payload = [
            {
                "id": d.id,
                "avatarId": d.avatar_id,
                "filename": d.filename,
                "sourceType": d.source_type,
                "createdAt": d.created_at.isoformat(),
                "characters": len(d.content_text),
            }
            for d in docs
        ]
        return self._json(start_response, HTTPStatus.OK, payload)

    def _handle_clear_training_documents(self, environ, start_response):
        user_id = self._require_role(environ, AccountRole.TEACHER.value).id
        avatar_id = self._resolve_avatar_id(self._optional_query_param(environ, "avatarId"))
        deleted = self.service.clear_training_documents(user_id=user_id, avatar_id=avatar_id)
        return self._json(start_response, HTTPStatus.OK, {"deleted": deleted})

    def _handle_student_image_upload(self, environ, start_response, conversation_id: str):
        user_id = self._require_role(environ, AccountRole.STUDENT.value).id
        upload = self._read_student_image_upload(environ)
        result = self.service.upload_student_image_context(
            user_id=user_id,
            conversation_id=conversation_id,
            filename=upload["filename"],
            file_bytes=upload["file_bytes"],
        )
        return self._json(start_response, HTTPStatus.CREATED, {
            "imageId": result.image_id,
            "conversationId": result.conversation_id,
            "filename": result.filename,
            "extractedChars": result.extracted_chars,
            "preview": result.preview,
        })

    def _handle_send_message(self, environ, start_response, conversation_id: str):
        user_id = self._require_role(environ, AccountRole.STUDENT.value).id
        body = self._read_json(environ)
        text = body.get("text", "")

        try:
            turn = self.service.send_message(
                user_id=user_id,
                conversation_id=conversation_id,
                text=text,
            )
        except NotFoundError as exc:
            if str(exc) != "conversation_not_found":
                raise

            convo = self.service.repository.get_conversation(conversation_id)
            if convo is None:
                # Conversation id is stale; create a replacement and continue.
                avatars = self.service.list_avatars()
                if not avatars:
                    raise NotFoundError("avatar_not_found")
                fresh = self.service.create_conversation(user_id=user_id, avatar_id=avatars[0].id)
                turn = self.service.send_message(
                    user_id=user_id,
                    conversation_id=fresh.id,
                    text=text,
                )
            else:
                # Conversation belongs to a different user.
                raise

        return self._json(start_response, HTTPStatus.CREATED, {
            "conversationId": turn.conversation_id,
            "userMessage": self._serialize_message(turn.user_message),
            "assistantMessage": self._serialize_message(turn.assistant_message),
        })

    def _handle_get_conversation(self, environ, start_response, conversation_id: str):
        user_id = self._require_role(environ, AccountRole.STUDENT.value).id
        convo = self.service.get_conversation(user_id=user_id, conversation_id=conversation_id)
        return self._json(start_response, HTTPStatus.OK, self._serialize_conversation(convo))

    def _handle_stt(self, environ, start_response):
        self._require_role(environ, AccountRole.STUDENT.value)
        body = self._read_json(environ)
        encoded = body.get("audioBase64", "")
        mime_type = body.get("mimeType", "audio/wav")
        audio = base64.b64decode(encoded) if encoded else b""
        text = self.service.transcribe_audio(audio_bytes=audio, mime_type=mime_type)
        return self._json(start_response, HTTPStatus.OK, {"text": text})

    def _handle_tts(self, environ, start_response):
        self._require_role(environ, AccountRole.STUDENT.value)
        body = self._read_json(environ)
        result = self.service.synthesize_audio(
            text=body.get("text", ""),
            voice_id=body.get("voiceId", "alloy"),
        )
        return self._json(start_response, HTTPStatus.OK, {
            "mimeType": result.mime_type,
            "audioBase64": base64.b64encode(result.audio_bytes).decode("ascii"),
        })

    def _handle_voice_health(self, start_response):
        try:
            result = self.service.synthesize_audio(text="Voice health check", voice_id="alloy")
            payload = {
                "ok": True,
                "mimeType": result.mime_type,
                "bytes": len(result.audio_bytes),
            }
        except Exception as exc:
            payload = {
                "ok": False,
                "error": str(exc),
                "bytes": 0,
            }
        return self._json(start_response, HTTPStatus.OK, payload)

    def _handle_ai_health(self, start_response):
        health = self.service.ai_health()
        return self._json(start_response, HTTPStatus.OK, health)

    def _handle_sse(self, environ, start_response):
        params = parse_qs(environ.get("QUERY_STRING", ""))
        user_id = (params.get("userId") or [""])[0].strip()
        conversation_id = (params.get("conversationId") or [""])[0].strip()
        text = (params.get("text") or [""])[0]
        if not conversation_id:
            raise ValidationError("missing_conversation_id")
        student_user = self._require_authenticated_user(environ)
        if (
            not str(environ.get("HTTP_AUTHORIZATION", "")).strip()
            and not str(environ.get("HTTP_X_USER_ID", "")).strip()
            and user_id
        ):
            queried_user = self.service.get_user(user_id)
            if queried_user is not None:
                student_user = queried_user
        if student_user.role.lower() != AccountRole.STUDENT.value and not (
            self.allow_legacy_role_bypass and self._is_legacy_request(environ)
        ):
            raise ValidationError("forbidden_role")
        user_id = student_user.id

        convo = self.service.repository.get_conversation(conversation_id)
        if convo is None:
            avatars = self.service.list_avatars()
            if not avatars:
                raise NotFoundError("avatar_not_found")
            # Auto-heal stale conversation IDs by creating a new conversation.
            convo = self.service.create_conversation(user_id=user_id, avatar_id=avatars[0].id)
            conversation_id = convo.id
            user_id = convo.user_id
        else:
            if convo.user_id != user_id:
                raise NotFoundError("conversation_not_found")

        events = self.service.stream_message(user_id=user_id, conversation_id=conversation_id, text=text)

        start_response(
            "200 OK",
            [
                ("Content-Type", "text/event-stream"),
                ("Cache-Control", "no-cache"),
                ("Connection", "keep-alive"),
            ] + self._cors_headers(),
        )

        def _iter_events():
            yield b": stream-start\n\n"
            try:
                for event in events:
                    name = event.pop("event")
                    payload = json.dumps(event)
                    yield f"event: {name}\n".encode("utf-8")
                    yield f"data: {payload}\n\n".encode("utf-8")
            except Exception as exc:
                payload = json.dumps({"error": str(exc)})
                yield b"event: stream.error\n"
                yield f"data: {payload}\n\n".encode("utf-8")
            yield b"event: done\ndata: {}\n\n"

        return _iter_events()

    def _handle_options(self, start_response):
        start_response(
            "204 No Content",
            [("Content-Length", "0")] + self._cors_headers(),
        )
        return [b""]

    def _read_json(self, environ) -> dict:
        raw = self._read_body_bytes(environ, default_empty=b"{}")
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _read_training_upload(self, environ) -> dict:
        content_type = str(environ.get("CONTENT_TYPE", "")).strip()
        lowered = content_type.lower()
        if lowered.startswith("multipart/form-data"):
            return self._read_training_upload_multipart(environ, content_type)

        body = self._read_json(environ)
        avatar_id = str(body.get("avatarId", "")).strip()
        filename = str(body.get("filename", "")).strip()
        encoded = str(body.get("fileBase64", "")).strip()
        if encoded.startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        encoded = re.sub(r"\s+", "", encoded)
        try:
            raw = base64.b64decode(encoded) if encoded else b""
        except Exception as exc:
            raise ValidationError("invalid_file_base64") from exc
        return {
            "avatar_id": avatar_id,
            "filename": filename,
            "file_bytes": raw,
        }

    def _read_student_image_upload(self, environ) -> dict:
        parsed = self._read_training_upload(environ)
        return {
            "filename": str(parsed.get("filename", "")).strip(),
            "file_bytes": parsed.get("file_bytes", b""),
        }

    def _read_training_upload_multipart(self, environ, content_type: str) -> dict:
        boundary_match = re.search(r'boundary="?([^";]+)"?', content_type, flags=re.I)
        if boundary_match is None:
            raise ValidationError("invalid_multipart_boundary")
        boundary = boundary_match.group(1).encode("utf-8", errors="ignore")
        if not boundary:
            raise ValidationError("invalid_multipart_boundary")

        max_file_bytes = int(getattr(self.service.config, "max_training_file_bytes", 8_000_000))
        # Multipart overhead (headers/boundaries) can exceed file bytes slightly.
        max_body_bytes = max_file_bytes + 1_500_000
        content_length = self._parse_content_length(environ)
        if content_length <= 0:
            raise ValidationError("file_empty")
        if content_length > max_body_bytes:
            raise ValidationError("payload_too_large")

        stream = environ.get("wsgi.input")
        if stream is None:
            raise ValidationError("invalid_multipart_body")

        boundary_line = b"--" + boundary
        terminal_boundary = boundary_line + b"--"

        remaining = content_length
        first_line, remaining = self._stream_readline(stream, remaining)
        while first_line in (b"\r\n", b"\n"):
            first_line, remaining = self._stream_readline(stream, remaining)
        if first_line.rstrip(b"\r\n") != boundary_line:
            raise ValidationError("invalid_multipart_body")

        fields: dict[str, str] = {}
        file_bytes = b""
        filename = ""
        done = False

        while not done:
            headers: list[str] = []
            while True:
                line, remaining = self._stream_readline(stream, remaining)
                if not line:
                    raise ValidationError("invalid_multipart_body")
                if line in (b"\r\n", b"\n"):
                    break
                headers.append(line.decode("latin1", errors="ignore").strip())

            disposition = ""
            for header in headers:
                if ":" not in header:
                    continue
                key, value = header.split(":", 1)
                if key.strip().lower() == "content-disposition":
                    disposition = value.strip()
                    break

            name = ""
            part_filename = ""
            if disposition:
                name_match = re.search(r'name="([^"]+)"', disposition)
                if name_match is not None:
                    name = name_match.group(1).strip()
                filename_match = re.search(r'filename="([^"]*)"', disposition)
                if filename_match is not None:
                    part_filename = filename_match.group(1).strip()

            is_file_part = bool(part_filename)
            if is_file_part:
                sink = tempfile.SpooledTemporaryFile(max_size=min(max_file_bytes, 2_000_000), mode="w+b")
                sink_size = 0
            else:
                sink = bytearray()

            reached_boundary = False
            is_terminal = False
            while True:
                line, remaining = self._stream_readline(stream, remaining)
                if not line:
                    break
                stripped = line.rstrip(b"\r\n")
                if stripped == boundary_line or stripped == terminal_boundary:
                    reached_boundary = True
                    is_terminal = stripped == terminal_boundary
                    if is_file_part:
                        sink_size = self._trim_file_sink_crlf(sink, sink_size)
                    else:
                        if sink.endswith(b"\r\n"):
                            del sink[-2:]
                        elif sink.endswith(b"\n"):
                            del sink[-1:]
                    break

                if is_file_part:
                    sink_size += len(line)
                    if sink_size > max_file_bytes:
                        try:
                            sink.close()
                        except Exception:
                            pass
                        raise ValidationError("file_too_large")
                    sink.write(line)
                else:
                    sink.extend(line)
                    if len(sink) > 200_000:
                        raise ValidationError("payload_too_large")

            if not reached_boundary:
                raise ValidationError("invalid_multipart_body")

            if is_file_part:
                sink.seek(0)
                file_bytes = sink.read()
                sink.close()
                filename = part_filename
            else:
                text_value = bytes(sink).decode("utf-8", errors="ignore").strip()
                if name:
                    fields[name] = text_value

            if is_terminal:
                done = True

        if not file_bytes and fields.get("fileBase64"):
            encoded = fields.get("fileBase64", "")
            if encoded.startswith("data:") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            encoded = re.sub(r"\s+", "", encoded)
            try:
                file_bytes = base64.b64decode(encoded) if encoded else b""
            except Exception as exc:
                raise ValidationError("invalid_file_base64") from exc

        if len(file_bytes) > max_file_bytes:
            raise ValidationError("file_too_large")

        final_name = fields.get("filename", "").strip() or filename
        return {
            "avatar_id": fields.get("avatarId", "").strip(),
            "filename": final_name,
            "file_bytes": file_bytes,
        }

    def _parse_content_length(self, environ) -> int:
        size_raw = str(environ.get("CONTENT_LENGTH", "0") or "0").strip()
        try:
            return int(size_raw)
        except ValueError:
            raise ValidationError("invalid_content_length")

    def _stream_readline(self, stream, remaining: int) -> tuple[bytes, int]:
        if remaining <= 0:
            return b"", 0
        line = stream.readline(min(65536, remaining))
        if not line:
            return b"", 0
        return line, max(0, remaining - len(line))

    def _trim_file_sink_crlf(self, sink, size: int) -> int:
        if size <= 0:
            return 0
        trim = 0
        if size >= 2:
            sink.seek(size - 2)
            tail2 = sink.read(2)
            if tail2 == b"\r\n":
                trim = 2
        if trim == 0 and size >= 1:
            sink.seek(size - 1)
            tail1 = sink.read(1)
            if tail1 == b"\n":
                trim = 1
        if trim > 0:
            sink.seek(size - trim)
            sink.truncate()
            size -= trim
        return size

    def _read_body_bytes(self, environ, *, limit: int | None = None, default_empty: bytes = b"") -> bytes:
        size = self._parse_content_length(environ)
        if size <= 0:
            return default_empty
        if limit is not None and size > limit:
            raise ValidationError("payload_too_large")
        stream = environ.get("wsgi.input")
        if stream is None:
            return default_empty
        raw = stream.read(size)
        if limit is not None and len(raw) > limit:
            raise ValidationError("payload_too_large")
        return raw

    def _require_user_id(self, environ) -> str:
        user = self._require_authenticated_user(environ)
        return user.id

    def _is_legacy_request(self, environ) -> bool:
        auth_header = str(environ.get("HTTP_AUTHORIZATION", "")).strip()
        return bool(environ.get("HTTP_X_USER_ID", "").strip()) and not auth_header

    def _require_authenticated_user(self, environ) -> User:
        auth_header = str(environ.get("HTTP_AUTHORIZATION", "")).strip()
        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        if token:
            user_id = self._parse_session_token(token)
            if not user_id:
                raise ValidationError("invalid_auth_token")
            user = self.service.get_user(user_id)
            if user is None:
                raise ValidationError("invalid_auth_token")
            return user

        legacy_user_id = str(environ.get("HTTP_X_USER_ID", "")).strip()
        if legacy_user_id:
            user = self.service.get_user(legacy_user_id)
            if user is None:
                raise NotFoundError("user_not_found")
            return user

        return self.service.get_user(self.demo_user_id) or self.service.register_user(
            f"demo-{uuid4().hex[:8]}@local.dev",
            role=AccountRole.STUDENT.value,
            auth_provider="demo",
            display_name="Demo User",
        )

    def _require_role(self, environ, *roles: str) -> User:
        user = self._require_authenticated_user(environ)
        allowed = {str(r).strip().lower() for r in roles if str(r).strip()}
        if not allowed:
            return user
        if user.role.lower() in allowed:
            return user
        if self.allow_legacy_role_bypass and self._is_legacy_request(environ):
            return user
        raise ValidationError("forbidden_role")

    def _require_query_param(self, environ, key: str) -> str:
        params = parse_qs(environ.get("QUERY_STRING", ""))
        value = (params.get(key) or [""])[0].strip()
        if not value:
            raise ValidationError(f"missing_{key.lower()}")
        return value

    def _optional_query_param(self, environ, key: str) -> str:
        params = parse_qs(environ.get("QUERY_STRING", ""))
        return (params.get(key) or [""])[0].strip()

    def _resolve_avatar_id(self, avatar_id: str) -> str:
        clean = avatar_id.strip()
        if clean:
            return clean
        avatars = self.service.list_avatars()
        if not avatars:
            raise NotFoundError("avatar_not_found")
        return avatars[0].id

    def _serialize_user(self, user: User) -> dict:
        return {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "name": user.display_name or "",
            "authProvider": user.auth_provider,
            "createdAt": user.created_at.isoformat(),
        }

    def _issue_session_token(self, user_id: str) -> str:
        now = int(time.time())
        payload = {"uid": user_id, "iat": now, "exp": now + self.auth_ttl_seconds}
        payload_raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(payload_raw).decode("ascii").rstrip("=")
        signature = hmac.new(self.auth_secret, payload_b64.encode("utf-8"), hashlib.sha256).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
        return f"{payload_b64}.{signature_b64}"

    def _parse_session_token(self, token: str) -> str | None:
        if "." not in token:
            return None
        payload_b64, signature_b64 = token.split(".", 1)
        if not payload_b64 or not signature_b64:
            return None
        expected_sig = hmac.new(self.auth_secret, payload_b64.encode("utf-8"), hashlib.sha256).digest()
        try:
            actual_sig = base64.urlsafe_b64decode(self._add_padding(signature_b64))
        except Exception:
            return None
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        try:
            payload_raw = base64.urlsafe_b64decode(self._add_padding(payload_b64))
            payload = json.loads(payload_raw.decode("utf-8"))
        except Exception:
            return None
        exp = int(payload.get("exp", 0))
        uid = str(payload.get("uid", "")).strip()
        if not uid or exp <= int(time.time()):
            return None
        return uid

    def _add_padding(self, value: str) -> str:
        rem = len(value) % 4
        if rem == 0:
            return value
        return value + ("=" * (4 - rem))

    def _verify_google_credential(self, credential: str) -> dict:
        token = credential.strip()
        if not token:
            raise ValidationError("google_credential_required")
        url = f"https://oauth2.googleapis.com/tokeninfo?id_token={quote_plus(token)}"
        req = urllib.request.Request(url, headers={"User-Agent": "avatar-api/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ValidationError(f"google_verify_failed: {exc}") from exc
        except Exception as exc:
            raise ValidationError("google_verify_failed") from exc

        aud = str(payload.get("aud", "")).strip()
        if self.google_client_id and aud and aud != self.google_client_id:
            raise ValidationError("google_client_mismatch")
        email = str(payload.get("email", "")).strip().lower()
        if "@" not in email:
            raise ValidationError("invalid_google_email")
        return {
            "email": email,
            "name": str(payload.get("name", "")).strip(),
        }

    def _serialize_message(self, message: Message) -> dict:
        return {
            "id": message.id,
            "role": message.role.value,
            "text": message.text,
            "emotion": message.emotion,
            "createdAt": message.created_at.isoformat(),
        }

    def _serialize_conversation(self, convo: Conversation) -> dict:
        return {
            "id": convo.id,
            "userId": convo.user_id,
            "avatar": {
                "id": convo.avatar.id,
                "name": convo.avatar.name,
                "personaPrompt": convo.avatar.persona_prompt,
                "voiceId": convo.avatar.voice_id,
            },
            "createdAt": convo.created_at.isoformat(),
            "messages": [self._serialize_message(m) for m in convo.messages],
        }

    def _json(
        self,
        start_response: Callable,
        status: HTTPStatus,
        payload: dict | list,
        *,
        include_cors: bool = True,
    ):
        raw = json.dumps(payload).encode("utf-8")
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(raw))),
        ]
        if include_cors:
            headers.extend(self._cors_headers())
        start_response(
            f"{status.value} {status.phrase}",
            headers,
        )
        return [raw]

    def _serve_static(self, start_response: Callable, name: str, content_type: str):
        raw = (self.web_dir / name).read_bytes()
        start_response(
            "200 OK",
            [
                ("Content-Type", content_type),
                ("Content-Length", str(len(raw))),
            ],
        )
        return [raw]

    def _serve_asset_path(self, start_response: Callable, path: str):
        relative = path.removeprefix("/assets/")
        if ".." in relative or relative.startswith("/"):
            return self._json(start_response, HTTPStatus.BAD_REQUEST, {"error": "invalid_asset_path"})

        file_path = self.web_assets_dir / relative
        if not file_path.exists() or not file_path.is_file():
            # Legacy fallback where assets can live directly under the web root.
            legacy_path = self.web_dir / relative
            if not legacy_path.exists() or not legacy_path.is_file():
                return self._json(start_response, HTTPStatus.NOT_FOUND, {"error": "asset_not_found"})
            file_path = legacy_path

        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type is None:
            content_type = "application/octet-stream"
        if content_type == "text/javascript":
            content_type = "application/javascript"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type = f"{content_type}; charset=utf-8"

        raw = file_path.read_bytes()
        start_response(
            "200 OK",
            [
                ("Content-Type", content_type),
                ("Content-Length", str(len(raw))),
            ],
        )
        return [raw]

    def _resolve_web_dirs(self) -> tuple[Path, Path]:
        legacy_web_dir = Path(__file__).with_name("web")
        repo_root = Path(__file__).resolve().parents[2]
        react_dist_dir = repo_root / "apps" / "web" / "dist"
        react_assets_dir = react_dist_dir / "assets"
        if (react_dist_dir / "index.html").exists() and react_assets_dir.exists():
            return react_dist_dir, react_assets_dir
        return legacy_web_dir, legacy_web_dir

    def _cors_headers(self) -> list[tuple[str, str]]:
        headers = [
            ("Access-Control-Allow-Origin", self.cors_allow_origin),
            ("Access-Control-Allow-Methods", self.cors_allow_methods),
            ("Access-Control-Allow-Headers", self.cors_allow_headers),
            ("Access-Control-Max-Age", "86400"),
            ("Vary", "Origin"),
        ]
        if self.cors_allow_credentials and self.cors_allow_origin != "*":
            headers.append(("Access-Control-Allow-Credentials", "true"))
        return headers
