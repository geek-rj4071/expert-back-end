"""Persistent chat service using SQLite repository and pluggable providers."""
from __future__ import annotations

import json
import os
import re
import zlib
from binascii import unhexlify
import base64
import hashlib
import html
import io
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import tempfile
import zipfile
from typing import Iterator
from urllib.parse import quote, quote_plus
import urllib.error
import urllib.request
from xml.etree import ElementTree as ET

from .errors import ModerationError, NotFoundError, RateLimitError, ValidationError
from .models import AccountRole, Avatar, Conversation, Message, Role, TurnResponse, User, new_id
from .moderation import ModerationPolicy
from .persistence import SQLiteRepository, TrainingChunkVector, TrainingDocument
from .providers import (
    DeterministicLLMProvider,
    LLMResult,
    LLMProvider,
    MockSTTProvider,
    MockTTSProvider,
    STTProvider,
    TTSProvider,
)
from .rate_limit import FixedWindowRateLimiter, RateLimitConfig


@dataclass(frozen=True)
class PersistentServiceConfig:
    max_message_chars: int = 4000
    max_tts_chars: int = 12000
    memory_messages_limit: int = 40
    rate_limit_max_requests: int = 20
    rate_limit_window_seconds: int = 60
    max_training_file_bytes: int = 8_000_000
    max_training_docs_per_avatar: int = 40
    internet_lookup_enabled: bool = False
    internet_lookup_timeout_seconds: float = 6.0
    internet_lookup_max_snippets: int = 4
    math_specialization_threshold: float = 0.35
    chunk_chars: int = 900
    chunk_overlap_chars: int = 130
    embedding_dimension: int = 256
    embedding_provider: str = "local"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://127.0.0.1:11434"
    embedding_timeout_seconds: float = 20.0
    strict_book_only_mode: bool = True
    ocr_enabled: bool = True
    ocr_max_pages: int = 15
    ocr_language: str = "eng"
    max_student_image_file_bytes: int = 6_000_000
    hinglish_enabled: bool = True


@dataclass(frozen=True)
class TrainingUploadResult:
    document_id: str
    filename: str
    chunks_indexed: int
    extracted_chars: int
    embeddings_indexed: int


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    token_overlap: int
    vector_similarity: float


@dataclass(frozen=True)
class StudentImageUploadResult:
    image_id: str
    conversation_id: str
    filename: str
    extracted_chars: int
    preview: str


class PersistentChatService:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        config: PersistentServiceConfig | None = None,
        moderation: ModerationPolicy | None = None,
        llm_provider: LLMProvider | None = None,
        stt_provider: STTProvider | None = None,
        tts_provider: TTSProvider | None = None,
        llm_fallback_enabled: bool = True,
        ai_provider_name: str | None = None,
        tts_provider_name: str | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or PersistentServiceConfig()
        self.moderation = moderation or ModerationPolicy()
        self.llm_provider = llm_provider or DeterministicLLMProvider()
        self.llm_fallback_enabled = llm_fallback_enabled
        self.stt_provider = stt_provider or MockSTTProvider()
        self.tts_provider = tts_provider or MockTTSProvider()
        self.ai_provider_name = ai_provider_name or type(self.llm_provider).__name__
        self.tts_provider_name = tts_provider_name or type(self.tts_provider).__name__
        self._ollama_embedding_available = True
        self.rate_limiter = FixedWindowRateLimiter(
            RateLimitConfig(
                max_requests=self.config.rate_limit_max_requests,
                window_seconds=self.config.rate_limit_window_seconds,
            )
        )
        self._teacher_scope_keywords = {
            "math",
            "mathematics",
            "physics",
            "chemistry",
            "biology",
            "science",
            "english",
            "urdu",
            "history",
            "geography",
            "algebra",
            "equation",
            "chapter",
            "exercise",
            "lesson",
            "class",
            "matric",
            "grade",
            "school",
            "question",
            "solve",
            "theorem",
            "grammar",
            "essay",
            "poem",
            "acid",
            "acids",
            "base",
            "bases",
            "litmus",
            "photosynthesis",
            "respiration",
            "force",
            "motion",
            "newton",
            "velocity",
            "acceleration",
            "cell",
            "organ",
            "digestion",
            "ecosystem",
        }
        self._token_stop_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "what",
            "when",
            "where",
            "which",
            "how",
            "who",
            "why",
            "are",
            "can",
            "could",
            "would",
            "should",
            "will",
            "into",
            "about",
            "your",
            "have",
            "has",
            "had",
            "was",
            "were",
            "you",
            "our",
            "their",
        }

    def seed_avatars(self, avatars: list[Avatar]) -> None:
        for avatar in avatars:
            self.repository.upsert_avatar(avatar)

    def list_avatars(self) -> list[Avatar]:
        return self.repository.list_avatars()

    def register_user(
        self,
        email: str,
        *,
        role: str = AccountRole.STUDENT.value,
        display_name: str | None = None,
        auth_provider: str = "google",
    ) -> User:
        clean = email.strip().lower()
        if "@" not in clean:
            raise ValidationError("invalid_email")
        allowed = {
            AccountRole.ADMIN.value,
            AccountRole.TEACHER.value,
            AccountRole.STUDENT.value,
        }
        role_clean = str(role or "").strip().lower()
        if role_clean not in allowed:
            raise ValidationError("invalid_role")
        return self.repository.create_user(
            user_id=new_id("usr"),
            email=clean,
            role=role_clean,
            display_name=(display_name or "").strip() or None,
            auth_provider=(auth_provider or "google").strip().lower() or "google",
        )

    def get_user(self, user_id: str) -> User | None:
        return self.repository.get_user(user_id)

    def get_user_by_email(self, email: str) -> User | None:
        clean = email.strip().lower()
        if not clean:
            return None
        return self.repository.get_user_by_email(clean)

    def list_users(self) -> list[User]:
        return self.repository.list_users()

    def has_admin_users(self) -> bool:
        return self.repository.count_users_by_role(AccountRole.ADMIN.value) > 0

    def create_managed_user(self, *, email: str, role: str, display_name: str | None = None) -> User:
        clean = email.strip().lower()
        if "@" not in clean:
            raise ValidationError("invalid_email")
        role_clean = str(role or "").strip().lower()
        if role_clean not in {AccountRole.TEACHER.value, AccountRole.STUDENT.value}:
            raise ValidationError("invalid_role")
        existing = self.repository.get_user_by_email(clean)
        if existing is not None:
            raise ValidationError("user_already_exists")
        return self.repository.create_user(
            user_id=new_id("usr"),
            email=clean,
            role=role_clean,
            display_name=(display_name or "").strip() or None,
            auth_provider="google",
        )

    def delete_managed_user(self, *, user_id: str) -> bool:
        target = self.repository.get_user(user_id)
        if target is None:
            raise NotFoundError("user_not_found")
        role = str(target.role or "").strip().lower()
        if role not in {AccountRole.TEACHER.value, AccountRole.STUDENT.value}:
            raise ValidationError("cannot_delete_admin")
        return self.repository.delete_user(user_id=user_id)

    def create_conversation(self, *, user_id: str, avatar_id: str) -> Conversation:
        user = self.repository.get_user(user_id)
        if user is None:
            raise NotFoundError("user_not_found")
        avatar = self.repository.get_avatar(avatar_id)
        if avatar is None:
            raise NotFoundError("avatar_not_found")
        return self.repository.create_conversation(
            conversation_id=new_id("cnv"),
            user_id=user.id,
            avatar_id=avatar.id,
        )

    def get_conversation(self, *, user_id: str, conversation_id: str) -> Conversation:
        self._ensure_user(user_id)
        convo = self.repository.get_conversation(conversation_id)
        if convo is None or convo.user_id != user_id:
            raise NotFoundError("conversation_not_found")
        return convo

    def send_message(self, *, user_id: str, conversation_id: str, text: str) -> TurnResponse:
        return self._process_turn(user_id=user_id, conversation_id=conversation_id, text=text)

    def stream_message(self, *, user_id: str, conversation_id: str, text: str) -> Iterator[dict]:
        turn = self._process_turn(user_id=user_id, conversation_id=conversation_id, text=text, persist_assistant=False)
        yield {
            "event": "user.accepted",
            "conversationId": conversation_id,
            "message": {
                "id": turn.user_message.id,
                "role": turn.user_message.role.value,
                "text": turn.user_message.text,
                "createdAt": turn.user_message.created_at.isoformat(),
            },
        }

        assistant_text = turn.assistant_message.text
        chunk_size = 18
        for index in range(0, len(assistant_text), chunk_size):
            chunk = assistant_text[index:index + chunk_size]
            yield {
                "event": "assistant.delta",
                "conversationId": conversation_id,
                "delta": chunk,
            }

        assistant_msg = self.repository.add_message(
            conversation_id=conversation_id,
            message_id=turn.assistant_message.id,
            role=Role.ASSISTANT,
            text=turn.assistant_message.text,
            emotion=turn.assistant_message.emotion,
        )
        self.repository.trim_messages(
            conversation_id=conversation_id,
            keep_last=self.config.memory_messages_limit,
        )

        yield {
            "event": "assistant.final",
            "conversationId": conversation_id,
            "message": {
                "id": assistant_msg.id,
                "role": assistant_msg.role.value,
                "text": assistant_msg.text,
                "emotion": assistant_msg.emotion,
                "createdAt": assistant_msg.created_at.isoformat(),
            },
        }

    def _process_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        text: str,
        persist_assistant: bool = True,
    ) -> TurnResponse:
        self._ensure_user(user_id)
        convo = self.get_conversation(user_id=user_id, conversation_id=conversation_id)
        clean = self._validate_text(text)

        if not self.rate_limiter.allow(user_id):
            raise RateLimitError("too_many_requests")

        incoming_check = self.moderation.check(clean)
        if not incoming_check.ok:
            raise ModerationError(incoming_check.reason or "moderation_blocked")

        user_msg = self.repository.add_message(
            conversation_id=convo.id,
            message_id=new_id("msg"),
            role=Role.USER,
            text=clean,
            emotion=None,
        )

        ai = self._build_teacher_answer(avatar_id=convo.avatar.id, user_text=clean, conversation=convo)
        outgoing_check = self.moderation.check(ai.text)
        if not outgoing_check.ok:
            raise ModerationError(outgoing_check.reason or "unsafe_assistant_output")

        if persist_assistant:
            assistant_msg = self.repository.add_message(
                conversation_id=convo.id,
                message_id=new_id("msg"),
                role=Role.ASSISTANT,
                text=ai.text,
                emotion=ai.emotion,
            )
            self.repository.trim_messages(
                conversation_id=convo.id,
                keep_last=self.config.memory_messages_limit,
            )
        else:
            assistant_msg = Message(
                id=new_id("msg"),
                role=Role.ASSISTANT,
                text=ai.text,
                emotion=ai.emotion,
                created_at=user_msg.created_at,
            )

        return TurnResponse(
            conversation_id=convo.id,
            user_message=user_msg,
            assistant_message=assistant_msg,
        )

    def upload_training_material(
        self,
        *,
        user_id: str,
        avatar_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> TrainingUploadResult:
        self._ensure_user(user_id)
        avatar = self.repository.get_avatar(avatar_id)
        if avatar is None:
            raise NotFoundError("avatar_not_found")

        if not filename.strip():
            raise ValidationError("filename_required")
        if not file_bytes:
            raise ValidationError("file_empty")
        if len(file_bytes) > self.config.max_training_file_bytes:
            raise ValidationError("file_too_large")

        existing_docs = self.repository.list_training_documents(avatar_id=avatar_id)
        if len(existing_docs) >= self.config.max_training_docs_per_avatar:
            raise ValidationError("training_docs_limit_reached")

        source_type = self._detect_source_type(filename)
        extracted = self._extract_training_text(filename=filename, file_bytes=file_bytes)
        cleaned = self._normalize_whitespace(extracted)
        min_chars = 12 if source_type == "pdf" else 120
        if len(cleaned) < min_chars:
            raise ValidationError("insufficient_readable_text")

        doc_id = new_id("tdoc")
        self.repository.add_training_document(
            doc_id=doc_id,
            avatar_id=avatar_id,
            filename=filename.strip(),
            content_text=cleaned,
            source_type=source_type,
        )

        chunks = self._chunk_text(
            cleaned,
            chunk_chars=self.config.chunk_chars,
            overlap_chars=self.config.chunk_overlap_chars,
        )
        chunk_rows = [(new_id("tchk"), chunk, idx) for idx, chunk in enumerate(chunks)]
        self.repository.replace_training_chunks(document_id=doc_id, avatar_id=avatar_id, chunks=chunk_rows)
        embeddings = self._generate_embeddings([chunk for _, chunk, _ in chunk_rows])
        vector_rows: list[tuple[str, str, int, str, list[float], int]] = []
        for (chunk_id, chunk_text, chunk_idx), vector in zip(chunk_rows, embeddings):
            vector_rows.append(
                (
                    new_id("tvec"),
                    chunk_id,
                    chunk_idx,
                    chunk_text,
                    vector,
                    len(vector),
                )
            )
        self.repository.replace_training_chunk_vectors(
            document_id=doc_id,
            avatar_id=avatar_id,
            vectors=vector_rows,
        )
        return TrainingUploadResult(
            document_id=doc_id,
            filename=filename.strip(),
            chunks_indexed=len(chunks),
            extracted_chars=len(cleaned),
            embeddings_indexed=len(vector_rows),
        )

    def list_training_documents(self, *, avatar_id: str) -> list[TrainingDocument]:
        avatar = self.repository.get_avatar(avatar_id)
        if avatar is None:
            raise NotFoundError("avatar_not_found")
        return self.repository.list_training_documents(avatar_id=avatar_id)

    def clear_training_documents(self, *, user_id: str, avatar_id: str) -> int:
        self._ensure_user(user_id)
        avatar = self.repository.get_avatar(avatar_id)
        if avatar is None:
            raise NotFoundError("avatar_not_found")
        return self.repository.clear_training_documents(avatar_id=avatar_id)

    def upload_student_image_context(
        self,
        *,
        user_id: str,
        conversation_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> StudentImageUploadResult:
        self._ensure_user(user_id)
        convo = self.get_conversation(user_id=user_id, conversation_id=conversation_id)

        clean_name = (filename or "").strip()
        if not clean_name:
            raise ValidationError("filename_required")
        if not file_bytes:
            raise ValidationError("file_empty")
        if len(file_bytes) > self.config.max_student_image_file_bytes:
            raise ValidationError("file_too_large")
        if not self._looks_like_image_file(filename=clean_name, file_bytes=file_bytes):
            raise ValidationError("invalid_image_file")

        extracted = self._normalize_whitespace(
            self._extract_image_text(filename=clean_name, file_bytes=file_bytes)
        )
        if len(extracted) < 8:
            if not self._image_ocr_support_available():
                raise ValidationError("image_ocr_unavailable")
            raise ValidationError("insufficient_readable_text")

        image_id = new_id("imgctx")
        self.repository.add_conversation_image_context(
            context_id=image_id,
            conversation_id=convo.id,
            user_id=user_id,
            filename=clean_name,
            content_text=extracted,
        )
        return StudentImageUploadResult(
            image_id=image_id,
            conversation_id=convo.id,
            filename=clean_name,
            extracted_chars=len(extracted),
            preview=extracted[:220],
        )

    def training_status(self, *, avatar_id: str) -> dict:
        avatar = self.repository.get_avatar(avatar_id)
        if avatar is None:
            raise NotFoundError("avatar_not_found")
        docs = self.repository.list_training_documents(avatar_id=avatar_id)
        chunks = self.repository.list_training_chunks(avatar_id=avatar_id)
        vectors = self.repository.count_training_chunk_vectors(avatar_id=avatar_id)
        total_chars = sum(len(d.content_text) for d in docs)
        return {
            "avatarId": avatar_id,
            "documents": len(docs),
            "chunks": len(chunks),
            "vectors": vectors,
            "totalChars": total_chars,
            "teacherMode": True,
        }

    def _build_teacher_answer(
        self,
        *,
        avatar_id: str,
        user_text: str,
        conversation: Conversation | None = None,
    ) -> LLMResult:
        docs = self.repository.list_training_documents(avatar_id=avatar_id)
        if not docs:
            return LLMResult(
                text=self._finalize_teacher_text(
                    "I am your matric teacher. No training books are uploaded yet. "
                    "Please upload matric-level PDF/books first, then ask your question."
                ),
                emotion="neutral",
            )

        effective_question = self._effective_question(user_text=user_text, conversation=conversation)
        retrieval_question = self._question_for_retrieval(effective_question)
        is_math_question = self._is_math_question(retrieval_question)
        if self._is_casual_non_academic_question(retrieval_question):
            return LLMResult(
                text=self._finalize_teacher_text(
                    "I am your matric-level teacher only. I can answer school syllabus questions "
                    "from your uploaded books. Please ask a matric subject question."
                ),
                emotion="neutral",
            )
        retrieved_chunks = self._retrieve_relevant_chunks_scored(
            avatar_id=avatar_id,
            question=retrieval_question,
            top_k=7 if is_math_question else 5,
            prefer_math=is_math_question,
        )
        if not retrieved_chunks or not self._retrieval_is_confident(retrieval_question, retrieved_chunks):
            return LLMResult(
                text=self._finalize_teacher_text(
                    "I can answer only from uploaded books. I could not find this question clearly in the uploaded "
                    "material. Please ask from uploaded chapters or upload the relevant chapter/book."
                ),
                emotion="neutral",
            )

        top_chunks = [item.text for item in retrieved_chunks]
        context = " ".join(top_chunks)
        local_answer = (
            self._build_math_structured_answer(question=retrieval_question, context=context)
            if is_math_question
            else self._summarize_from_context(question=retrieval_question, context=context)
        )

        lines = [
            "As your matric teacher, here is the answer from your uploaded material:",
            local_answer,
            "I am strictly limited to uploaded books only.",
        ]
        lines.append("If you want, I can also give step-by-step practice questions from the same topic.")
        local_text = " ".join(lines)

        # Use external providers for natural language while preserving local fallback.
        if not isinstance(self.llm_provider, DeterministicLLMProvider):
            prompt = self._build_rag_user_prompt(
                question=retrieval_question,
                context=context,
                is_math_question=is_math_question,
                internet_context="",
                original_question=effective_question,
            )
            try:
                llm_result = self.llm_provider.complete(
                    persona=self._build_rag_system_prompt(allow_internet_notes=False),
                    user_text=prompt,
                )
                text = self._normalize_whitespace(llm_result.text)
                if text:
                    return LLMResult(
                        text=self._finalize_teacher_text(text, can_use_llm_rewriter=True),
                        emotion=(llm_result.emotion or "confident"),
                    )
            except Exception:
                if not self.llm_fallback_enabled:
                    raise

        return LLMResult(text=self._finalize_teacher_text(local_text), emotion="confident")

    def _should_use_internet_fallback(self, *, user_text: str, conversation: Conversation | None) -> bool:
        if self.config.strict_book_only_mode:
            return False
        if not self.config.internet_lookup_enabled:
            return False
        if conversation is None:
            return False
        has_prior_teacher_reply = any(msg.role == Role.ASSISTANT for msg in conversation.messages)
        if not has_prior_teacher_reply:
            return False

        lowered = user_text.lower()
        indicators = (
            "i don't understand",
            "i do not understand",
            "didn't understand",
            "did not understand",
            "not clear",
            "confusing",
            "explain again",
            "again please",
            "simpler",
            "easy words",
            "can you explain",
            "one more time",
        )
        return any(marker in lowered for marker in indicators)

    def _build_web_query(self, *, question: str, local_context: str) -> str:
        keywords = self._top_keywords(local_context, max_words=5)
        if keywords:
            return f"{question} {' '.join(keywords)} matric"
        return f"{question} matric"

    def _is_casual_non_academic_question(self, text: str) -> bool:
        lowered = text.lower()
        casual_markers = (
            "movie",
            "song",
            "music",
            "cricket score",
            "football score",
            "joke",
            "meme",
            "celebrity",
            "fashion",
            "shopping",
            "romantic",
            "travel plan",
            "restaurant",
            "weather",
        )
        return any(marker in lowered for marker in casual_markers)

    def _math_profile_strength(self, docs: list[TrainingDocument]) -> float:
        if not docs:
            return 0.0
        math_like = 0
        for doc in docs:
            if self._math_signal_score(f"{doc.filename} {doc.content_text}") >= 0.09:
                math_like += 1
        return math_like / len(docs)

    def _is_math_question(self, text: str) -> bool:
        return self._math_signal_score(text) >= 0.08

    def _math_signal_score(self, text: str) -> float:
        lowered = text.lower()
        words = re.findall(r"[a-zA-Z]{2,}", lowered)
        if not words and not re.search(r"\d", lowered):
            return 0.0

        math_keywords = {
            "math", "mathematics", "algebra", "geometry", "trigonometry", "calculus",
            "equation", "factor", "quadratic", "polynomial", "theorem", "proof",
            "matrix", "determinant", "vector", "ratio", "proportion", "percentage",
            "fraction", "integer", "derivative", "integral", "simplify", "solve",
            "formula", "angle", "triangle", "circle", "perimeter", "area", "volume",
            "mean", "median", "mode", "probability", "statistics",
        }
        kw_hits = sum(1 for word in words if word in math_keywords)
        kw_ratio = kw_hits / max(1, len(words))

        equation_hits = len(re.findall(r"\d+\s*[\+\-\*/=\^]\s*\d+|[a-z]\s*[\+\-\*/=\^]\s*[a-z0-9]", lowered))
        symbol_density = len(re.findall(r"[\+\-\*/=\^]", lowered)) / max(1, len(lowered))
        number_density = len(re.findall(r"\b\d+(?:\.\d+)?\b", lowered)) / max(1, len(words))
        return (kw_ratio * 1.8) + min(0.6, equation_hits * 0.1) + (symbol_density * 3.0) + (number_density * 0.4)

    def _effective_question(self, *, user_text: str, conversation: Conversation | None) -> str:
        current = user_text.strip()
        if conversation is None:
            return current
        recent_user = [m.text for m in conversation.messages if m.role == Role.USER]
        if recent_user:
            previous = recent_user[-1].strip()
            lowered = current.lower()
            referential = any(
                marker in lowered
                for marker in ("this", "that", "it", "again", "once more", "same", "previous", "above")
            )
            if referential:
                current = f"{previous}. Follow-up: {current}"

        image_context = self.repository.pop_latest_unconsumed_image_context(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
        )
        if image_context is None:
            return current

        excerpt = self._question_context_excerpt(image_context.content_text, limit=1700)
        if not excerpt:
            return current
        return (
            f"{current}\n\n"
            f"Student image context ({image_context.filename}): {excerpt}"
        )

    def _question_for_retrieval(self, question: str) -> str:
        clean = self._normalize_whitespace(question)
        if not clean:
            return clean
        if not self.config.hinglish_enabled:
            return clean
        if not self._looks_hinglish_or_hindi(clean):
            return clean

        translated = self._translate_hinglish_to_english(clean)
        if translated:
            return translated
        return self._heuristic_hinglish_to_english(clean)

    def _looks_hinglish_or_hindi(self, text: str) -> bool:
        lowered = text.lower()
        if re.search(r"[\u0900-\u097F]", text):
            return True
        markers = (
            "kya",
            "kaise",
            "kyun",
            "kyu",
            "samjhao",
            "samjha",
            "batao",
            "matlab",
            "ka ",
            "ke ",
            "ki ",
            "nahi",
            "hain",
            "hai",
            "karna",
            "karo",
            "question ka",
            "chapter ka",
            "iske bare",
            "iska",
        )
        return any(token in lowered for token in markers)

    def _translate_hinglish_to_english(self, text: str) -> str:
        if isinstance(self.llm_provider, DeterministicLLMProvider):
            return ""
        try:
            result = self.llm_provider.complete(
                persona=(
                    "You are a translation engine. Convert Hinglish/Hindi student questions into concise English "
                    "questions for retrieval. Return only English text."
                ),
                user_text=f"Translate to English only: {text}",
            )
        except Exception:
            return ""
        translated = self._normalize_whitespace(result.text)
        translated = re.sub(r"^(english|translation)\s*:\s*", "", translated, flags=re.I)
        return translated

    def _heuristic_hinglish_to_english(self, text: str) -> str:
        lowered = text.lower()
        replacements = [
            (r"\bkya\b", "what"),
            (r"\bkaise\b", "how"),
            (r"\bkyu?n\b", "why"),
            (r"\bkab\b", "when"),
            (r"\bkahan\b", "where"),
            (r"\bka\b", "of"),
            (r"\bke\b", "of"),
            (r"\bki\b", "of"),
            (r"\bka matlab\b", "meaning"),
            (r"\bsamjhao\b", "explain"),
            (r"\bsamjha[o]?\b", "explain"),
            (r"\bbatao\b", "tell"),
            (r"\bkaro\b", "do"),
            (r"\bsolve karo\b", "solve"),
            (r"\bnahi\b", "not"),
            (r"\biske bare mein\b", "about this"),
            (r"\biske bare\b", "about this"),
            (r"\biska\b", "its"),
            (r"\baur\b", "and"),
        ]
        out = lowered
        for pattern, replacement in replacements:
            out = re.sub(pattern, replacement, out)
        out = re.sub(r"[\u0900-\u097F]+", " ", out)
        out = self._normalize_whitespace(out)
        return out or text

    def _finalize_teacher_text(self, text: str, *, can_use_llm_rewriter: bool = False) -> str:
        clean = self._normalize_whitespace(text)
        if not clean or not self.config.hinglish_enabled:
            return clean
        if self._looks_hinglish_or_hindi(clean):
            return clean
        llm_rewriter_allowed = type(self.llm_provider).__name__ in {
            "OpenAIChatProvider",
            "GeminiChatProvider",
            "OllamaChatProvider",
        }
        if can_use_llm_rewriter and llm_rewriter_allowed:
            rewritten = self._rewrite_answer_to_hinglish(clean)
            if rewritten:
                return rewritten
        return f"{clean} Hinglish mein: Main is topic ko simple Hindi + English mix mein samjha raha hoon."

    def _rewrite_answer_to_hinglish(self, text: str) -> str:
        try:
            result = self.llm_provider.complete(
                persona=(
                    "You are SP Sir language rewriter. Rewrite the text in natural Hinglish (Hindi + English mix) "
                    "using Roman script, preserving facts exactly."
                ),
                user_text=f"Rewrite in Hinglish (Roman script), keep meaning exactly:\n{text}",
            )
        except Exception:
            return ""
        rewritten = self._normalize_whitespace(result.text)
        return rewritten

    def _build_math_structured_answer(self, *, question: str, context: str) -> str:
        concept = self._summarize_from_context(question=question, context=context)
        formula = self._extract_formula_line(context)
        steps = self._build_math_steps(question=question, concept=concept, formula=formula)
        final_line = "Final Answer: Based on your uploaded maths book context, this is the required result."
        lines = [
            "Concept:",
            concept,
            "Formula:",
            formula,
            "Step-by-step:",
            steps,
            final_line,
        ]
        return " ".join(lines)

    def _extract_formula_line(self, context: str) -> str:
        explicit = re.findall(r"[A-Za-z0-9\(\)\s]+\s*=\s*[A-Za-z0-9\+\-\*/\^\(\)\s]+", context)
        if explicit:
            return self._normalize_whitespace(explicit[0])[:180]

        candidates = re.split(r"(?<=[.!?])\s+", context)
        for sentence in candidates:
            lowered = sentence.lower()
            if any(k in lowered for k in ("formula", "equation", "theorem", "rule", "identity")):
                s = self._normalize_whitespace(sentence)
                if s:
                    return s[:180]
        return "Use the standard formula from the relevant chapter in your uploaded maths book."

    def _build_math_steps(self, *, question: str, concept: str, formula: str) -> str:
        question_hint = self._normalize_whitespace(question)
        concept_hint = self._normalize_whitespace(concept)[:220]
        formula_hint = self._normalize_whitespace(formula)[:160]
        return (
            f"Step 1: Identify the given values/terms in the question ({question_hint}). "
            f"Step 2: Choose the correct method from context ({formula_hint}). "
            f"Step 3: Substitute values carefully and simplify each operation in order. "
            f"Step 4: Verify units/signs and state the final result clearly. "
            f"Reference: {concept_hint}"
        )

    def _top_keywords(self, text: str, *, max_words: int) -> list[str]:
        tokens = self._tokenize(text)
        freq: dict[str, int] = {}
        for token in tokens:
            if len(token) < 4:
                continue
            freq[token] = freq.get(token, 0) + 1
        ranked = sorted(freq.items(), key=lambda pair: pair[1], reverse=True)
        return [word for word, _ in ranked[:max_words]]

    def _search_web_snippets(self, *, query: str, max_snippets: int) -> list[str]:
        if not query.strip():
            return []
        url = (
            "https://api.duckduckgo.com/"
            f"?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "avatar-teacher/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.config.internet_lookup_timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []

        out: list[str] = []
        abstract = self._normalize_whitespace(str(payload.get("AbstractText", "")))
        if abstract:
            out.append(abstract)

        def _collect(items) -> None:
            for item in items:
                if len(out) >= max_snippets:
                    return
                if isinstance(item, dict) and "Topics" in item:
                    _collect(item.get("Topics") or [])
                    if len(out) >= max_snippets:
                        return
                text = self._normalize_whitespace(str((item or {}).get("Text", "")))
                if text and text not in out:
                    out.append(text)
                    if len(out) >= max_snippets:
                        return

        _collect(payload.get("RelatedTopics") or [])
        return out[:max_snippets]

    def transcribe_audio(self, *, audio_bytes: bytes, mime_type: str) -> str:
        text = self.stt_provider.transcribe(audio_bytes=audio_bytes, mime_type=mime_type).text
        return text.strip()

    def synthesize_audio(self, *, text: str, voice_id: str):
        clean = self._validate_tts_text(text)
        return self.tts_provider.synthesize(text=clean, voice_id=voice_id)

    def ai_health(self) -> dict:
        try:
            probe = self.llm_provider.complete(persona="System", user_text="Reply with OK")
            ok = bool(probe.text.strip())
            return {
                "ok": ok,
                "provider": self.ai_provider_name,
                "fallbackEnabled": self.llm_fallback_enabled,
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": self.ai_provider_name,
                "fallbackEnabled": self.llm_fallback_enabled,
                "error": str(exc),
            }

    def system_status(self) -> dict:
        return {
            "aiProvider": self.ai_provider_name,
            "ttsProvider": self.tts_provider_name,
            "llmFallbackEnabled": self.llm_fallback_enabled,
            "internetLookupEnabled": self.config.internet_lookup_enabled,
            "strictBookOnlyMode": self.config.strict_book_only_mode,
            "hinglishEnabled": self.config.hinglish_enabled,
        }

    def _ensure_user(self, user_id: str) -> None:
        if self.repository.get_user(user_id) is None:
            raise NotFoundError("user_not_found")

    def _validate_text(self, text: str) -> str:
        clean = text.strip()
        if not clean:
            raise ValidationError("message_empty")
        if len(clean) > self.config.max_message_chars:
            raise ValidationError("message_too_long")
        return clean

    def _validate_tts_text(self, text: str) -> str:
        clean = self._normalize_whitespace(text)
        if not clean:
            raise ValidationError("message_empty")

        limit = max(200, self.config.max_tts_chars)
        if len(clean) <= limit:
            return clean

        clipped = clean[:limit].rstrip()
        cut_points = (
            clipped.rfind(". "),
            clipped.rfind("! "),
            clipped.rfind("? "),
            clipped.rfind("; "),
            clipped.rfind(", "),
        )
        best_cut = max(cut_points)
        if best_cut >= int(limit * 0.7):
            clipped = clipped[: best_cut + 1].rstrip()
        if clipped.endswith(("...", "…")):
            return clipped
        return f"{clipped} ..."

    def _detect_source_type(self, filename: str) -> str:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return "pdf"
        if lower.endswith(".docx"):
            return "docx"
        if lower.endswith(".txt") or lower.endswith(".md"):
            return "text"
        return "book"

    def _question_context_excerpt(self, text: str, *, limit: int) -> str:
        clean = self._normalize_whitespace(text)
        if len(clean) <= limit:
            return clean
        clipped = clean[:limit].rstrip()
        cut = max(clipped.rfind(". "), clipped.rfind("? "), clipped.rfind("! "), clipped.rfind("; "))
        if cut >= int(limit * 0.6):
            clipped = clipped[: cut + 1].rstrip()
        return clipped

    def _looks_like_image_file(self, *, filename: str, file_bytes: bytes) -> bool:
        lower = filename.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif")):
            return True
        if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return True
        if file_bytes.startswith(b"\xff\xd8\xff"):
            return True
        if file_bytes.startswith((b"GIF87a", b"GIF89a")):
            return True
        if file_bytes.startswith(b"BM"):
            return True
        if len(file_bytes) > 12 and file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WEBP":
            return True
        return False

    def _extract_image_text(self, *, filename: str, file_bytes: bytes) -> str:
        for extractor in (
            self._extract_image_with_pytesseract,
            self._extract_image_with_tesseract_cli,
            self._extract_image_with_gemini_vision,
        ):
            text = extractor(filename=filename, file_bytes=file_bytes)
            if text:
                return self._normalize_whitespace(text)
        return ""

    def _image_ocr_support_available(self) -> bool:
        has_tesseract = bool(shutil.which("tesseract"))
        has_python_ocr = False
        try:
            import PIL  # type: ignore
            import pytesseract  # type: ignore

            has_python_ocr = bool(PIL) and bool(pytesseract)
        except Exception:
            has_python_ocr = False
        has_gemini_vision = bool(str(os.getenv("GEMINI_API_KEY", "")).strip())
        return has_tesseract or has_python_ocr or has_gemini_vision

    def _extract_image_with_pytesseract(self, *, filename: str, file_bytes: bytes) -> str:
        del filename
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
        except Exception:
            return ""
        try:
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image, lang=(self.config.ocr_language.strip() or "eng"))
        except Exception:
            return ""
        return self._normalize_whitespace(text)

    def _extract_image_with_tesseract_cli(self, *, filename: str, file_bytes: bytes) -> str:
        if not shutil.which("tesseract"):
            return ""
        suffix = Path(filename).suffix.lower() or ".png"
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / f"upload{suffix}"
            image_path.write_bytes(file_bytes)
            cmd = ["tesseract", str(image_path), "stdout", "-l", (self.config.ocr_language.strip() or "eng")]
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
            except Exception:
                return ""
            return self._normalize_whitespace(proc.stdout or "")

    def _extract_image_with_gemini_vision(self, *, filename: str, file_bytes: bytes) -> str:
        api_key = str(os.getenv("GEMINI_API_KEY", "")).strip()
        if not api_key:
            return ""
        if len(file_bytes) > 18_000_000:
            return ""

        mime_type = self._guess_image_mime_type(filename=filename, file_bytes=file_bytes)
        image_b64 = base64.b64encode(file_bytes).decode("ascii")
        model = str(
            os.getenv("GEMINI_VISION_MODEL", "") or os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        ).strip()
        model = model or "gemini-flash-latest"
        base_url = str(os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")).strip().rstrip("/")
        model_path = quote(model, safe=":-._")
        url = f"{base_url}/v1beta/models/{model_path}:generateContent"
        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Extract all readable text from this image. "
                                "Return plain text only without commentary."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "X-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return ""
        return self._extract_text_from_gemini_payload(payload)

    def _extract_text_from_gemini_payload(self, payload: dict) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        content = (candidates[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        collected: list[str] = []
        for part in parts:
            text = str((part or {}).get("text", "")).strip()
            if text:
                collected.append(text)
        return self._normalize_whitespace(" ".join(collected))

    def _guess_image_mime_type(self, *, filename: str, file_bytes: bytes) -> str:
        lower = filename.lower()
        if lower.endswith(".png") or file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if lower.endswith((".jpg", ".jpeg")) or file_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if lower.endswith(".webp") or (len(file_bytes) > 12 and file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WEBP"):
            return "image/webp"
        if lower.endswith(".gif") or file_bytes.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if lower.endswith(".bmp") or file_bytes.startswith(b"BM"):
            return "image/bmp"
        if lower.endswith((".tif", ".tiff")):
            return "image/tiff"
        if lower.endswith((".heic", ".heif")):
            return "image/heic"
        return "image/png"

    def _extract_training_text(self, *, filename: str, file_bytes: bytes) -> str:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return self._extract_pdf_text(file_bytes)
        if lower.endswith(".docx"):
            return self._extract_docx_text(file_bytes)
        if lower.endswith(".doc"):
            return self._extract_doc_text(file_bytes)
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin1", errors="ignore")

    def _extract_pdf_text(self, file_bytes: bytes) -> str:
        # Preferred extraction path: PyMuPDF/pdfplumber for robust encoded PDFs.
        preferred = self._extract_pdf_text_with_python_libraries(file_bytes)
        if len(preferred) >= 40:
            return preferred

        # Stream-aware extraction for common PDF encodings.
        parts: list[str] = []
        for header, payload in self._iter_pdf_streams(file_bytes):
            decoded = self._decode_pdf_stream_payload(header=header, payload=payload)
            if not decoded:
                continue
            parts.extend(self._extract_pdf_stream_text(decoded))

        raw = file_bytes.decode("latin1", errors="ignore")
        if not parts:
            # Fallback for simple uncompressed PDFs where text appears in-line.
            parts.extend(self._extract_pdf_stream_text(raw))

        extracted = self._normalize_whitespace(" ".join([p for p in parts if p and p.strip()]))

        # Fallback to native tools for complex PDFs (object streams/font encodings).
        if len(extracted) < 120:
            tool_text = self._extract_pdf_text_with_system_tools(file_bytes)
            if tool_text:
                extracted = self._normalize_whitespace(f"{extracted} {tool_text}".strip())

        # Last-resort fallback: salvage human-readable strings from raw PDF bytes.
        if len(extracted) < 40:
            raw_text = self._extract_pdf_text_from_raw_bytes(file_bytes)
            if raw_text:
                extracted = self._normalize_whitespace(f"{extracted} {raw_text}".strip())

        # OCR fallback for scanned/image-only PDFs.
        if len(extracted) < 40 and self.config.ocr_enabled:
            ocr_text = self._extract_pdf_text_with_ocr(file_bytes)
            if ocr_text:
                extracted = self._normalize_whitespace(f"{extracted} {ocr_text}".strip())

        return extracted

    def _extract_pdf_text_with_python_libraries(self, file_bytes: bytes) -> str:
        for extractor in (self._extract_with_pymupdf, self._extract_with_pdfplumber):
            text = extractor(file_bytes)
            if text:
                return text
        return ""

    def _extract_with_pymupdf(self, file_bytes: bytes) -> str:
        try:
            import pymupdf  # type: ignore
        except Exception:
            try:
                import fitz as pymupdf  # type: ignore
            except Exception:
                return ""
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception:
            return ""
        parts: list[str] = []
        try:
            for page in doc:
                try:
                    text = page.get_text("text") or ""
                except Exception:
                    text = ""
                if text:
                    parts.append(text)
        finally:
            try:
                doc.close()
            except Exception:
                pass
        return self._normalize_whitespace(" ".join(parts))

    def _extract_with_pdfplumber(self, file_bytes: bytes) -> str:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return ""
        parts: list[str] = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text:
                        parts.append(text)
        except Exception:
            return ""
        return self._normalize_whitespace(" ".join(parts))

    def _extract_docx_text(self, file_bytes: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                xml_blobs: list[str] = []
                for name in zf.namelist():
                    if not name.startswith("word/") or not name.endswith(".xml"):
                        continue
                    if "/media/" in name:
                        continue
                    try:
                        xml_blobs.append(zf.read(name).decode("utf-8", errors="ignore"))
                    except Exception:
                        continue
        except Exception:
            return ""

        if not xml_blobs:
            return ""

        fragments: list[str] = []
        for blob in xml_blobs:
            try:
                root = ET.fromstring(blob)
            except ET.ParseError:
                continue
            for node in root.iter():
                tag = str(node.tag)
                if tag.endswith("}t") and node.text:
                    fragments.append(node.text)
        text = html.unescape(" ".join(fragments))
        return self._normalize_whitespace(text)

    def _extract_doc_text(self, file_bytes: bytes) -> str:
        if shutil.which("textutil"):
            with tempfile.TemporaryDirectory() as tmp:
                doc_path = Path(tmp) / "input.doc"
                doc_path.write_bytes(file_bytes)
                cmd = ["textutil", "-convert", "txt", "-stdout", str(doc_path)]
                try:
                    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                except Exception:
                    return ""
                return self._normalize_whitespace(proc.stdout)
        try:
            return self._normalize_whitespace(file_bytes.decode("latin1", errors="ignore"))
        except Exception:
            return ""

    def _extract_pdf_text_with_system_tools(self, file_bytes: bytes) -> str:
        for extractor in (self._extract_with_pdftotext, self._extract_with_textutil):
            text = extractor(file_bytes)
            if text:
                return text
        return ""

    def _extract_pdf_text_with_ocr(self, file_bytes: bytes) -> str:
        for extractor in (self._extract_with_ocrmypdf, self._extract_with_pdftoppm_tesseract):
            text = extractor(file_bytes)
            if text:
                return self._normalize_whitespace(text)
        return ""

    def _extract_with_ocrmypdf(self, file_bytes: bytes) -> str:
        if not shutil.which("ocrmypdf"):
            return ""
        with tempfile.TemporaryDirectory() as tmp:
            in_pdf = Path(tmp) / "input.pdf"
            out_pdf = Path(tmp) / "ocr-output.pdf"
            sidecar = Path(tmp) / "sidecar.txt"
            in_pdf.write_bytes(file_bytes)
            cmd = [
                "ocrmypdf",
                "--force-ocr",
                "--skip-text",
                "--sidecar",
                str(sidecar),
                "--optimize",
                "0",
                str(in_pdf),
                str(out_pdf),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except Exception:
                return ""
            if sidecar.exists():
                try:
                    return sidecar.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    return ""
        return ""

    def _extract_with_pdftoppm_tesseract(self, file_bytes: bytes) -> str:
        if not shutil.which("pdftoppm") or not shutil.which("tesseract"):
            return ""
        with tempfile.TemporaryDirectory() as tmp:
            in_pdf = Path(tmp) / "input.pdf"
            in_pdf.write_bytes(file_bytes)
            image_prefix = Path(tmp) / "page"
            max_pages = max(1, int(self.config.ocr_max_pages))
            cmd = ["pdftoppm", "-png", "-f", "1", "-l", str(max_pages), str(in_pdf), str(image_prefix)]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
            except Exception:
                return ""

            images = sorted(Path(tmp).glob("page-*.png"))
            if not images:
                return ""

            fragments: list[str] = []
            lang = self.config.ocr_language.strip() or "eng"
            for img in images:
                t_cmd = ["tesseract", str(img), "stdout", "-l", lang]
                try:
                    proc = subprocess.run(t_cmd, check=True, capture_output=True, text=True, timeout=60)
                except Exception:
                    continue
                text = (proc.stdout or "").strip()
                if text:
                    fragments.append(text)
            return " ".join(fragments).strip()

    def _extract_with_pdftotext(self, file_bytes: bytes) -> str:
        if not shutil.which("pdftotext"):
            return ""
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "input.pdf"
            pdf_path.write_bytes(file_bytes)
            cmd = ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"]
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            except Exception:
                return ""
            return self._normalize_whitespace(proc.stdout)

    def _extract_with_textutil(self, file_bytes: bytes) -> str:
        if not shutil.which("textutil"):
            return ""
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "input.pdf"
            pdf_path.write_bytes(file_bytes)
            cmd = ["textutil", "-convert", "txt", "-stdout", str(pdf_path)]
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            except Exception:
                return ""
            return self._normalize_whitespace(proc.stdout)

    def _extract_pdf_text_from_raw_bytes(self, file_bytes: bytes) -> str:
        raw = file_bytes.decode("latin1", errors="ignore")
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9 ,;:!?'\"()\[\]/\-]{10,}", raw)
        seen: set[str] = set()
        out: list[str] = []
        for candidate in candidates:
            cleaned = self._normalize_whitespace(candidate)
            if len(cleaned) < 12:
                continue
            alpha_count = sum(1 for ch in cleaned if ch.isalpha())
            if alpha_count < 6:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(cleaned)
            if len(out) >= 80:
                break
        return " ".join(out)

    def _iter_pdf_streams(self, file_bytes: bytes) -> list[tuple[str, bytes]]:
        pattern = re.compile(rb"(<<.*?>>)\s*stream\r?\n(.*?)\r?\nendstream", flags=re.S)
        out: list[tuple[str, bytes]] = []
        for match in pattern.finditer(file_bytes):
            header = match.group(1).decode("latin1", errors="ignore")
            payload = match.group(2)
            out.append((header, payload))
        return out

    def _decode_pdf_stream_payload(self, *, header: str, payload: bytes) -> str:
        data = payload
        lowered = header.lower()

        if "ascii85decode" in lowered:
            try:
                data = self._decode_ascii85_payload(data)
            except Exception:
                return ""
        if "asciihexdecode" in lowered:
            try:
                data = self._decode_asciihex_payload(data)
            except Exception:
                return ""
        if "flatedecode" in lowered:
            data = self._decompress_flate_payload(data)
            if not data:
                return ""

        return data.decode("latin1", errors="ignore")

    def _decode_ascii85_payload(self, data: bytes) -> bytes:
        cleaned = re.sub(rb"\s+", b"", data)
        if b"~>" in cleaned:
            cleaned = cleaned[: cleaned.find(b"~>") + 2]
        return base64.a85decode(cleaned, adobe=True)

    def _decode_asciihex_payload(self, data: bytes) -> bytes:
        cleaned = re.sub(rb"\s+", b"", data).rstrip(b">")
        if len(cleaned) % 2 == 1:
            cleaned += b"0"
        return unhexlify(cleaned)

    def _decompress_flate_payload(self, data: bytes) -> bytes:
        candidates = [data, data.rstrip(b"\r\n")]
        for candidate in candidates:
            try:
                return zlib.decompress(candidate)
            except zlib.error:
                pass
            try:
                return zlib.decompress(candidate, -15)
            except zlib.error:
                pass
        return b""

    def _extract_pdf_stream_text(self, stream_text: str) -> list[str]:
        out: list[str] = []
        for s in re.findall(r"\((.*?)\)\s*Tj", stream_text, flags=re.S):
            out.append(self._unescape_pdf_text(s))
        for s in re.findall(r"<([0-9A-Fa-f\s]{2,})>\s*Tj", stream_text, flags=re.S):
            out.append(self._decode_pdf_hex_text(s))
        for arr in re.findall(r"\[(.*?)\]\s*TJ", stream_text, flags=re.S):
            for s in re.findall(r"\((.*?)\)", arr, flags=re.S):
                out.append(self._unescape_pdf_text(s))
            for s in re.findall(r"<([0-9A-Fa-f\s]{2,})>", arr, flags=re.S):
                out.append(self._decode_pdf_hex_text(s))
        return [t for t in out if t and t.strip()]

    def _decode_pdf_hex_text(self, value: str) -> str:
        compact = re.sub(r"\s+", "", value)
        if len(compact) % 2 == 1:
            compact += "0"
        try:
            data = unhexlify(compact.encode("ascii"))
        except Exception:
            return ""
        if data.startswith(b"\xfe\xff"):
            try:
                return data[2:].decode("utf-16-be", errors="ignore")
            except Exception:
                return data.decode("latin1", errors="ignore")
        return data.decode("latin1", errors="ignore")

    def _unescape_pdf_text(self, value: str) -> str:
        def _octal_repl(match: re.Match[str]) -> str:
            digits = match.group(1)
            try:
                return chr(int(digits, 8))
            except ValueError:
                return ""

        value = value.replace(r"\(", "(").replace(r"\)", ")").replace(r"\n", " ").replace(r"\r", " ")
        value = value.replace(r"\t", " ").replace(r"\b", " ").replace(r"\f", " ")
        value = re.sub(r"\\([0-7]{1,3})", _octal_repl, value)
        return value

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _chunk_text(self, text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
        if len(text) <= chunk_chars:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_chars)
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = max(0, end - overlap_chars)
        return [c for c in chunks if c]

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        provider = self.config.embedding_provider.strip().lower()
        if provider not in {"local", "ollama", "auto"}:
            provider = "local"

        should_try_ollama = provider == "ollama" or (
            provider == "auto" and str(self.ai_provider_name).lower() in {"ollama", "auto"}
        )
        if should_try_ollama and self._ollama_embedding_available:
            vectors = self._generate_ollama_embeddings(texts)
            if vectors:
                return vectors
            self._ollama_embedding_available = False

        if provider == "ollama":
            # Keep training resilient when Ollama embeddings endpoint is unavailable.
            return [self._local_embedding(text) for text in texts]

        return [self._local_embedding(text) for text in texts]

    def _generate_ollama_embeddings(self, texts: list[str]) -> list[list[float]]:
        base = self.config.embedding_base_url.rstrip("/")
        model = self.config.embedding_model.strip()
        if not base or not model:
            return []

        # Newer Ollama endpoint with batch input.
        try:
            payload = self._post_json(
                url=f"{base}/api/embed",
                body={"model": model, "input": texts},
                timeout=self.config.embedding_timeout_seconds,
            )
            raw_vectors = payload.get("embeddings")
            if isinstance(raw_vectors, list) and len(raw_vectors) == len(texts):
                out = [self._normalize_vector(v) for v in raw_vectors if isinstance(v, list)]
                if len(out) == len(texts):
                    return out
        except Exception:
            pass

        # Compatibility fallback endpoint.
        out: list[list[float]] = []
        for text in texts:
            try:
                payload = self._post_json(
                    url=f"{base}/api/embeddings",
                    body={"model": model, "prompt": text},
                    timeout=self.config.embedding_timeout_seconds,
                )
            except Exception:
                return []
            raw = payload.get("embedding")
            if not isinstance(raw, list):
                return []
            out.append(self._normalize_vector(raw))
        return out

    def _post_json(self, *, url: str, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _local_embedding(self, text: str) -> list[float]:
        dim = max(64, int(self.config.embedding_dimension))
        vector = [0.0] * dim
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            self._accumulate_hashed_feature(vector, token, weight=1.0)
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    self._accumulate_hashed_feature(vector, token[i:i + 3], weight=0.35)
        return self._normalize_vector(vector)

    def _accumulate_hashed_feature(self, vector: list[float], feature: str, *, weight: float) -> None:
        dim = len(vector)
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dim
        sign = -1.0 if (digest[8] & 1) else 1.0
        vector[index] += sign * weight

    def _normalize_vector(self, values: list[float]) -> list[float]:
        if not values:
            return []
        casted = [float(v) for v in values]
        norm = math.sqrt(sum(v * v for v in casted))
        if norm <= 1e-12:
            return [0.0 for _ in casted]
        return [v / norm for v in casted]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        size = min(len(a), len(b))
        if size <= 0:
            return 0.0
        return float(sum(a[i] * b[i] for i in range(size)))

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens: list[str] = []

        # General word tokens.
        tokens.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9]{1,}", lowered))
        # Numbers matter in mathematics questions.
        tokens.extend(re.findall(r"\d+(?:\.\d+)?", lowered))
        # Single-letter variable names.
        tokens.extend(re.findall(r"\b[a-z]\b", lowered))
        # Equation-style tokens.
        tokens.extend(re.findall(r"[a-z]\^\d+|\d+[a-z]|[a-z]\d+|[+\-*/=]", lowered))

        variable_names = {"x", "y", "z", "a", "b", "c", "m", "n", "p", "q"}
        filtered: list[str] = []
        for token in tokens:
            if token in self._token_stop_words and token not in variable_names:
                continue
            if len(token) == 1 and token.isalpha() and token not in variable_names:
                continue
            filtered.append(token)
            # Lightweight singularization helps matching speech text with book phrasing (e.g., humans -> human).
            if token.isalpha() and len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                filtered.append(token[:-1])
        return filtered

    def _expand_question_for_retrieval(self, question: str) -> str:
        lowered = question.lower()
        expansions: list[str] = []
        concept_aliases = {
            "brain": "nervous system neuron neurons cerebrum cerebellum medulla spinal cord",
            "human brain": "nervous system neuron neurons cerebrum cerebellum medulla spinal cord",
            "nervous": "neuron neurons spinal cord brain",
            "photosynthesis": "chlorophyll chloroplast sunlight glucose carbon dioxide oxygen",
            "respiration": "breathing oxygen carbon dioxide mitochondria",
            "heart": "circulatory system atrium ventricle blood vessel artery vein",
            "force": "motion acceleration velocity newton law inertia",
            "cell": "nucleus cytoplasm membrane organelle tissue",
        }
        for key, alias in concept_aliases.items():
            if key in lowered:
                expansions.append(alias)
        if not expansions:
            return question
        return f"{question} {' '.join(expansions)}"

    def _is_teacher_scope_question(self, user_text: str) -> bool:
        lowered = user_text.lower()
        tokens = set(self._tokenize(lowered))
        if tokens & self._teacher_scope_keywords:
            return True

        return any(
            marker in lowered
            for marker in (
                "teach",
                "explain",
                "solve",
                "formula",
                "definition",
                "chapter",
                "exercise",
                "homework",
                "school",
                "class",
            )
        )

    def _retrieve_relevant_chunks(
        self,
        *,
        avatar_id: str,
        question: str,
        top_k: int,
        prefer_math: bool = False,
    ) -> list[str]:
        retrieved = self._retrieve_relevant_chunks_scored(
            avatar_id=avatar_id,
            question=question,
            top_k=top_k,
            prefer_math=prefer_math,
        )
        return [item.text for item in retrieved]

    def _retrieve_relevant_chunks_scored(
        self,
        *,
        avatar_id: str,
        question: str,
        top_k: int,
        prefer_math: bool = False,
    ) -> list[RetrievedChunk]:
        retrieval_question = self._expand_question_for_retrieval(question)
        vector_rows = self.repository.list_training_chunk_vectors(avatar_id=avatar_id)
        chunks: list[str]
        if vector_rows:
            chunks = [row.chunk_text for row in vector_rows]
        else:
            chunks = self.repository.list_training_chunks(avatar_id=avatar_id)
        if not chunks:
            return []

        q_tokens_list = self._tokenize(retrieval_question)
        q_tokens = set(q_tokens_list)
        q_phrase = self._normalize_whitespace(question.lower())
        q_vector = self._generate_embeddings([retrieval_question])[0]
        scored: list[RetrievedChunk] = []
        vector_by_chunk: dict[str, list[float]] = {}
        if vector_rows:
            for row in vector_rows:
                vector_by_chunk[row.chunk_text] = row.embedding

        for chunk in chunks:
            chunk_tokens_list = self._tokenize(chunk)
            if not chunk_tokens_list:
                continue
            c_tokens = set(chunk_tokens_list)
            overlap = len(q_tokens & c_tokens)

            token_freq: dict[str, int] = {}
            for token in chunk_tokens_list:
                token_freq[token] = token_freq.get(token, 0) + 1
            freq_bonus = sum(token_freq.get(token, 0) for token in q_tokens)
            phrase_bonus = 2.5 if q_phrase and q_phrase in chunk.lower() else 0.0
            math_bonus = self._math_signal_score(chunk) * 2.1 if prefer_math else 0.0
            fuzzy_bonus = self._char_ngram_overlap(question, chunk, n=3) * 7.5
            equation_bonus = self._equation_pattern_overlap(question, chunk) * 3.2
            chunk_vector = vector_by_chunk.get(chunk, [])
            vector_similarity = self._cosine_similarity(q_vector, chunk_vector) if chunk_vector else 0.0
            vector_bonus = vector_similarity * 4.0
            score = (
                float(overlap)
                + float(freq_bonus) * 0.18
                + phrase_bonus
                + math_bonus
                + fuzzy_bonus
                + equation_bonus
                + vector_bonus
            )
            if score <= 0.35:
                continue
            scored.append(
                RetrievedChunk(
                    text=chunk,
                    score=float(score),
                    token_overlap=int(overlap),
                    vector_similarity=float(vector_similarity),
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _char_ngram_overlap(self, a: str, b: str, *, n: int) -> float:
        a_norm = self._normalize_whitespace(a.lower())
        b_norm = self._normalize_whitespace(b.lower())
        if len(a_norm) < n or len(b_norm) < n:
            return 0.0
        a_set = {a_norm[i:i + n] for i in range(len(a_norm) - n + 1)}
        b_set = {b_norm[i:i + n] for i in range(len(b_norm) - n + 1)}
        if not a_set or not b_set:
            return 0.0
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return inter / max(1, union)

    def _equation_pattern_overlap(self, question: str, chunk: str) -> float:
        q_patterns = set(re.findall(r"[a-z]\^\d+|\d+[a-z]|[a-z]\d+|\d+(?:\.\d+)?|[+\-*/=]", question.lower()))
        if not q_patterns:
            return 0.0
        c_patterns = set(re.findall(r"[a-z]\^\d+|\d+[a-z]|[a-z]\d+|\d+(?:\.\d+)?|[+\-*/=]", chunk.lower()))
        if not c_patterns:
            return 0.0
        return len(q_patterns & c_patterns) / max(1, len(q_patterns))

    def _build_rag_system_prompt(self, *, allow_internet_notes: bool) -> str:
        extra = (
            "Internet notes are supplemental only and must never override uploaded book context."
            if allow_internet_notes and not self.config.strict_book_only_mode
            else "Never use external knowledge. If context is missing, refuse and ask for relevant uploaded chapter."
        )
        language_instruction = (
            "Respond in natural Hinglish (Hindi + English mix) using Roman script."
            if self.config.hinglish_enabled
            else "Respond in clear English."
        )
        return (
            "You are SP Sir, a matric-level teacher using strict retrieval-augmented generation (RAG). "
            "Hard rules: "
            "1) Answer only from retrieved syllabus context provided by the user. "
            "2) If context is insufficient, clearly state which chapter/topic from uploaded books is missing. "
            "3) Do not invent facts, formulas, chapters, or references. "
            "4) Keep wording clear for school students. "
            f"5) {language_instruction} "
            f"{extra}"
        )

    def _retrieval_is_confident(self, question: str, retrieved: list[RetrievedChunk]) -> bool:
        if not retrieved:
            return False

        top = retrieved[0]
        if top.score < 0.9:
            return False

        top_k = retrieved[:3]
        total_overlap = sum(item.token_overlap for item in top_k)
        avg_vector = sum(item.vector_similarity for item in top_k) / max(1, len(top_k))

        meaningful = [
            t for t in self._tokenize(question)
            if len(t) >= 3 and t not in self._token_stop_words
        ]
        meaningful_count = len(set(meaningful))

        if meaningful_count >= 3 and total_overlap == 0 and avg_vector < 0.20:
            return False
        if meaningful_count >= 5 and total_overlap < 2 and top.score < 1.8 and avg_vector < 0.26:
            return False
        if top.token_overlap == 0 and top.score < 1.2 and avg_vector < 0.22:
            return False
        return True

    def _build_rag_user_prompt(
        self,
        *,
        question: str,
        context: str,
        is_math_question: bool,
        internet_context: str,
        original_question: str = "",
    ) -> str:
        lines = [
            f"Student question (normalized English for retrieval): {question}",
        ]
        if original_question and self._normalize_whitespace(original_question.lower()) != self._normalize_whitespace(question.lower()):
            lines.append(f"Original student question: {original_question}")
        lines.extend(
            [
                "",
                "Answer style: Hinglish (Hindi + English mix in Roman script).",
            "",
            "Retrieved syllabus context:",
            context,
            ]
        )
        if internet_context:
            lines.extend(["", "Supplemental internet notes:", internet_context])
        lines.extend(
            [
                "",
                "If context does not support the answer, clearly mention the missing chapter/topic from uploaded books.",
            ]
        )
        if is_math_question:
            lines.extend(
                [
                    "If answerable, use this structure:",
                    "1) Concept",
                    "2) Formula",
                    "3) Step-by-step solution",
                    "4) Final answer",
                ]
            )
        else:
            lines.append("If answerable, reply in clear student-friendly language in 3-6 sentences.")
        return "\n".join(lines)

    def _summarize_from_context(self, *, question: str, context: str) -> str:
        # Deterministic short-form answer based on top matching sentences.
        sentences = re.split(r"(?<=[.!?])\s+", context)
        q_tokens = set(self._tokenize(question))
        ranked: list[tuple[int, str]] = []
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            overlap = len(q_tokens & set(self._tokenize(s)))
            ranked.append((overlap, s))
        ranked.sort(key=lambda x: x[0], reverse=True)
        best = [s for _, s in ranked[:3] if s]
        if not best:
            return context[:420]
        return " ".join(best)
