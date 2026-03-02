"""Main application service for user and conversation orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from .errors import ModerationError, NotFoundError, RateLimitError, ValidationError
from .llm import AvatarResponder
from .models import Avatar, Conversation, Message, Role, TurnResponse, User, new_id, utc_now
from .moderation import ModerationPolicy
from .rate_limit import FixedWindowRateLimiter, RateLimitConfig


@dataclass(frozen=True)
class ServiceConfig:
    max_message_chars: int = 1000
    memory_messages_limit: int = 40
    rate_limit_max_requests: int = 20
    rate_limit_window_seconds: int = 60


class ChatService:
    """Coordinates moderation, rate limits, and AI response generation."""

    def __init__(
        self,
        avatars: Iterable[Avatar],
        responder: AvatarResponder | None = None,
        moderation: ModerationPolicy | None = None,
        config: ServiceConfig | None = None,
    ) -> None:
        self.config = config or ServiceConfig()
        self.avatars: Dict[str, Avatar] = {a.id: a for a in avatars}
        self.users: Dict[str, User] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.responder = responder or AvatarResponder()
        self.moderation = moderation or ModerationPolicy()
        self.rate_limiter = FixedWindowRateLimiter(
            RateLimitConfig(
                max_requests=self.config.rate_limit_max_requests,
                window_seconds=self.config.rate_limit_window_seconds,
            )
        )

    def register_user(self, email: str) -> User:
        normalized = email.strip().lower()
        if "@" not in normalized:
            raise ValidationError("invalid_email")

        user = User(id=new_id("usr"), email=normalized, created_at=utc_now())
        self.users[user.id] = user
        return user

    def create_conversation(self, user_id: str, avatar_id: str) -> Conversation:
        user = self.users.get(user_id)
        if user is None:
            raise NotFoundError("user_not_found")

        avatar = self.avatars.get(avatar_id)
        if avatar is None:
            raise NotFoundError("avatar_not_found")

        conversation = Conversation(
            id=new_id("cnv"),
            user_id=user.id,
            avatar=avatar,
            created_at=utc_now(),
        )
        self.conversations[conversation.id] = conversation
        return conversation

    def send_user_message(self, user_id: str, conversation_id: str, text: str) -> TurnResponse:
        self._ensure_user(user_id)
        conversation = self._ensure_conversation_belongs_to_user(conversation_id, user_id)
        clean_text = self._validate_text(text)

        if not self.rate_limiter.allow(user_id):
            raise RateLimitError("too_many_requests")

        check = self.moderation.check(clean_text)
        if not check.ok:
            raise ModerationError(check.reason or "moderation_blocked")

        user_message = Message(
            id=new_id("msg"),
            role=Role.USER,
            text=clean_text,
            created_at=utc_now(),
        )
        conversation.messages.append(user_message)

        ai_out = self.responder.generate(conversation.avatar.persona_prompt, clean_text)
        output_check = self.moderation.check(ai_out.text)
        if not output_check.ok:
            raise ModerationError(output_check.reason or "unsafe_assistant_output")

        assistant_message = Message(
            id=new_id("msg"),
            role=Role.ASSISTANT,
            text=ai_out.text,
            emotion=ai_out.emotion,
            created_at=utc_now(),
        )
        conversation.messages.append(assistant_message)
        self._trim_history(conversation)

        return TurnResponse(
            conversation_id=conversation.id,
            user_message=user_message,
            assistant_message=assistant_message,
        )

    def get_conversation(self, user_id: str, conversation_id: str) -> Conversation:
        self._ensure_user(user_id)
        return self._ensure_conversation_belongs_to_user(conversation_id, user_id)

    def _ensure_user(self, user_id: str) -> User:
        user = self.users.get(user_id)
        if user is None:
            raise NotFoundError("user_not_found")
        return user

    def _ensure_conversation_belongs_to_user(self, conversation_id: str, user_id: str) -> Conversation:
        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            raise NotFoundError("conversation_not_found")
        if conversation.user_id != user_id:
            raise NotFoundError("conversation_not_found")
        return conversation

    def _validate_text(self, text: str) -> str:
        clean = text.strip()
        if not clean:
            raise ValidationError("message_empty")
        if len(clean) > self.config.max_message_chars:
            raise ValidationError("message_too_long")
        return clean

    def _trim_history(self, conversation: Conversation) -> None:
        limit = self.config.memory_messages_limit
        if len(conversation.messages) > limit:
            conversation.messages = conversation.messages[-limit:]
