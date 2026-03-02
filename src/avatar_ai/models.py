"""Core data models for avatar chat domain."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List
from uuid import uuid4


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class AccountRole(str, Enum):
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"


@dataclass(frozen=True)
class Avatar:
    id: str
    name: str
    persona_prompt: str
    voice_id: str


@dataclass(frozen=True)
class Message:
    id: str
    role: Role
    text: str
    created_at: datetime
    emotion: str | None = None


@dataclass
class Conversation:
    id: str
    user_id: str
    avatar: Avatar
    created_at: datetime
    messages: List[Message] = field(default_factory=list)


@dataclass(frozen=True)
class TurnResponse:
    conversation_id: str
    user_message: Message
    assistant_message: Message


@dataclass(frozen=True)
class User:
    id: str
    email: str
    created_at: datetime
    role: str = AccountRole.STUDENT.value
    display_name: str | None = None
    auth_provider: str = "google"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"
