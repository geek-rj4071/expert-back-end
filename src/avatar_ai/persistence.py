"""SQLite persistence for users, avatars, conversations, and messages."""
from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List

from .models import Avatar, Conversation, Message, Role, User


@dataclass(frozen=True)
class PersistedTurn:
    user_message: Message
    assistant_message: Message


@dataclass(frozen=True)
class TrainingDocument:
    id: str
    avatar_id: str
    filename: str
    content_text: str
    source_type: str
    created_at: datetime


@dataclass(frozen=True)
class TrainingChunkVector:
    id: str
    document_id: str
    avatar_id: str
    chunk_id: str
    chunk_index: int
    chunk_text: str
    embedding: list[float]
    dimension: int
    created_at: datetime


@dataclass(frozen=True)
class ConversationImageContext:
    id: str
    conversation_id: str
    user_id: str
    filename: str
    content_text: str
    consumed: bool
    created_at: datetime


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SQLiteRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'student',
                    display_name TEXT,
                    auth_provider TEXT NOT NULL DEFAULT 'google'
                );

                CREATE TABLE IF NOT EXISTS avatars (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    persona_prompt TEXT NOT NULL,
                    voice_id TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    avatar_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id),
                    FOREIGN KEY(avatar_id) REFERENCES avatars(id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    emotion TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );

                CREATE TABLE IF NOT EXISTS training_documents (
                    id TEXT PRIMARY KEY,
                    avatar_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(avatar_id) REFERENCES avatars(id)
                );

                CREATE TABLE IF NOT EXISTS training_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    avatar_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES training_documents(id),
                    FOREIGN KEY(avatar_id) REFERENCES avatars(id)
                );

                CREATE TABLE IF NOT EXISTS training_chunk_vectors (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    avatar_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES training_documents(id),
                    FOREIGN KEY(avatar_id) REFERENCES avatars(id),
                    FOREIGN KEY(chunk_id) REFERENCES training_chunks(id)
                );

                CREATE TABLE IF NOT EXISTS conversation_image_contexts (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_training_docs_avatar ON training_documents(avatar_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_training_chunks_avatar ON training_chunks(avatar_id, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_training_vectors_avatar ON training_chunk_vectors(avatar_id, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_image_ctx_conversation ON conversation_image_contexts(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_image_ctx_user ON conversation_image_contexts(user_id, created_at);
                """
            )
            self._ensure_column(conn, "users", "role", "role TEXT NOT NULL DEFAULT 'student'")
            self._ensure_column(conn, "users", "display_name", "display_name TEXT")
            self._ensure_column(conn, "users", "auth_provider", "auth_provider TEXT NOT NULL DEFAULT 'google'")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(r["name"]) for r in rows}
        if column in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    def upsert_avatar(self, avatar: Avatar) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO avatars (id, name, persona_prompt, voice_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    persona_prompt=excluded.persona_prompt,
                    voice_id=excluded.voice_id
                """,
                (avatar.id, avatar.name, avatar.persona_prompt, avatar.voice_id),
            )

    def list_avatars(self) -> List[Avatar]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, name, persona_prompt, voice_id FROM avatars ORDER BY name").fetchall()
        return [Avatar(id=r["id"], name=r["name"], persona_prompt=r["persona_prompt"], voice_id=r["voice_id"]) for r in rows]

    def get_avatar(self, avatar_id: str) -> Avatar | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, name, persona_prompt, voice_id FROM avatars WHERE id = ?",
                (avatar_id,),
            ).fetchone()
        if row is None:
            return None
        return Avatar(id=row["id"], name=row["name"], persona_prompt=row["persona_prompt"], voice_id=row["voice_id"])

    def create_user(
        self,
        *,
        user_id: str,
        email: str,
        role: str = "student",
        display_name: str | None = None,
        auth_provider: str = "google",
    ) -> User:
        created_at = _now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO users (id, email, created_at, role, display_name, auth_provider)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, email, created_at, role, display_name, auth_provider),
            )
        return User(
            id=user_id,
            email=email,
            created_at=_parse_iso(created_at),
            role=role,
            display_name=display_name,
            auth_provider=auth_provider,
        )

    def get_user(self, user_id: str) -> User | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, email, created_at, role, display_name, auth_provider FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return User(
            id=row["id"],
            email=row["email"],
            created_at=_parse_iso(row["created_at"]),
            role=str(row["role"] or "student"),
            display_name=row["display_name"],
            auth_provider=str(row["auth_provider"] or "google"),
        )

    def get_user_by_email(self, email: str) -> User | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, email, created_at, role, display_name, auth_provider FROM users WHERE lower(email) = lower(?)",
                (email,),
            ).fetchone()
        if row is None:
            return None
        return User(
            id=row["id"],
            email=row["email"],
            created_at=_parse_iso(row["created_at"]),
            role=str(row["role"] or "student"),
            display_name=row["display_name"],
            auth_provider=str(row["auth_provider"] or "google"),
        )

    def list_users(self) -> list[User]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, email, created_at, role, display_name, auth_provider
                FROM users
                ORDER BY created_at
                """
            ).fetchall()
        return [
            User(
                id=row["id"],
                email=row["email"],
                created_at=_parse_iso(row["created_at"]),
                role=str(row["role"] or "student"),
                display_name=row["display_name"],
                auth_provider=str(row["auth_provider"] or "google"),
            )
            for row in rows
        ]

    def count_users_by_role(self, role: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS cnt FROM users WHERE role = ?",
                (role,),
            ).fetchone()
        return int((row["cnt"] if row is not None else 0) or 0)

    def delete_user(self, *, user_id: str) -> bool:
        with self._conn() as conn:
            convo_rows = conn.execute(
                "SELECT id FROM conversations WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            convo_ids = [row["id"] for row in convo_rows]
            if convo_ids:
                conn.executemany(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    [(cid,) for cid in convo_ids],
                )
                conn.executemany(
                    "DELETE FROM conversation_image_contexts WHERE conversation_id = ?",
                    [(cid,) for cid in convo_ids],
                )
                conn.executemany(
                    "DELETE FROM conversations WHERE id = ?",
                    [(cid,) for cid in convo_ids],
                )
            conn.execute(
                "DELETE FROM conversation_image_contexts WHERE user_id = ?",
                (user_id,),
            )
            deleted = conn.execute(
                "DELETE FROM users WHERE id = ?",
                (user_id,),
            ).rowcount
        return bool(deleted)

    def create_conversation(self, *, conversation_id: str, user_id: str, avatar_id: str) -> Conversation:
        created_at = _now_iso()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_id, avatar_id, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, user_id, avatar_id, created_at),
            )
        avatar = self.get_avatar(avatar_id)
        assert avatar is not None
        return Conversation(id=conversation_id, user_id=user_id, avatar=avatar, created_at=_parse_iso(created_at), messages=[])

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT c.id AS conversation_id, c.user_id, c.created_at,
                       a.id AS avatar_id, a.name, a.persona_prompt, a.voice_id
                FROM conversations c
                JOIN avatars a ON a.id = c.avatar_id
                WHERE c.id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            messages = conn.execute(
                "SELECT id, role, text, emotion, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()

        avatar = Avatar(
            id=row["avatar_id"],
            name=row["name"],
            persona_prompt=row["persona_prompt"],
            voice_id=row["voice_id"],
        )
        conversation = Conversation(
            id=row["conversation_id"],
            user_id=row["user_id"],
            avatar=avatar,
            created_at=_parse_iso(row["created_at"]),
            messages=[
                Message(
                    id=m["id"],
                    role=Role(m["role"]),
                    text=m["text"],
                    emotion=m["emotion"],
                    created_at=_parse_iso(m["created_at"]),
                )
                for m in messages
            ],
        )
        return conversation

    def add_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        role: Role,
        text: str,
        emotion: str | None,
    ) -> Message:
        created_at = _now_iso()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, text, emotion, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role.value, text, emotion, created_at),
            )
        return Message(
            id=message_id,
            role=role,
            text=text,
            emotion=emotion,
            created_at=_parse_iso(created_at),
        )

    def trim_messages(self, *, conversation_id: str, keep_last: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                DELETE FROM messages
                WHERE id IN (
                    SELECT id FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (conversation_id, keep_last),
            )

    def add_training_document(
        self,
        *,
        doc_id: str,
        avatar_id: str,
        filename: str,
        content_text: str,
        source_type: str,
    ) -> TrainingDocument:
        created_at = _now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO training_documents (id, avatar_id, filename, content_text, source_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, avatar_id, filename, content_text, source_type, created_at),
            )
        return TrainingDocument(
            id=doc_id,
            avatar_id=avatar_id,
            filename=filename,
            content_text=content_text,
            source_type=source_type,
            created_at=_parse_iso(created_at),
        )

    def replace_training_chunks(self, *, document_id: str, avatar_id: str, chunks: list[tuple[str, str, int]]) -> None:
        created_at = _now_iso()
        with self._conn() as conn:
            conn.execute("DELETE FROM training_chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO training_chunks (id, document_id, avatar_id, chunk_text, chunk_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(chunk_id, document_id, avatar_id, chunk_text, chunk_idx, created_at) for chunk_id, chunk_text, chunk_idx in chunks],
            )

    def list_training_documents(self, *, avatar_id: str) -> list[TrainingDocument]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, avatar_id, filename, content_text, source_type, created_at
                FROM training_documents
                WHERE avatar_id = ?
                ORDER BY created_at DESC
                """,
                (avatar_id,),
            ).fetchall()
        return [
            TrainingDocument(
                id=row["id"],
                avatar_id=row["avatar_id"],
                filename=row["filename"],
                content_text=row["content_text"],
                source_type=row["source_type"],
                created_at=_parse_iso(row["created_at"]),
            )
            for row in rows
        ]

    def list_training_chunks(self, *, avatar_id: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT chunk_text
                FROM training_chunks
                WHERE avatar_id = ?
                ORDER BY created_at, chunk_index
                """,
                (avatar_id,),
            ).fetchall()
        return [row["chunk_text"] for row in rows]

    def replace_training_chunk_vectors(
        self,
        *,
        document_id: str,
        avatar_id: str,
        vectors: list[tuple[str, str, int, str, list[float], int]],
    ) -> None:
        created_at = _now_iso()
        with self._conn() as conn:
            conn.execute("DELETE FROM training_chunk_vectors WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO training_chunk_vectors (
                    id, document_id, avatar_id, chunk_id, chunk_index, chunk_text, embedding_json, dimension, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        vector_id,
                        document_id,
                        avatar_id,
                        chunk_id,
                        chunk_index,
                        chunk_text,
                        json.dumps(embedding),
                        dimension,
                        created_at,
                    )
                    for vector_id, chunk_id, chunk_index, chunk_text, embedding, dimension in vectors
                ],
            )

    def list_training_chunk_vectors(self, *, avatar_id: str) -> list[TrainingChunkVector]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, document_id, avatar_id, chunk_id, chunk_index, chunk_text, embedding_json, dimension, created_at
                FROM training_chunk_vectors
                WHERE avatar_id = ?
                ORDER BY created_at, chunk_index
                """,
                (avatar_id,),
            ).fetchall()
        out: list[TrainingChunkVector] = []
        for row in rows:
            try:
                embedding_raw = json.loads(row["embedding_json"])
            except Exception:
                embedding_raw = []
            embedding = [float(v) for v in embedding_raw if isinstance(v, (int, float))]
            out.append(
                TrainingChunkVector(
                    id=row["id"],
                    document_id=row["document_id"],
                    avatar_id=row["avatar_id"],
                    chunk_id=row["chunk_id"],
                    chunk_index=int(row["chunk_index"]),
                    chunk_text=row["chunk_text"],
                    embedding=embedding,
                    dimension=int(row["dimension"]),
                    created_at=_parse_iso(row["created_at"]),
                )
            )
        return out

    def count_training_chunk_vectors(self, *, avatar_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS cnt FROM training_chunk_vectors WHERE avatar_id = ?",
                (avatar_id,),
            ).fetchone()
        return int((row["cnt"] if row is not None else 0) or 0)

    def clear_training_documents(self, *, avatar_id: str) -> int:
        with self._conn() as conn:
            doc_rows = conn.execute(
                "SELECT id FROM training_documents WHERE avatar_id = ?",
                (avatar_id,),
            ).fetchall()
            doc_ids = [row["id"] for row in doc_rows]
            if doc_ids:
                conn.executemany("DELETE FROM training_chunks WHERE document_id = ?", [(doc_id,) for doc_id in doc_ids])
                conn.executemany("DELETE FROM training_chunk_vectors WHERE document_id = ?", [(doc_id,) for doc_id in doc_ids])
            deleted = conn.execute("DELETE FROM training_documents WHERE avatar_id = ?", (avatar_id,)).rowcount
        return int(deleted or 0)

    def add_conversation_image_context(
        self,
        *,
        context_id: str,
        conversation_id: str,
        user_id: str,
        filename: str,
        content_text: str,
    ) -> ConversationImageContext:
        created_at = _now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO conversation_image_contexts (
                    id, conversation_id, user_id, filename, content_text, consumed, created_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (context_id, conversation_id, user_id, filename, content_text, created_at),
            )
        return ConversationImageContext(
            id=context_id,
            conversation_id=conversation_id,
            user_id=user_id,
            filename=filename,
            content_text=content_text,
            consumed=False,
            created_at=_parse_iso(created_at),
        )

    def pop_latest_unconsumed_image_context(
        self,
        *,
        conversation_id: str,
        user_id: str,
    ) -> ConversationImageContext | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, conversation_id, user_id, filename, content_text, consumed, created_at
                FROM conversation_image_contexts
                WHERE conversation_id = ? AND user_id = ? AND consumed = 0
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE conversation_image_contexts SET consumed = 1 WHERE id = ?",
                (row["id"],),
            )
        return ConversationImageContext(
            id=row["id"],
            conversation_id=row["conversation_id"],
            user_id=row["user_id"],
            filename=row["filename"],
            content_text=row["content_text"],
            consumed=True,
            created_at=_parse_iso(row["created_at"]),
        )
