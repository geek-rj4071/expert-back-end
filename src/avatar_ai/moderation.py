"""Safety checks for incoming and outgoing chat messages."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModerationResult:
    ok: bool
    reason: str | None = None


class ModerationPolicy:
    """Simple deterministic moderation policy for MVP environments."""

    _BANNED_TERMS = {
        "build a bomb": "violence_instruction",
        "kill yourself": "self_harm_abuse",
        "credit card number": "sensitive_financial_data",
    }

    _EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    _PHONE_PATTERN = re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b")

    def check(self, text: str) -> ModerationResult:
        normalized = text.strip().lower()
        if not normalized:
            return ModerationResult(ok=False, reason="empty_message")

        for term, reason in self._BANNED_TERMS.items():
            if term in normalized:
                return ModerationResult(ok=False, reason=reason)

        if self._EMAIL_PATTERN.search(text) and self._PHONE_PATTERN.search(text):
            return ModerationResult(ok=False, reason="pii_combination")

        return ModerationResult(ok=True)
