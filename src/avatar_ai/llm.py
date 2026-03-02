"""Deterministic response generator used as an injectable AI provider."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssistantOutput:
    text: str
    emotion: str


class AvatarResponder:
    """Rule-based responder that can be swapped with a real LLM provider."""

    def generate(self, persona: str, user_text: str) -> AssistantOutput:
        lowered = user_text.lower()

        if "sad" in lowered or "stressed" in lowered:
            return AssistantOutput(
                text=f"{persona}: I hear you. Let's take one small next step together.",
                emotion="empathetic",
            )

        if lowered.endswith("?"):
            return AssistantOutput(
                text=f"{persona}: Great question. Here's a practical way to think about it.",
                emotion="curious",
            )

        return AssistantOutput(
            text=f"{persona}: Thanks for sharing. Want to go one level deeper?",
            emotion="neutral",
        )
