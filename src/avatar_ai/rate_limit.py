"""In-memory fixed-window rate limiter for per-user turn control."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class RateLimitConfig:
    max_requests: int = 20
    window_seconds: int = 60


class FixedWindowRateLimiter:
    """Tracks event timestamps per key and enforces a max in rolling window."""

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self.config = config or RateLimitConfig()
        self._events: dict[str, deque[datetime]] = defaultdict(deque)

    def allow(self, key: str, now: datetime | None = None) -> bool:
        current = now or datetime.now(timezone.utc)
        events = self._events[key]
        window_start = current - timedelta(seconds=self.config.window_seconds)

        while events and events[0] <= window_start:
            events.popleft()

        if len(events) >= self.config.max_requests:
            return False

        events.append(current)
        return True
