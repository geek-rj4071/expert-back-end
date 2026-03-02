import unittest
from datetime import datetime, timezone

from avatar_ai.rate_limit import FixedWindowRateLimiter, RateLimitConfig


class FixedWindowRateLimiterTests(unittest.TestCase):
    def test_enforces_limit_and_expires_window(self) -> None:
        limiter = FixedWindowRateLimiter(RateLimitConfig(max_requests=2, window_seconds=10))
        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        self.assertTrue(limiter.allow("u1", now=t0))
        self.assertTrue(limiter.allow("u1", now=t0))
        self.assertFalse(limiter.allow("u1", now=t0))

        t1 = datetime(2026, 1, 1, 0, 0, 11, tzinfo=timezone.utc)
        self.assertTrue(limiter.allow("u1", now=t1))


if __name__ == "__main__":
    unittest.main()
