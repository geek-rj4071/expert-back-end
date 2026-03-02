import unittest

from avatar_ai.moderation import ModerationPolicy


class ModerationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ModerationPolicy()

    def test_empty_message_is_blocked(self) -> None:
        result = self.policy.check("  ")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "empty_message")

    def test_banned_term_is_blocked(self) -> None:
        result = self.policy.check("Can you share my credit card number?")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "sensitive_financial_data")

    def test_combined_pii_is_blocked(self) -> None:
        result = self.policy.check("reach me at jane@example.com or 415-555-1234")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "pii_combination")

    def test_regular_message_passes(self) -> None:
        result = self.policy.check("I want to learn Spanish")
        self.assertTrue(result.ok)
        self.assertIsNone(result.reason)


if __name__ == "__main__":
    unittest.main()
