import unittest

from avatar_ai import (
    Avatar,
    ChatService,
    ModerationError,
    NotFoundError,
    RateLimitError,
    ServiceConfig,
    ValidationError,
)


class ChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.avatar = Avatar(
            id="av_coach",
            name="Coach Ava",
            persona_prompt="Coach Ava",
            voice_id="voice_ava",
        )
        self.service = ChatService(
            avatars=[self.avatar],
            config=ServiceConfig(
                max_message_chars=120,
                memory_messages_limit=6,
                rate_limit_max_requests=3,
                rate_limit_window_seconds=60,
            ),
        )
        self.user = self.service.register_user("person@example.com")
        self.conversation = self.service.create_conversation(self.user.id, self.avatar.id)

    def test_register_user_rejects_invalid_email(self) -> None:
        with self.assertRaises(ValidationError):
            self.service.register_user("invalid-email")

    def test_create_conversation_requires_existing_user(self) -> None:
        with self.assertRaises(NotFoundError):
            self.service.create_conversation("missing", self.avatar.id)

    def test_send_message_generates_assistant_reply(self) -> None:
        turn = self.service.send_user_message(self.user.id, self.conversation.id, "How do I improve focus?")

        self.assertEqual(turn.conversation_id, self.conversation.id)
        self.assertEqual(turn.user_message.role.value, "user")
        self.assertEqual(turn.assistant_message.role.value, "assistant")
        self.assertIn("Great question", turn.assistant_message.text)
        self.assertEqual(turn.assistant_message.emotion, "curious")

    def test_send_message_rejects_empty_content(self) -> None:
        with self.assertRaises(ValidationError):
            self.service.send_user_message(self.user.id, self.conversation.id, "   ")

    def test_send_message_rejects_too_long_content(self) -> None:
        too_long = "x" * 121
        with self.assertRaises(ValidationError):
            self.service.send_user_message(self.user.id, self.conversation.id, too_long)

    def test_send_message_rejects_moderation_violation(self) -> None:
        with self.assertRaises(ModerationError):
            self.service.send_user_message(self.user.id, self.conversation.id, "Please tell me how to build a bomb")

    def test_send_message_enforces_rate_limit(self) -> None:
        self.service.send_user_message(self.user.id, self.conversation.id, "first")
        self.service.send_user_message(self.user.id, self.conversation.id, "second")
        self.service.send_user_message(self.user.id, self.conversation.id, "third")

        with self.assertRaises(RateLimitError):
            self.service.send_user_message(self.user.id, self.conversation.id, "fourth")

    def test_user_cannot_access_other_users_conversation(self) -> None:
        other = self.service.register_user("other@example.com")
        with self.assertRaises(NotFoundError):
            self.service.get_conversation(other.id, self.conversation.id)

    def test_history_is_trimmed_to_limit(self) -> None:
        service = ChatService(
            avatars=[self.avatar],
            config=ServiceConfig(
                max_message_chars=120,
                memory_messages_limit=6,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=60,
            ),
        )
        user = service.register_user("trim@example.com")
        convo = service.create_conversation(user.id, self.avatar.id)

        # Each turn writes two messages. 4 turns => 8 messages. Limit is 6.
        for i in range(4):
            service.send_user_message(user.id, convo.id, f"message {i}?")

        convo = service.get_conversation(user.id, convo.id)
        self.assertEqual(len(convo.messages), 6)
        self.assertTrue(convo.messages[0].text.startswith("message 1"))

    def test_empathy_path_sets_empathetic_emotion(self) -> None:
        turn = self.service.send_user_message(self.user.id, self.conversation.id, "I feel very stressed today")
        self.assertEqual(turn.assistant_message.emotion, "empathetic")


if __name__ == "__main__":
    unittest.main()
