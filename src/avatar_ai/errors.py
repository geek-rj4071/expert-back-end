"""Domain-specific exceptions for avatar chat service."""


class ChatServiceError(Exception):
    """Base error for all chat service failures."""


class ValidationError(ChatServiceError):
    """Raised when input validation fails."""


class NotFoundError(ChatServiceError):
    """Raised when a requested entity does not exist."""


class RateLimitError(ChatServiceError):
    """Raised when a user exceeds allowed request rates."""


class ModerationError(ChatServiceError):
    """Raised when content violates safety policy."""
