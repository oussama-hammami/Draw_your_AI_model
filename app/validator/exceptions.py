"""Custom exceptions for IR validation."""


class ValidationError(Exception):
    """Base exception for all validation errors."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = messages
        summary = "; ".join(messages)
        super().__init__(f"Validation failed ({len(messages)} issue(s)): {summary}")


class CycleDetectedError(ValidationError):
    """Raised when the IR graph contains a cycle."""


class UnreachableNodeError(ValidationError):
    """Raised when nodes exist that cannot be reached from any root."""


class InvalidEdgeError(ValidationError):
    """Raised when an edge references a non-existent node."""


class DuplicateNodeError(ValidationError):
    """Raised when two nodes share the same id."""


class InvalidParameterError(ValidationError):
    """Raised when a layer has invalid or missing parameters."""
