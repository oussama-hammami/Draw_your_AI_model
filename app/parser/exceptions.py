"""Custom exceptions for diagram parsing."""


class ParserError(Exception):
    """Base exception for all parser errors."""


class DiagramSyntaxError(ParserError):
    """Raised when the diagram has invalid syntax."""

    def __init__(self, message: str, line: int | None = None) -> None:
        self.line = line
        prefix = f"Line {line}: " if line is not None else ""
        super().__init__(f"{prefix}{message}")


class UnsupportedLayerError(ParserError):
    """Raised when a diagram references an unknown layer type."""

    def __init__(self, layer_type: str) -> None:
        self.layer_type = layer_type
        super().__init__(f"Unsupported layer type: '{layer_type}'")


class InvalidParameterError(ParserError):
    """Raised when a layer parameter cannot be parsed."""

    def __init__(self, node_id: str, detail: str) -> None:
        self.node_id = node_id
        super().__init__(f"Invalid parameter in node '{node_id}': {detail}")


class EmptyDiagramError(ParserError):
    """Raised when the diagram contains no parsable content."""
