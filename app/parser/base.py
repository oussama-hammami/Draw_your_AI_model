"""Abstract base class for diagram parsers."""

from abc import ABC, abstractmethod

from app.ir.models import IRGraph


class BaseParser(ABC):
    """Interface that every diagram parser must implement."""

    @abstractmethod
    def parse(self, content: str) -> IRGraph:
        """Parse raw diagram content into an IR graph.

        Args:
            content: The raw diagram text (Mermaid) or XML string (Draw.io).

        Returns:
            A validated IRGraph instance.

        Raises:
            ParserError: If the diagram cannot be parsed.
        """

    @staticmethod
    def _cast_param_value(raw: str) -> int | float | str:
        """Try to cast a parameter value string to int, then float, else keep as str."""
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw
