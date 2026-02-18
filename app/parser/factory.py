"""Parser factory — selects the correct parser based on input format."""

from app.ir.models import IRGraph
from app.parser.drawio_parser import DrawioParser
from app.parser.exceptions import ParserError
from app.parser.mermaid_parser import MermaidParser


def detect_format(content: str) -> str:
    """Auto-detect diagram format from content.

    Args:
        content: Raw diagram string.

    Returns:
        'mermaid' or 'drawio'.

    Raises:
        ParserError: If format cannot be determined.
    """
    stripped = content.strip()
    if stripped.startswith("<?xml") or stripped.startswith("<mxfile") or stripped.startswith("<mxGraphModel"):
        return "drawio"
    if stripped.startswith("graph "):
        return "mermaid"
    raise ParserError(
        "Cannot detect diagram format. Expected Mermaid ('graph TD ...') "
        "or Draw.io XML ('<?xml ...', '<mxfile ...', '<mxGraphModel ...')."
    )


def parse_diagram(content: str, fmt: str | None = None) -> IRGraph:
    """Parse a diagram string into an IRGraph.

    Args:
        content: The raw diagram text or XML.
        fmt: Explicit format ('mermaid' or 'drawio'). Auto-detected if None.

    Returns:
        The parsed IRGraph.

    Raises:
        ParserError: On any parsing failure.
    """
    if fmt is None:
        fmt = detect_format(content)

    if fmt == "mermaid":
        return MermaidParser().parse(content)
    if fmt == "drawio":
        return DrawioParser().parse(content)
    raise ParserError(f"Unknown diagram format: '{fmt}'")
