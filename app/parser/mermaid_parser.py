"""Parser for Mermaid flowchart diagrams.

Supported Mermaid syntax (graph TD / graph LR):

    graph TD
        A[Input: shape=784] --> B[Linear: 256]
        B --> C[ReLU]
        C --> D[Linear: 10]
        D --> E[Softmax]

Node label format:
    NodeId[Type]                       — no parameters
    NodeId[Type: primary_value]        — shorthand for primary parameter
    NodeId[Type: p1=v1, p2=v2]        — explicit key=value parameters
"""

import logging
import re
from typing import Any

from app.ir.models import SUPPORTED_LAYER_TYPES, PRIMARY_PARAM, IREdge, IRGraph, IRNode
from app.parser.base import BaseParser
from app.parser.exceptions import (
    DiagramSyntaxError,
    EmptyDiagramError,
    InvalidParameterError,
    UnsupportedLayerError,
)

logger = logging.getLogger(__name__)

# Matches: A[Label text] --> B[Label text]  or  A --> B  (node may lack label)
_EDGE_RE = re.compile(
    r"(?P<src>\w+)"                # source node id
    r"(?:\[(?P<src_label>[^\]]*)\])?"  # optional [label]
    r"\s*-->\s*"                    # arrow
    r"(?P<dst>\w+)"                # target node id
    r"(?:\[(?P<dst_label>[^\]]*)\])?"  # optional [label]
)

# Matches standalone node definitions: A[Label text]
_NODE_DEF_RE = re.compile(
    r"^\s+(?P<id>\w+)\[(?P<label>[^\]]+)\]\s*$"
)

# Matches the label inside brackets: "Type: param_string" or just "Type"
_LABEL_RE = re.compile(
    r"^(?P<type>\w+)(?:\s*:\s*(?P<params>.+))?$"
)


class MermaidParser(BaseParser):
    """Parse Mermaid flowchart syntax into an IRGraph."""

    def parse(self, content: str) -> IRGraph:
        """Parse a Mermaid diagram string into an IRGraph.

        Args:
            content: Raw Mermaid flowchart text.

        Returns:
            A populated IRGraph.

        Raises:
            EmptyDiagramError: If no nodes or edges are found.
            DiagramSyntaxError: If the diagram header is missing.
            UnsupportedLayerError: If a node references an unknown layer.
            InvalidParameterError: If parameters cannot be parsed.
        """
        content = content.strip()
        if not content:
            raise EmptyDiagramError("Diagram content is empty")

        lines = content.splitlines()
        self._validate_header(lines)

        nodes: dict[str, IRNode] = {}
        edges: list[IREdge] = []

        for line_no, line in enumerate(lines, start=1):
            # Skip header line and comments
            stripped = line.strip()
            if stripped.startswith("graph ") or stripped.startswith("%%") or not stripped:
                continue

            edge_match = _EDGE_RE.search(line)
            if edge_match:
                self._process_edge_line(edge_match, nodes, edges, line_no)
                continue

            node_match = _NODE_DEF_RE.match(line)
            if node_match:
                node_id = node_match.group("id")
                label = node_match.group("label")
                if node_id not in nodes:
                    nodes[node_id] = self._parse_label(node_id, label, line_no)

        if not nodes:
            raise EmptyDiagramError("No nodes found in Mermaid diagram")

        graph = IRGraph(
            nodes=list(nodes.values()),
            edges=edges,
        )
        logger.info("Parsed Mermaid diagram: %d nodes, %d edges", len(nodes), len(edges))
        return graph

    def _validate_header(self, lines: list[str]) -> None:
        """Check that the first non-empty line is a valid graph directive."""
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("%%"):
                continue
            if re.match(r"^graph\s+(TD|TB|BT|LR|RL)$", stripped):
                return
            raise DiagramSyntaxError(
                f"Expected 'graph TD|LR|...' header, got: '{stripped}'", line=1
            )
        raise EmptyDiagramError("Diagram content is empty")

    def _process_edge_line(
        self,
        match: re.Match[str],
        nodes: dict[str, IRNode],
        edges: list[IREdge],
        line_no: int,
    ) -> None:
        """Extract nodes and edge from a matched edge line."""
        src_id = match.group("src")
        dst_id = match.group("dst")
        src_label = match.group("src_label")
        dst_label = match.group("dst_label")

        if src_id not in nodes and src_label:
            nodes[src_id] = self._parse_label(src_id, src_label, line_no)
        if dst_id not in nodes and dst_label:
            nodes[dst_id] = self._parse_label(dst_id, dst_label, line_no)

        edges.append(IREdge(**{"from": src_id, "to": dst_id}))

    def _parse_label(self, node_id: str, label: str, line_no: int) -> IRNode:
        """Parse a bracket label like 'Linear: 256' into an IRNode.

        Args:
            node_id: The node identifier (e.g. 'A').
            label: The text inside brackets (e.g. 'Linear: out_features=256').
            line_no: Line number for error reporting.

        Returns:
            An IRNode with parsed type and parameters.
        """
        label = label.strip()
        label_match = _LABEL_RE.match(label)
        if not label_match:
            raise DiagramSyntaxError(f"Cannot parse node label: '{label}'", line=line_no)

        layer_type = label_match.group("type")
        if layer_type not in SUPPORTED_LAYER_TYPES:
            raise UnsupportedLayerError(layer_type)

        raw_params = label_match.group("params")
        params = self._parse_params(node_id, layer_type, raw_params) if raw_params else {}

        return IRNode(id=node_id, type=layer_type, params=params)

    def _parse_params(
        self, node_id: str, layer_type: str, raw: str
    ) -> dict[str, Any]:
        """Parse parameter string into a dict.

        Handles both:
            - Shorthand: '256' → {primary_param: 256}
            - Explicit: 'out_channels=32, kernel_size=3'
        """
        raw = raw.strip()
        params: dict[str, Any] = {}

        # Check if it's shorthand (a single value with no '=' and no commas)
        if "=" not in raw:
            if "," in raw:
                raise InvalidParameterError(
                    node_id,
                    f"Multiple values without key=value syntax: '{raw}'",
                )
            primary = PRIMARY_PARAM.get(layer_type)
            if primary is None:
                raise InvalidParameterError(
                    node_id,
                    f"Layer '{layer_type}' has no primary parameter; use key=value syntax",
                )
            params[primary] = self._cast_param_value(raw)
            return params

        # Explicit key=value pairs
        for pair in raw.split(","):
            pair = pair.strip()
            if "=" not in pair:
                raise InvalidParameterError(
                    node_id, f"Expected key=value, got: '{pair}'"
                )
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise InvalidParameterError(node_id, "Empty parameter name")
            params[key] = self._cast_param_value(value)

        return params
