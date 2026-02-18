"""Parser for Draw.io (diagrams.net) XML files.

Draw.io exports XML where each node/edge is an <mxCell> element.
Nodes have a 'value' attribute with the layer label and 'vertex="1"'.
Edges have 'source' and 'target' attributes and 'edge="1"'.

Expected label format inside the 'value' attribute matches the same
convention as Mermaid labels:
    "Linear: 256"
    "Conv2D: out_channels=32, kernel_size=3"
    "ReLU"
"""

import logging
import re
import xml.etree.ElementTree as ET
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

_LABEL_RE = re.compile(r"^(?P<type>\w+)(?:\s*:\s*(?P<params>.+))?$")


class DrawioParser(BaseParser):
    """Parse Draw.io XML into an IRGraph."""

    def parse(self, content: str) -> IRGraph:
        """Parse a Draw.io XML string into an IRGraph.

        Args:
            content: Raw Draw.io XML string.

        Returns:
            A populated IRGraph.

        Raises:
            EmptyDiagramError: If no cells are found.
            DiagramSyntaxError: If the XML is malformed.
            UnsupportedLayerError: If a node references an unknown layer.
        """
        content = content.strip()
        if not content:
            raise EmptyDiagramError("Diagram content is empty")

        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise DiagramSyntaxError(f"Invalid XML: {exc}") from exc

        cells = root.findall(".//mxCell")
        if not cells:
            raise EmptyDiagramError("No mxCell elements found in Draw.io XML")

        nodes: dict[str, IRNode] = {}
        edges: list[IREdge] = []

        for cell in cells:
            cell_id = cell.get("id", "")

            # Skip the root and default parent cells
            if cell_id in ("0", "1"):
                continue

            if cell.get("edge") == "1":
                source = cell.get("source")
                target = cell.get("target")
                if source and target:
                    edges.append(IREdge(**{"from": source, "to": target}))
                else:
                    logger.warning("Edge cell '%s' missing source or target, skipped", cell_id)
                continue

            if cell.get("vertex") == "1":
                value = cell.get("value", "").strip()
                if not value:
                    logger.warning("Vertex cell '%s' has no value, skipped", cell_id)
                    continue
                nodes[cell_id] = self._parse_label(cell_id, value)

        if not nodes:
            raise EmptyDiagramError("No layer nodes found in Draw.io XML")

        graph = IRGraph(nodes=list(nodes.values()), edges=edges)
        logger.info("Parsed Draw.io XML: %d nodes, %d edges", len(nodes), len(edges))
        return graph

    def _parse_label(self, node_id: str, label: str) -> IRNode:
        """Parse a cell value label into an IRNode."""
        label_match = _LABEL_RE.match(label)
        if not label_match:
            raise DiagramSyntaxError(f"Cannot parse node label: '{label}'")

        layer_type = label_match.group("type")
        if layer_type not in SUPPORTED_LAYER_TYPES:
            raise UnsupportedLayerError(layer_type)

        raw_params = label_match.group("params")
        params = self._parse_params(node_id, layer_type, raw_params) if raw_params else {}
        return IRNode(id=node_id, type=layer_type, params=params)

    def _parse_params(
        self, node_id: str, layer_type: str, raw: str
    ) -> dict[str, Any]:
        """Parse parameter string — same logic as MermaidParser."""
        raw = raw.strip()
        params: dict[str, Any] = {}

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

        for pair in raw.split(","):
            pair = pair.strip()
            if "=" not in pair:
                raise InvalidParameterError(node_id, f"Expected key=value, got: '{pair}'")
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise InvalidParameterError(node_id, "Empty parameter name")
            params[key] = self._cast_param_value(value)

        return params
