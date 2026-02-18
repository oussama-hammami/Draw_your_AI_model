"""Tests for diagram parsers (Mermaid + Draw.io)."""

import pytest

from app.ir.models import IRGraph
from app.parser import parse_diagram, detect_format
from app.parser.exceptions import (
    DiagramSyntaxError,
    EmptyDiagramError,
    UnsupportedLayerError,
    InvalidParameterError,
    ParserError,
)

# ─── Sample diagrams ─────────────────────────────────────────────

SIMPLE_MERMAID = """\
graph TD
    A[Input: shape=784] --> B[Linear: 256]
    B --> C[ReLU]
    C --> D[Linear: 10]
    D --> E[Softmax]
"""

SHORTHAND_MERMAID = """\
graph TD
    A[Input: 784] --> B[Linear: 256]
    B --> C[ReLU]
    C --> D[Dropout: 0.5]
    D --> E[Linear: 10]
"""

CONV_MERMAID = """\
graph TD
    A[Input: shape=1] --> B[Conv2D: out_channels=32, kernel_size=3]
    B --> C[BatchNorm: num_features=32]
    C --> D[ReLU]
    D --> E[MaxPool2D: 2]
    E --> F[Flatten]
    F --> G[Linear: 10]
"""

SIMPLE_DRAWIO = """\
<mxGraphModel>
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="n1" value="Input: shape=784" vertex="1" parent="1">
      <mxGeometry x="100" y="100" width="120" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="n2" value="Linear: 256" vertex="1" parent="1">
      <mxGeometry x="100" y="200" width="120" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="n3" value="ReLU" vertex="1" parent="1">
      <mxGeometry x="100" y="300" width="120" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="e1" edge="1" source="n1" target="n2" parent="1"/>
    <mxCell id="e2" edge="1" source="n2" target="n3" parent="1"/>
  </root>
</mxGraphModel>
"""


# ─── Format detection ─────────────────────────────────────────────

class TestFormatDetection:
    """Tests for auto-detection of diagram format."""

    def test_detect_mermaid(self) -> None:
        assert detect_format("graph TD\n  A --> B") == "mermaid"

    def test_detect_drawio_mxgraphmodel(self) -> None:
        assert detect_format("<mxGraphModel></mxGraphModel>") == "drawio"

    def test_detect_drawio_xml_header(self) -> None:
        assert detect_format('<?xml version="1.0"?>') == "drawio"

    def test_detect_drawio_mxfile(self) -> None:
        assert detect_format("<mxfile></mxfile>") == "drawio"

    def test_detect_unknown_raises(self) -> None:
        with pytest.raises(ParserError, match="Cannot detect diagram format"):
            detect_format("random text here")


# ─── Mermaid parser ──────────────────────────────────────────────

class TestMermaidParser:
    """Tests for the Mermaid diagram parser."""

    def test_simple_diagram_nodes(self) -> None:
        """Parsed graph must have 5 nodes with correct types."""
        graph = parse_diagram(SIMPLE_MERMAID, fmt="mermaid")
        assert isinstance(graph, IRGraph)
        assert len(graph.nodes) == 5
        types = [n.type for n in graph.nodes]
        assert types == ["Input", "Linear", "ReLU", "Linear", "Softmax"]

    def test_simple_diagram_edges(self) -> None:
        """Parsed graph must have 4 edges in order."""
        graph = parse_diagram(SIMPLE_MERMAID, fmt="mermaid")
        assert len(graph.edges) == 4
        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert edge_pairs == [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]

    def test_explicit_params(self) -> None:
        """Key=value params must be parsed correctly."""
        graph = parse_diagram(SIMPLE_MERMAID, fmt="mermaid")
        input_node = graph.nodes[0]
        assert input_node.params == {"shape": 784}

    def test_shorthand_params(self) -> None:
        """Shorthand primary parameter must be resolved."""
        graph = parse_diagram(SHORTHAND_MERMAID, fmt="mermaid")
        input_node = graph.nodes[0]
        assert input_node.params == {"shape": 784}
        dropout_node = next(n for n in graph.nodes if n.type == "Dropout")
        assert dropout_node.params == {"p": 0.5}

    def test_conv_diagram(self) -> None:
        """Conv2D with multiple params must parse correctly."""
        graph = parse_diagram(CONV_MERMAID, fmt="mermaid")
        conv_node = next(n for n in graph.nodes if n.type == "Conv2D")
        assert conv_node.params == {"out_channels": 32, "kernel_size": 3}

    def test_no_param_layers(self) -> None:
        """Layers like ReLU and Flatten must have empty params."""
        graph = parse_diagram(CONV_MERMAID, fmt="mermaid")
        relu_node = next(n for n in graph.nodes if n.type == "ReLU")
        assert relu_node.params == {}
        flatten_node = next(n for n in graph.nodes if n.type == "Flatten")
        assert flatten_node.params == {}

    def test_ir_serialization(self) -> None:
        """to_dict() must produce 'from'/'to' keys (not source/target)."""
        graph = parse_diagram(SIMPLE_MERMAID, fmt="mermaid")
        data = graph.to_dict()
        assert "from" in data["edges"][0]
        assert "to" in data["edges"][0]

    def test_empty_content_raises(self) -> None:
        with pytest.raises(EmptyDiagramError):
            parse_diagram("", fmt="mermaid")

    def test_missing_header_raises(self) -> None:
        with pytest.raises(DiagramSyntaxError, match="Expected 'graph TD"):
            parse_diagram("A[Input: 784] --> B[Linear: 10]", fmt="mermaid")

    def test_unsupported_layer_raises(self) -> None:
        bad = "graph TD\n    A[Input: 784] --> B[LSTM: 128]"
        with pytest.raises(UnsupportedLayerError, match="LSTM"):
            parse_diagram(bad, fmt="mermaid")

    def test_invalid_param_format_raises(self) -> None:
        bad = "graph TD\n    A[Linear: foo, bar]"
        with pytest.raises(InvalidParameterError):
            parse_diagram(bad, fmt="mermaid")

    def test_no_nodes_raises(self) -> None:
        with pytest.raises(EmptyDiagramError, match="No nodes found"):
            parse_diagram("graph TD\n    %% empty diagram", fmt="mermaid")


# ─── Draw.io parser ──────────────────────────────────────────────

class TestDrawioParser:
    """Tests for the Draw.io XML parser."""

    def test_simple_drawio_nodes(self) -> None:
        graph = parse_diagram(SIMPLE_DRAWIO, fmt="drawio")
        assert len(graph.nodes) == 3
        types = [n.type for n in graph.nodes]
        assert types == ["Input", "Linear", "ReLU"]

    def test_simple_drawio_edges(self) -> None:
        graph = parse_diagram(SIMPLE_DRAWIO, fmt="drawio")
        assert len(graph.edges) == 2
        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert edge_pairs == [("n1", "n2"), ("n2", "n3")]

    def test_drawio_params(self) -> None:
        graph = parse_diagram(SIMPLE_DRAWIO, fmt="drawio")
        input_node = graph.nodes[0]
        assert input_node.params == {"shape": 784}
        linear_node = graph.nodes[1]
        assert linear_node.params == {"out_features": 256}

    def test_invalid_xml_raises(self) -> None:
        with pytest.raises(DiagramSyntaxError, match="Invalid XML"):
            parse_diagram("<not-closed", fmt="drawio")

    def test_no_mxcells_raises(self) -> None:
        with pytest.raises(EmptyDiagramError, match="No mxCell"):
            parse_diagram("<mxGraphModel><root></root></mxGraphModel>", fmt="drawio")

    def test_unsupported_layer_in_drawio_raises(self) -> None:
        xml = """\
<mxGraphModel><root>
    <mxCell id="0"/><mxCell id="1" parent="0"/>
    <mxCell id="n1" value="GRU: 128" vertex="1" parent="1"/>
</root></mxGraphModel>"""
        with pytest.raises(UnsupportedLayerError, match="GRU"):
            parse_diagram(xml, fmt="drawio")

    def test_empty_drawio_raises(self) -> None:
        with pytest.raises(EmptyDiagramError):
            parse_diagram("", fmt="drawio")


# ─── Auto-detect integration ─────────────────────────────────────

class TestAutoDetect:
    """Tests for parse_diagram with auto format detection."""

    def test_auto_detect_mermaid(self) -> None:
        graph = parse_diagram(SIMPLE_MERMAID)
        assert len(graph.nodes) == 5

    def test_auto_detect_drawio(self) -> None:
        graph = parse_diagram(SIMPLE_DRAWIO)
        assert len(graph.nodes) == 3
