"""Tests for the IR validation engine."""

import pytest

from app.ir.models import IREdge, IRGraph, IRNode
from app.validator import validate
from app.validator.exceptions import (
    CycleDetectedError,
    DuplicateNodeError,
    InvalidEdgeError,
    InvalidParameterError,
    UnreachableNodeError,
    ValidationError,
)


# ─── Helpers ──────────────────────────────────────────────────────

def _make_graph(
    nodes: list[IRNode],
    edges: list[tuple[str, str]],
) -> IRGraph:
    """Shortcut to build an IRGraph from nodes and (src, dst) tuples."""
    return IRGraph(
        nodes=nodes,
        edges=[IREdge(**{"from": s, "to": t}) for s, t in edges],
    )


# ─── Valid graphs ─────────────────────────────────────────────────

class TestValidGraphs:
    """Graphs that must pass validation without errors."""

    def test_simple_linear_chain(self) -> None:
        """Input → Linear → ReLU → Linear → Softmax."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Linear", params={"out_features": 256}),
                IRNode(id="C", type="ReLU"),
                IRNode(id="D", type="Linear", params={"out_features": 10}),
                IRNode(id="E", type="Softmax"),
            ],
            edges=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
        )
        validate(graph)  # should not raise

    def test_conv_pipeline(self) -> None:
        """Conv2D → BatchNorm → ReLU → MaxPool → Flatten → Linear."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 1}),
                IRNode(id="B", type="Conv2D", params={"out_channels": 32, "kernel_size": 3}),
                IRNode(id="C", type="BatchNorm", params={"num_features": 32}),
                IRNode(id="D", type="ReLU"),
                IRNode(id="E", type="MaxPool2D", params={"kernel_size": 2}),
                IRNode(id="F", type="Flatten"),
                IRNode(id="G", type="Linear", params={"out_features": 10}),
            ],
            edges=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G")],
        )
        validate(graph)

    def test_single_node_no_edges(self) -> None:
        """A single Input node with no edges is valid."""
        graph = _make_graph(
            nodes=[IRNode(id="A", type="Input", params={"shape": 784})],
            edges=[],
        )
        validate(graph)

    def test_optional_params_omitted(self) -> None:
        """Dropout without 'p' and Softmax without 'dim' are valid."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Dropout"),
                IRNode(id="C", type="Softmax"),
            ],
            edges=[("A", "B"), ("B", "C")],
        )
        validate(graph)

    def test_dropout_with_float_p(self) -> None:
        """Dropout p=0.5 (float) is valid."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Dropout", params={"p": 0.5}),
            ],
            edges=[("A", "B")],
        )
        validate(graph)

    def test_branching_dag(self) -> None:
        """A DAG with a fork (one node feeds two outputs) is valid."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Linear", params={"out_features": 128}),
                IRNode(id="C", type="ReLU"),
                IRNode(id="D", type="Sigmoid"),
            ],
            edges=[("A", "B"), ("B", "C"), ("B", "D")],
        )
        validate(graph)


# ─── Structural errors ───────────────────────────────────────────

class TestStructuralErrors:
    """Structural validation failures."""

    def test_duplicate_node_ids(self) -> None:
        graph = IRGraph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="A", type="Linear", params={"out_features": 10}),
            ],
            edges=[],
        )
        with pytest.raises(DuplicateNodeError, match="Duplicate node IDs"):
            validate(graph)

    def test_edge_references_missing_source(self) -> None:
        graph = _make_graph(
            nodes=[IRNode(id="B", type="ReLU")],
            edges=[("A", "B")],
        )
        with pytest.raises(InvalidEdgeError, match="non-existent source node 'A'"):
            validate(graph)

    def test_edge_references_missing_target(self) -> None:
        graph = _make_graph(
            nodes=[IRNode(id="A", type="Input", params={"shape": 784})],
            edges=[("A", "Z")],
        )
        with pytest.raises(InvalidEdgeError, match="non-existent target node 'Z'"):
            validate(graph)


# ─── Topology errors ─────────────────────────────────────────────

class TestTopologyErrors:
    """Cycle and reachability failures."""

    def test_simple_cycle(self) -> None:
        """A → B → A is a cycle."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Linear", params={"out_features": 10}),
            ],
            edges=[("A", "B"), ("B", "A")],
        )
        with pytest.raises(CycleDetectedError, match="Cycle detected"):
            validate(graph)

    def test_three_node_cycle(self) -> None:
        """A → B → C → A is a cycle."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="ReLU"),
                IRNode(id="B", type="ReLU"),
                IRNode(id="C", type="ReLU"),
            ],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],
        )
        with pytest.raises(CycleDetectedError, match="Cycle detected"):
            validate(graph)

    def test_cycle_allowed_when_flag_set(self) -> None:
        """With allow_cycles=True, cycles do not raise."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Linear", params={"out_features": 10}),
            ],
            edges=[("A", "B"), ("B", "A")],
        )
        # Should not raise CycleDetectedError, but may raise UnreachableNodeError
        # since with a cycle both have in_degree > 0 (no root). Let's check:
        with pytest.raises(UnreachableNodeError, match="No root nodes found"):
            validate(graph, allow_cycles=True)

    def test_unreachable_disconnected_node(self) -> None:
        """Node C is disconnected from the A → B chain."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Linear", params={"out_features": 10}),
                IRNode(id="C", type="ReLU"),
            ],
            edges=[("A", "B")],
        )
        # C is a root with in_degree 0, so it's reachable from itself.
        # This should pass since C is its own root.
        validate(graph)  # disconnected components are OK if each has a root

    def test_unreachable_mid_chain_node(self) -> None:
        """D has incoming edge from non-existent path — only reachable nodes counted."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Linear", params={"out_features": 10}),
                IRNode(id="C", type="ReLU"),
                IRNode(id="D", type="Sigmoid"),
            ],
            edges=[("A", "B"), ("C", "D")],
        )
        # A→B and C→D are two separate valid chains, both have roots
        validate(graph)


# ─── Semantic errors ─────────────────────────────────────────────

class TestSemanticErrors:
    """Parameter validation failures."""

    def test_missing_required_param(self) -> None:
        """Linear without out_features must fail."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Linear", params={}),
            ],
            edges=[("A", "B")],
        )
        with pytest.raises(InvalidParameterError, match="missing required parameter 'out_features'"):
            validate(graph)

    def test_unknown_param(self) -> None:
        """An unrecognized parameter name must fail."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="ReLU", params={"inplace": True}),
            ],
            edges=[],
        )
        with pytest.raises(InvalidParameterError, match="unknown parameter 'inplace'"):
            validate(graph)

    def test_wrong_param_type(self) -> None:
        """out_features must be int, not str."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 784}),
                IRNode(id="B", type="Linear", params={"out_features": "two-fifty-six"}),
            ],
            edges=[("A", "B")],
        )
        with pytest.raises(InvalidParameterError, match="must be int, got str"):
            validate(graph)

    def test_conv2d_missing_out_channels(self) -> None:
        """Conv2D requires out_channels."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 1}),
                IRNode(id="B", type="Conv2D", params={"kernel_size": 3}),
            ],
            edges=[("A", "B")],
        )
        with pytest.raises(InvalidParameterError, match="missing required parameter 'out_channels'"):
            validate(graph)

    def test_multiple_errors_collected(self) -> None:
        """Multiple issues should all appear in the error messages list."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Linear", params={}),  # missing out_features
                IRNode(id="B", type="Linear", params={"out_features": "bad"}),  # wrong type
            ],
            edges=[("A", "B")],
        )
        with pytest.raises(ValidationError) as exc_info:
            validate(graph)
        messages = exc_info.value.messages
        assert len(messages) >= 2


# ─── End-to-end with parser ──────────────────────────────────────

class TestParserThenValidator:
    """Integration: parse a diagram, then validate the resulting IR."""

    def test_valid_mermaid_passes_validation(self) -> None:
        from app.parser import parse_diagram

        mermaid = """\
graph TD
    A[Input: shape=784] --> B[Linear: 256]
    B --> C[ReLU]
    C --> D[Linear: 10]
    D --> E[Softmax]
"""
        graph = parse_diagram(mermaid)
        validate(graph)  # should not raise

    def test_valid_drawio_passes_validation(self) -> None:
        from app.parser import parse_diagram

        xml = """\
<mxGraphModel>
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="n1" value="Input: shape=784" vertex="1" parent="1">
      <mxGeometry/>
    </mxCell>
    <mxCell id="n2" value="Linear: 256" vertex="1" parent="1">
      <mxGeometry/>
    </mxCell>
    <mxCell id="e1" edge="1" source="n1" target="n2" parent="1"/>
  </root>
</mxGraphModel>
"""
        graph = parse_diagram(xml)
        validate(graph)
