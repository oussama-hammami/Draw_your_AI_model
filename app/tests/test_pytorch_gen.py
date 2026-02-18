"""Tests for the PyTorch code generator.

Verifies that generated code:
1. Is valid Python that imports without error.
2. Instantiates an nn.Module.
3. Passes a dummy tensor through forward().
"""

import importlib.util
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
import torch

from app.generator import generate_pytorch
from app.ir.models import IREdge, IRGraph, IRNode
from app.parser import parse_diagram
from app.validator import validate


# ─── Helpers ──────────────────────────────────────────────────────

def _make_graph(nodes: list[IRNode], edges: list[tuple[str, str]]) -> IRGraph:
    return IRGraph(
        nodes=nodes,
        edges=[IREdge(**{"from": s, "to": t}) for s, t in edges],
    )


def _load_module(code: str, module_name: str = "gen_model"):
    """Write code to a temp file, import it, return the module."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(code)
        f.flush()
        spec = importlib.util.spec_from_file_location(module_name, f.name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod


# ─── MLP tests ────────────────────────────────────────────────────

class TestMLPGeneration:
    """Tests for fully-connected (MLP) network generation."""

    MLP_GRAPH = _make_graph(
        nodes=[
            IRNode(id="A", type="Input", params={"shape": 784}),
            IRNode(id="B", type="Linear", params={"out_features": 256}),
            IRNode(id="C", type="ReLU"),
            IRNode(id="D", type="Dropout", params={"p": 0.5}),
            IRNode(id="E", type="Linear", params={"out_features": 10}),
            IRNode(id="F", type="Softmax"),
        ],
        edges=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")],
    )

    def test_code_is_valid_python(self) -> None:
        """Generated code must compile without syntax errors."""
        code = generate_pytorch(self.MLP_GRAPH)
        compile(code, "<generated>", "exec")

    def test_code_imports_successfully(self) -> None:
        """Generated module must import without errors."""
        code = generate_pytorch(self.MLP_GRAPH)
        mod = _load_module(code, "test_mlp_import")
        assert hasattr(mod, "GeneratedModel")

    def test_model_instantiates(self) -> None:
        """Generated class must instantiate as nn.Module."""
        code = generate_pytorch(self.MLP_GRAPH)
        mod = _load_module(code, "test_mlp_inst")
        model = mod.GeneratedModel()
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self) -> None:
        """A dummy tensor must pass through forward() without error."""
        code = generate_pytorch(self.MLP_GRAPH)
        mod = _load_module(code, "test_mlp_fwd")
        model = mod.GeneratedModel()
        model.eval()
        x = torch.randn(2, 784)  # batch_size=2, features=784
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_custom_class_name(self) -> None:
        """class_name argument must be reflected in generated code."""
        code = generate_pytorch(self.MLP_GRAPH, class_name="MyMLP")
        assert "class MyMLP(nn.Module):" in code
        mod = _load_module(code, "test_mlp_name")
        assert hasattr(mod, "MyMLP")


# ─── CNN tests ────────────────────────────────────────────────────

class TestCNNGeneration:
    """Tests for convolutional network generation."""

    CNN_GRAPH = _make_graph(
        nodes=[
            IRNode(id="A", type="Input", params={"shape": 1}),
            IRNode(id="B", type="Conv2D", params={"out_channels": 32, "kernel_size": 3}),
            IRNode(id="C", type="BatchNorm", params={"num_features": 32}),
            IRNode(id="D", type="ReLU"),
            IRNode(id="E", type="MaxPool2D", params={"kernel_size": 2}),
            IRNode(id="F", type="Flatten"),
            IRNode(id="G", type="Linear", params={"out_features": 10}),
        ],
        edges=[
            ("A", "B"), ("B", "C"), ("C", "D"),
            ("D", "E"), ("E", "F"), ("F", "G"),
        ],
    )

    def test_code_is_valid_python(self) -> None:
        code = generate_pytorch(self.CNN_GRAPH)
        compile(code, "<generated>", "exec")

    def test_model_instantiates(self) -> None:
        code = generate_pytorch(self.CNN_GRAPH)
        mod = _load_module(code, "test_cnn_inst")
        model = mod.GeneratedModel()
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self) -> None:
        """Dummy 28x28 single-channel input must pass through CNN."""
        code = generate_pytorch(self.CNN_GRAPH)
        mod = _load_module(code, "test_cnn_fwd")
        model = mod.GeneratedModel()
        model.eval()
        x = torch.randn(2, 1, 28, 28)  # batch=2, C=1, H=28, W=28
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_batchnorm2d_used_for_conv(self) -> None:
        """BatchNorm after Conv2D must use BatchNorm2d, not BatchNorm1d."""
        code = generate_pytorch(self.CNN_GRAPH)
        assert "nn.BatchNorm2d(32)" in code


# ─── Edge cases ───────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case and integration tests."""

    def test_single_linear_layer(self) -> None:
        """Minimal: Input → Linear."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 10}),
                IRNode(id="B", type="Linear", params={"out_features": 5}),
            ],
            edges=[("A", "B")],
        )
        code = generate_pytorch(graph)
        mod = _load_module(code, "test_single_linear")
        model = mod.GeneratedModel()
        out = model(torch.randn(1, 10))
        assert out.shape == (1, 5)

    def test_sigmoid_activation(self) -> None:
        """Sigmoid must work as an activation layer."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 4}),
                IRNode(id="B", type="Linear", params={"out_features": 1}),
                IRNode(id="C", type="Sigmoid"),
            ],
            edges=[("A", "B"), ("B", "C")],
        )
        code = generate_pytorch(graph)
        mod = _load_module(code, "test_sigmoid")
        model = mod.GeneratedModel()
        out = model(torch.randn(3, 4))
        assert out.shape == (3, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_generated_code_contains_type_hints(self) -> None:
        """Generated code must include torch.Tensor type hints."""
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 10}),
                IRNode(id="B", type="Linear", params={"out_features": 5}),
            ],
            edges=[("A", "B")],
        )
        code = generate_pytorch(graph)
        assert "x: torch.Tensor" in code
        assert "-> torch.Tensor" in code


# ─── End-to-end: Mermaid → parse → validate → generate → run ────

class TestEndToEnd:
    """Full pipeline from Mermaid diagram to running model."""

    def test_mermaid_to_running_model(self) -> None:
        mermaid = textwrap.dedent("""\
            graph TD
                A[Input: shape=784] --> B[Linear: 256]
                B --> C[ReLU]
                C --> D[Dropout: 0.5]
                D --> E[Linear: 10]
                E --> F[Softmax]
        """)

        # Parse
        graph = parse_diagram(mermaid)
        assert len(graph.nodes) == 6

        # Validate
        validate(graph)

        # Generate
        code = generate_pytorch(graph)

        # Execute
        mod = _load_module(code, "test_e2e")
        model = mod.GeneratedModel()
        model.eval()
        x = torch.randn(4, 784)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10)
        # Softmax output should sum to ~1 per sample
        sums = out.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)
