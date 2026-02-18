"""Tests for the Keras code generator.

Verifies that generated code:
1. Is valid Python that imports without error.
2. Builds a tf.keras.Model.
3. Passes a dummy input through predict/call.
"""

import importlib.util
import sys
import tempfile
import textwrap

import pytest

from app.generator import generate_keras
from app.ir.models import IREdge, IRGraph, IRNode
from app.parser import parse_diagram
from app.validator import validate


# Lazy-import TF to give a clear skip message if not installed
tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")


# ─── Helpers ──────────────────────────────────────────────────────

def _make_graph(nodes: list[IRNode], edges: list[tuple[str, str]]) -> IRGraph:
    return IRGraph(
        nodes=nodes,
        edges=[IREdge(**{"from": s, "to": t}) for s, t in edges],
    )


def _load_module(code: str, module_name: str = "gen_keras"):
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

class TestKerasMLPGeneration:
    """Tests for Keras fully-connected network generation."""

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
        code = generate_keras(self.MLP_GRAPH)
        compile(code, "<generated>", "exec")

    def test_code_imports_successfully(self) -> None:
        """Generated module must import without errors."""
        code = generate_keras(self.MLP_GRAPH)
        mod = _load_module(code, "test_keras_mlp_import")
        assert hasattr(mod, "GeneratedModel")

    def test_model_is_keras_model(self) -> None:
        """Generated object must be a tf.keras.Model."""
        code = generate_keras(self.MLP_GRAPH)
        mod = _load_module(code, "test_keras_mlp_type")
        assert isinstance(mod.GeneratedModel, tf.keras.Model)

    def test_forward_pass(self) -> None:
        """A dummy tensor must pass through the model."""
        code = generate_keras(self.MLP_GRAPH)
        mod = _load_module(code, "test_keras_mlp_fwd")
        model = mod.GeneratedModel
        x = tf.random.normal((2, 784))
        out = model(x, training=False)
        assert out.shape == (2, 10)

    def test_custom_class_name(self) -> None:
        """class_name argument must be reflected in generated code."""
        code = generate_keras(self.MLP_GRAPH, class_name="MyKerasMLP")
        assert "create_mykerasmlp" in code
        mod = _load_module(code, "test_keras_mlp_name")
        assert hasattr(mod, "MyKerasMLP")


# ─── CNN tests ────────────────────────────────────────────────────

class TestKerasCNNGeneration:
    """Tests for Keras convolutional network generation."""

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
        code = generate_keras(self.CNN_GRAPH)
        compile(code, "<generated>", "exec")

    def test_model_builds(self) -> None:
        code = generate_keras(self.CNN_GRAPH)
        mod = _load_module(code, "test_keras_cnn_build")
        assert isinstance(mod.GeneratedModel, tf.keras.Model)

    def test_forward_pass(self) -> None:
        """Dummy 28x28 single-channel input must pass through CNN."""
        code = generate_keras(self.CNN_GRAPH)
        mod = _load_module(code, "test_keras_cnn_fwd")
        model = mod.GeneratedModel
        # Keras channels_last: (batch, H, W, C)
        x = tf.random.normal((2, 28, 28, 1))
        out = model(x, training=False)
        assert out.shape == (2, 10)

    def test_uses_batchnormalization(self) -> None:
        """Keras must use BatchNormalization (not BatchNorm2d)."""
        code = generate_keras(self.CNN_GRAPH)
        assert "BatchNormalization" in code


# ─── Edge cases ───────────────────────────────────────────────────

class TestKerasEdgeCases:

    def test_single_dense_layer(self) -> None:
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 10}),
                IRNode(id="B", type="Linear", params={"out_features": 5}),
            ],
            edges=[("A", "B")],
        )
        code = generate_keras(graph)
        mod = _load_module(code, "test_keras_single")
        model = mod.GeneratedModel
        x = tf.random.normal((1, 10))
        out = model(x, training=False)
        assert out.shape == (1, 5)

    def test_sigmoid_activation(self) -> None:
        graph = _make_graph(
            nodes=[
                IRNode(id="A", type="Input", params={"shape": 4}),
                IRNode(id="B", type="Linear", params={"out_features": 1}),
                IRNode(id="C", type="Sigmoid"),
            ],
            edges=[("A", "B"), ("B", "C")],
        )
        code = generate_keras(graph)
        mod = _load_module(code, "test_keras_sigmoid")
        model = mod.GeneratedModel
        x = tf.random.normal((3, 4))
        out = model(x, training=False)
        assert out.shape == (3, 1)


# ─── End-to-end: Mermaid → Keras ─────────────────────────────────

class TestKerasEndToEnd:

    def test_mermaid_to_running_keras_model(self) -> None:
        mermaid = textwrap.dedent("""\
            graph TD
                A[Input: shape=784] --> B[Linear: 256]
                B --> C[ReLU]
                C --> D[Dropout: 0.5]
                D --> E[Linear: 10]
                E --> F[Softmax]
        """)

        graph = parse_diagram(mermaid)
        validate(graph)
        code = generate_keras(graph)

        mod = _load_module(code, "test_keras_e2e")
        model = mod.GeneratedModel
        x = tf.random.normal((4, 784))
        out = model(x, training=False)
        assert out.shape == (4, 10)
