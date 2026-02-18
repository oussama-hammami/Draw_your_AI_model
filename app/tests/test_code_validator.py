"""Tests for the generated code validation engine."""

import pytest

from app.generator import generate_keras, generate_pytorch
from app.ir.models import IREdge, IRGraph, IRNode
from app.validator.code_validator import (
    ValidationResult,
    validate_code,
    validate_import,
    validate_syntax,
)


# ─── Helpers ──────────────────────────────────────────────────────

def _make_graph(nodes: list[IRNode], edges: list[tuple[str, str]]) -> IRGraph:
    return IRGraph(
        nodes=nodes,
        edges=[IREdge(**{"from": s, "to": t}) for s, t in edges],
    )


MLP_GRAPH = _make_graph(
    nodes=[
        IRNode(id="A", type="Input", params={"shape": 784}),
        IRNode(id="B", type="Linear", params={"out_features": 256}),
        IRNode(id="C", type="ReLU"),
        IRNode(id="D", type="Linear", params={"out_features": 10}),
    ],
    edges=[("A", "B"), ("B", "C"), ("C", "D")],
)


# ─── Syntax validation ───────────────────────────────────────────

class TestSyntaxValidation:

    def test_valid_code_passes(self) -> None:
        ok, err = validate_syntax("x = 1 + 2")
        assert ok is True
        assert err is None

    def test_syntax_error_detected(self) -> None:
        ok, err = validate_syntax("def foo(:\n  pass")
        assert ok is False
        assert "SyntaxError" in err

    def test_generated_pytorch_passes_syntax(self) -> None:
        code = generate_pytorch(MLP_GRAPH)
        ok, err = validate_syntax(code)
        assert ok is True

    def test_generated_keras_passes_syntax(self) -> None:
        code = generate_keras(MLP_GRAPH)
        ok, err = validate_syntax(code)
        assert ok is True


# ─── Import validation ───────────────────────────────────────────

class TestImportValidation:

    def test_valid_code_imports(self) -> None:
        ok, err = validate_import("x = 42")
        assert ok is True

    def test_import_error_detected(self) -> None:
        ok, err = validate_import("import nonexistent_module_xyz")
        assert ok is False
        assert "ImportError" in err


# ─── Full validation (PyTorch) ────────────────────────────────────

class TestPyTorchCodeValidation:

    def test_valid_pytorch_code_passes_all(self) -> None:
        code = generate_pytorch(MLP_GRAPH)
        result = validate_code(code, framework="pytorch")
        assert result.valid is True
        assert result.syntax_ok is True
        assert result.import_ok is True
        assert result.runtime_ok is True
        assert result.errors == []

    def test_broken_syntax_fails(self) -> None:
        result = validate_code("class Foo(:\n  pass", framework="pytorch")
        assert result.valid is False
        assert result.syntax_ok is False

    def test_broken_import_fails(self) -> None:
        code = "import nonexistent_xyz_module"
        result = validate_code(code, framework="pytorch")
        assert result.valid is False
        assert result.import_ok is False

    def test_missing_class_fails_runtime(self) -> None:
        code = "import torch\nx = 42"
        result = validate_code(code, framework="pytorch")
        assert result.valid is False
        assert result.runtime_ok is False


# ─── Full validation (Keras) ─────────────────────────────────────

class TestKerasCodeValidation:

    def test_valid_keras_code_passes_all(self) -> None:
        code = generate_keras(MLP_GRAPH)
        result = validate_code(code, framework="keras")
        assert result.valid is True
        assert result.syntax_ok is True
        assert result.import_ok is True
        assert result.runtime_ok is True

    def test_broken_keras_code_fails(self) -> None:
        code = "import tensorflow as tf\nx = 42"
        result = validate_code(code, framework="keras")
        assert result.valid is False
        assert result.runtime_ok is False


# ─── Valid code unchanged ─────────────────────────────────────────

class TestValidCodeUnchanged:

    def test_pytorch_code_is_not_modified_by_validation(self) -> None:
        """Validation must not alter the code."""
        code = generate_pytorch(MLP_GRAPH)
        result = validate_code(code, framework="pytorch")
        assert result.valid is True
        # The code hasn't been modified
        assert result.fixed_code is None
