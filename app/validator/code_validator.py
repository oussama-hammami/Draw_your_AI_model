"""Generated code validation engine.

Provides three levels of validation for generated model code:
1. Syntax check   — compiles the code string.
2. Import check   — executes the module in an isolated namespace.
3. Runtime check  — instantiates the model and runs a dummy forward pass.

Optionally, if code fails validation, the LLM can attempt auto-fix.
"""

import importlib.util
import logging
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation."""

    valid: bool
    syntax_ok: bool = False
    import_ok: bool = False
    runtime_ok: bool = False
    errors: list[str] = field(default_factory=list)
    fixed_code: str | None = None


def validate_syntax(code: str) -> tuple[bool, str | None]:
    """Check if code compiles without syntax errors.

    Returns:
        (True, None) on success, (False, error_message) on failure.
    """
    try:
        compile(code, "<generated>", "exec")
        return True, None
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"


def validate_import(code: str, module_name: str = "_val_module") -> tuple[bool, str | None]:
    """Write code to a temp file and import it.

    Returns:
        (True, None) on success, (False, error_message) on failure.
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(code)
            f.flush()
            spec = importlib.util.spec_from_file_location(module_name, f.name)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
        return True, None
    except Exception as exc:
        return False, f"ImportError: {exc}"
    finally:
        sys.modules.pop(module_name, None)


def validate_runtime_pytorch(code: str) -> tuple[bool, str | None]:
    """Instantiate a PyTorch model and run a dummy forward pass.

    Expects the code to define a class named 'GeneratedModel'.

    Returns:
        (True, None) on success, (False, error_message) on failure.
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch not installed"

    try:
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        model_cls = namespace.get("GeneratedModel")
        if model_cls is None:
            return False, "No 'GeneratedModel' class found in generated code"

        model = model_cls()
        model.eval()

        # Determine input shape from the first layer
        input_shape = _infer_pytorch_input_shape(model)
        x = torch.randn(1, *input_shape)

        with torch.no_grad():
            model(x)

        return True, None
    except Exception as exc:
        return False, f"RuntimeError: {exc}\n{traceback.format_exc()}"


def validate_runtime_keras(code: str) -> tuple[bool, str | None]:
    """Build a Keras model and run a dummy forward pass.

    Expects the code to define a module-level 'GeneratedModel' variable.

    Returns:
        (True, None) on success, (False, error_message) on failure.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return False, "TensorFlow not installed"

    try:
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        model = namespace.get("GeneratedModel")
        if model is None:
            return False, "No 'GeneratedModel' variable found in generated code"

        # Get input shape from model
        input_shape = model.input_shape
        # input_shape is like (None, 784) or (None, 28, 28, 1)
        concrete_shape = tuple(1 if d is None else d for d in input_shape)
        x = tf.random.normal(concrete_shape)
        model(x, training=False)

        return True, None
    except Exception as exc:
        return False, f"RuntimeError: {exc}\n{traceback.format_exc()}"


def validate_code(code: str, framework: str = "pytorch") -> ValidationResult:
    """Run all validation passes on generated code.

    Args:
        code: The generated Python source code.
        framework: 'pytorch' or 'keras'.

    Returns:
        A ValidationResult with details of each check.
    """
    result = ValidationResult(valid=False)

    # 1. Syntax
    syn_ok, syn_err = validate_syntax(code)
    result.syntax_ok = syn_ok
    if not syn_ok:
        result.errors.append(syn_err)
        return result

    # 2. Import
    imp_ok, imp_err = validate_import(code)
    result.import_ok = imp_ok
    if not imp_ok:
        result.errors.append(imp_err)
        return result

    # 3. Runtime
    if framework == "pytorch":
        rt_ok, rt_err = validate_runtime_pytorch(code)
    elif framework == "keras":
        rt_ok, rt_err = validate_runtime_keras(code)
    else:
        result.errors.append(f"Unknown framework: {framework}")
        return result

    result.runtime_ok = rt_ok
    if not rt_ok:
        result.errors.append(rt_err)
        return result

    result.valid = True
    logger.info("Code validation passed for %s", framework)
    return result


def _infer_pytorch_input_shape(model: Any) -> tuple[int, ...]:
    """Infer the input tensor shape from a PyTorch model's first layer."""
    import torch.nn as nn

    for module in model.modules():
        if module is model:
            continue
        if isinstance(module, nn.Linear):
            return (module.in_features,)
        if isinstance(module, nn.Conv2d):
            # Assume 28x28 input for CNN
            return (module.in_channels, 28, 28)

    return (1,)
