"""PyTorch code generator.

Takes a validated IRGraph and produces a complete, executable Python file
containing an nn.Module subclass with __init__ and forward methods.
"""

import logging
from typing import Any

from app.ir.models import IRGraph, IRNode
from app.generator.shape_tracker import Shape, compute_shapes
from app.generator.topo_sort import topological_sort

logger = logging.getLogger(__name__)


def generate_pytorch(graph: IRGraph, class_name: str = "GeneratedModel") -> str:
    """Generate a complete PyTorch nn.Module source file from an IR graph.

    Args:
        graph: A validated IRGraph (acyclic, all params valid).
        class_name: Name for the generated class.

    Returns:
        A string containing valid, executable Python code.
    """
    sorted_ids = topological_sort(graph)
    shapes = compute_shapes(graph, sorted_ids)
    node_map: dict[str, IRNode] = {n.id: n for n in graph.nodes}

    # Build predecessor map for in_features/in_channels inference
    predecessors: dict[str, str] = {}
    for edge in graph.edges:
        if edge.target not in predecessors:
            predecessors[edge.target] = edge.source

    # Collect init lines and forward lines
    init_lines: list[str] = []
    forward_lines: list[str] = []
    attr_names: dict[str, str] = {}  # node_id → self.attr_name

    for nid in sorted_ids:
        node = node_map[nid]
        if node.type == "Input":
            continue  # Input is just the function argument

        attr = _make_attr_name(node, attr_names)
        attr_names[nid] = attr

        prev_id = predecessors.get(nid)
        prev_shape = shapes.get(prev_id, None) if prev_id else None

        init_line = _gen_init_line(attr, node, prev_shape)
        init_lines.append(init_line)
        forward_lines.append(f"        x = self.{attr}(x)")

    return _assemble_file(class_name, init_lines, forward_lines)


def _make_attr_name(node: IRNode, existing: dict[str, str]) -> str:
    """Generate a unique, descriptive attribute name for a layer.

    Produces names like: linear_b, relu_c, conv2d_1.
    """
    base = node.type.lower()
    candidate = f"{base}_{node.id.lower()}"

    # Ensure uniqueness
    used = set(existing.values())
    if candidate not in used:
        return candidate

    counter = 2
    while f"{candidate}_{counter}" in used:
        counter += 1
    return f"{candidate}_{counter}"


def _gen_init_line(attr: str, node: IRNode, prev_shape: Shape | None) -> str:
    """Generate a single self.attr = nn.Layer(...) line."""
    p = node.params
    t = node.type

    if t in ("Linear", "Dense"):
        out_f = int(p.get("out_features", p.get("units", 0)))
        in_f = prev_shape.features if prev_shape else out_f
        return f"        self.{attr} = nn.Linear({in_f}, {out_f})"

    if t == "Conv2D":
        out_ch = int(p["out_channels"])
        in_ch = prev_shape.channels if prev_shape and prev_shape.is_spatial else 1
        k = int(p.get("kernel_size", 3))
        s = int(p.get("stride", 1))
        pad = int(p.get("padding", 0))
        parts = [f"{in_ch}", f"{out_ch}", f"kernel_size={k}"]
        if s != 1:
            parts.append(f"stride={s}")
        if pad != 0:
            parts.append(f"padding={pad}")
        return f"        self.{attr} = nn.Conv2d({', '.join(parts)})"

    if t == "MaxPool2D":
        k = int(p["kernel_size"])
        s = int(p.get("stride", k))
        if s != k:
            return f"        self.{attr} = nn.MaxPool2d(kernel_size={k}, stride={s})"
        return f"        self.{attr} = nn.MaxPool2d(kernel_size={k})"

    if t == "Flatten":
        return f"        self.{attr} = nn.Flatten()"

    if t == "Dropout":
        p_val = p.get("p", 0.5)
        return f"        self.{attr} = nn.Dropout(p={p_val})"

    if t == "BatchNorm":
        nf = int(p["num_features"])
        # Use BatchNorm2d if predecessor is spatial, else BatchNorm1d
        if prev_shape and prev_shape.is_spatial:
            return f"        self.{attr} = nn.BatchNorm2d({nf})"
        return f"        self.{attr} = nn.BatchNorm1d({nf})"

    if t == "ReLU":
        return f"        self.{attr} = nn.ReLU()"

    if t == "Sigmoid":
        return f"        self.{attr} = nn.Sigmoid()"

    if t == "Softmax":
        dim = int(p.get("dim", -1))
        return f"        self.{attr} = nn.Softmax(dim={dim})"

    raise ValueError(f"Unsupported layer type for PyTorch generation: '{t}'")


def _assemble_file(
    class_name: str,
    init_lines: list[str],
    forward_lines: list[str],
) -> str:
    """Assemble the final Python source file."""
    init_body = "\n".join(init_lines) if init_lines else "        pass"
    forward_body = "\n".join(forward_lines) if forward_lines else "        pass"

    return f'''import torch
import torch.nn as nn


class {class_name}(nn.Module):
    """Auto-generated PyTorch model."""

    def __init__(self) -> None:
        super().__init__()
{init_body}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
{forward_body}
        return x
'''
