"""Tensor shape inference engine.

Walks the IR graph in topological order and computes the output shape
at each node, so that downstream layers (e.g. nn.Linear) can know
their required input dimensions.

Shape convention:
    MLP path  → (features,)
    CNN path  → (channels, height, width)

After Flatten, a CNN path collapses to (features,).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from app.ir.models import IRGraph, IRNode

logger = logging.getLogger(__name__)

# Default spatial dimensions when Input only specifies channels.
DEFAULT_HEIGHT = 28
DEFAULT_WIDTH = 28


@dataclass(frozen=True)
class Shape:
    """Immutable tensor shape (excluding batch dimension)."""

    dims: tuple[int, ...]

    @property
    def is_spatial(self) -> bool:
        """True if shape is (C, H, W)."""
        return len(self.dims) == 3

    @property
    def features(self) -> int:
        """Total feature count (product of all dims)."""
        result = 1
        for d in self.dims:
            result *= d
        return result

    @property
    def channels(self) -> int:
        """Channel count (first dim of spatial shape)."""
        if not self.is_spatial:
            raise ValueError(f"Shape {self.dims} is not spatial (C, H, W)")
        return self.dims[0]

    @property
    def height(self) -> int:
        if not self.is_spatial:
            raise ValueError(f"Shape {self.dims} is not spatial")
        return self.dims[1]

    @property
    def width(self) -> int:
        if not self.is_spatial:
            raise ValueError(f"Shape {self.dims} is not spatial")
        return self.dims[2]


def _has_conv_layers(graph: IRGraph) -> bool:
    """Check if the graph contains any Conv2D layers."""
    return any(node.type == "Conv2D" for node in graph.nodes)


def infer_input_shape(node: IRNode, is_cnn: bool) -> Shape:
    """Determine the starting shape from an Input node.

    Args:
        node: The Input node.
        is_cnn: Whether the network contains convolution layers.

    Returns:
        The input shape (excluding batch dimension).
    """
    shape_val = node.params.get("shape", 1)
    height = node.params.get("height", DEFAULT_HEIGHT)
    width = node.params.get("width", DEFAULT_WIDTH)

    if is_cnn:
        return Shape((int(shape_val), int(height), int(width)))
    return Shape((int(shape_val),))


def compute_shapes(
    graph: IRGraph, sorted_ids: list[str]
) -> dict[str, Shape]:
    """Compute the output shape for every node in topological order.

    Args:
        graph: The validated IRGraph.
        sorted_ids: Node IDs in topological order.

    Returns:
        Mapping from node ID to its output Shape.
    """
    node_map: dict[str, IRNode] = {n.id: n for n in graph.nodes}
    # Build predecessor lookup (first predecessor in topo-order wins)
    predecessors: dict[str, str] = {}
    for edge in graph.edges:
        if edge.target not in predecessors:
            predecessors[edge.target] = edge.source

    is_cnn = _has_conv_layers(graph)
    shapes: dict[str, Shape] = {}

    for nid in sorted_ids:
        node = node_map[nid]
        prev_shape = shapes.get(predecessors.get(nid, ""))

        shapes[nid] = _apply_layer(node, prev_shape, is_cnn)

    return shapes


def _apply_layer(node: IRNode, prev: Shape | None, is_cnn: bool) -> Shape:
    """Compute the output shape after applying a single layer."""
    params = node.params

    if node.type == "Input":
        return infer_input_shape(node, is_cnn)

    if prev is None:
        raise ValueError(f"Node '{node.id}' ({node.type}) has no predecessor shape")

    if node.type in ("ReLU", "Sigmoid", "Dropout"):
        return prev

    if node.type == "Softmax":
        return prev

    if node.type in ("Linear", "Dense"):
        out = int(params["out_features"]) if node.type == "Linear" else int(params["units"])
        return Shape((out,))

    if node.type == "Conv2D":
        if not prev.is_spatial:
            raise ValueError(f"Conv2D node '{node.id}' requires spatial input, got {prev.dims}")
        out_ch = int(params["out_channels"])
        k = int(params.get("kernel_size", 3))
        s = int(params.get("stride", 1))
        p = int(params.get("padding", 0))
        h_out = math.floor((prev.height - k + 2 * p) / s) + 1
        w_out = math.floor((prev.width - k + 2 * p) / s) + 1
        return Shape((out_ch, h_out, w_out))

    if node.type == "MaxPool2D":
        if not prev.is_spatial:
            raise ValueError(f"MaxPool2D node '{node.id}' requires spatial input")
        k = int(params["kernel_size"])
        s = int(params.get("stride", k))  # default stride = kernel_size
        h_out = math.floor((prev.height - k) / s) + 1
        w_out = math.floor((prev.width - k) / s) + 1
        return Shape((prev.channels, h_out, w_out))

    if node.type == "BatchNorm":
        return prev

    if node.type == "Flatten":
        return Shape((prev.features,))

    raise ValueError(f"Unknown layer type for shape tracking: '{node.type}'")
