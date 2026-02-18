"""Keras code generator.

Takes a validated IRGraph and produces a complete, executable Python file
containing a tf.keras model built with the Sequential API.

Key differences from PyTorch:
    - Keras uses 'Dense' (not 'Linear'), with 'units' parameter.
    - Conv2D uses 'filters' instead of 'out_channels'.
    - Keras 3 prefers layers.Input(shape=...) as the first Sequential element.
    - BatchNormalization (not BatchNorm1d/2d).
    - Keras channel convention is channels_last by default (H, W, C).
"""

import logging

from app.ir.models import IRGraph, IRNode
from app.generator.shape_tracker import compute_shapes, Shape
from app.generator.topo_sort import topological_sort

logger = logging.getLogger(__name__)


def generate_keras(graph: IRGraph, class_name: str = "GeneratedModel") -> str:
    """Generate a complete Keras model source file from an IR graph.

    Args:
        graph: A validated IRGraph (acyclic, all params valid).
        class_name: Name for the generated model function.

    Returns:
        A string containing valid, executable Python code.
    """
    sorted_ids = topological_sort(graph)
    shapes = compute_shapes(graph, sorted_ids)
    node_map: dict[str, IRNode] = {n.id: n for n in graph.nodes}

    predecessors: dict[str, str] = {}
    for edge in graph.edges:
        if edge.target not in predecessors:
            predecessors[edge.target] = edge.source

    layer_lines: list[str] = []

    for nid in sorted_ids:
        node = node_map[nid]

        if node.type == "Input":
            shape = shapes[nid]
            if shape.is_spatial:
                # Keras channels_last: (H, W, C)
                layer_lines.append(
                    f"        layers.Input(shape=({shape.height}, {shape.width}, {shape.channels})),"
                )
            else:
                layer_lines.append(
                    f"        layers.Input(shape=({shape.features},)),"
                )
            continue

        prev_id = predecessors.get(nid)
        prev_shape = shapes.get(prev_id) if prev_id else None

        line = _gen_layer_line(node, prev_shape)
        layer_lines.append(line)

    return _assemble_file(class_name, layer_lines)


def _gen_layer_line(node: IRNode, prev_shape: Shape | None) -> str:
    """Generate a single layers.Layer(...) line for the Sequential model."""
    p = node.params
    t = node.type

    if t in ("Linear", "Dense"):
        units = int(p.get("out_features", p.get("units", 0)))
        return f"        layers.Dense({units}),"

    if t == "Conv2D":
        filters = int(p["out_channels"])
        k = int(p.get("kernel_size", 3))
        s = int(p.get("stride", 1))
        pad = p.get("padding", 0)
        if isinstance(pad, int) and pad == 0:
            pad_str = "'valid'"
        elif isinstance(pad, str):
            pad_str = f"'{pad}'"
        else:
            pad_str = "'valid'"
        parts = [str(filters), f"kernel_size={k}"]
        if s != 1:
            parts.append(f"strides={s}")
        if pad_str != "'valid'":
            parts.append(f"padding={pad_str}")
        return f"        layers.Conv2D({', '.join(parts)}),"

    if t == "MaxPool2D":
        k = int(p["kernel_size"])
        s = int(p.get("stride", k))
        if s != k:
            return f"        layers.MaxPool2D(pool_size={k}, strides={s}),"
        return f"        layers.MaxPool2D(pool_size={k}),"

    if t == "Flatten":
        return "        layers.Flatten(),"

    if t == "Dropout":
        rate = p.get("p", 0.5)
        return f"        layers.Dropout({rate}),"

    if t == "BatchNorm":
        return "        layers.BatchNormalization(),"

    if t == "ReLU":
        return "        layers.Activation('relu'),"

    if t == "Sigmoid":
        return "        layers.Activation('sigmoid'),"

    if t == "Softmax":
        return "        layers.Softmax(),"

    raise ValueError(f"Unsupported layer type for Keras generation: '{t}'")


def _assemble_file(class_name: str, layer_lines: list[str]) -> str:
    """Assemble the final Python source file."""
    layers_body = "\n".join(layer_lines) if layer_lines else "        # No layers"

    return f'''import tensorflow as tf
from tensorflow.keras import layers


def create_{class_name.lower()}() -> tf.keras.Model:
    """Auto-generated Keras model."""
    model = tf.keras.Sequential([
{layers_body}
    ], name="{class_name}")
    return model


# Build the model
{class_name} = create_{class_name.lower()}()
'''
