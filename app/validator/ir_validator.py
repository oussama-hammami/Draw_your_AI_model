"""IR graph validation engine.

Runs three validation passes on an IRGraph:
1. Structural  — duplicate node IDs, dangling edge references.
2. Topological — cycle detection (DAG enforcement), reachability.
3. Semantic    — layer parameter correctness.

All issues are collected before raising, so the caller gets a full
list of problems in a single exception.
"""

import logging
from collections import defaultdict
from typing import Any

from app.ir.models import IRGraph, SUPPORTED_LAYER_TYPES
from app.validator.exceptions import (
    CycleDetectedError,
    DuplicateNodeError,
    InvalidEdgeError,
    InvalidParameterError,
    UnreachableNodeError,
    ValidationError,
)

logger = logging.getLogger(__name__)


# ─── Layer parameter schemas ─────────────────────────────────────
# Each entry: {param_name: (allowed_types, required)}

_PARAM_SCHEMA: dict[str, dict[str, tuple[tuple[type, ...], bool]]] = {
    "Input": {
        "shape": ((int,), True),
    },
    "Dense": {
        "units": ((int,), True),
    },
    "Linear": {
        "out_features": ((int,), True),
    },
    "Conv2D": {
        "out_channels": ((int,), True),
        "kernel_size": ((int,), False),
        "stride": ((int,), False),
        "padding": ((int, str), False),
    },
    "MaxPool2D": {
        "kernel_size": ((int,), True),
        "stride": ((int,), False),
    },
    "Flatten": {},
    "Dropout": {
        "p": ((int, float), False),
    },
    "BatchNorm": {
        "num_features": ((int,), True),
    },
    "ReLU": {},
    "Sigmoid": {},
    "Softmax": {
        "dim": ((int,), False),
    },
}


def validate(graph: IRGraph, allow_cycles: bool = False) -> None:
    """Run all validation passes on the given IR graph.

    Args:
        graph: The IRGraph to validate.
        allow_cycles: If True, skip cycle detection.

    Raises:
        ValidationError (or subclass): With a list of all issues found.
    """
    errors: list[str] = []

    node_ids = _validate_structural(graph, errors)
    if node_ids is not None:
        _validate_topology(graph, node_ids, errors, allow_cycles)
    _validate_semantics(graph, errors)

    if errors:
        # Pick the most specific exception type based on what was found
        raise _pick_exception_type(errors)(errors)

    logger.info("IR validation passed: %d nodes, %d edges", len(graph.nodes), len(graph.edges))


# ─── Pass 1: Structural ──────────────────────────────────────────

def _validate_structural(graph: IRGraph, errors: list[str]) -> set[str] | None:
    """Check for duplicate IDs and dangling edge references.

    Returns the set of valid node IDs, or None if there are duplicate IDs.
    """
    # Duplicate IDs
    seen: dict[str, int] = {}
    for node in graph.nodes:
        seen[node.id] = seen.get(node.id, 0) + 1
    duplicates = [nid for nid, count in seen.items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate node IDs: {duplicates}")
        return None

    node_ids = {node.id for node in graph.nodes}

    # Dangling edge references
    for edge in graph.edges:
        if edge.source not in node_ids:
            errors.append(f"Edge references non-existent source node '{edge.source}'")
        if edge.target not in node_ids:
            errors.append(f"Edge references non-existent target node '{edge.target}'")

    return node_ids


# ─── Pass 2: Topology ────────────────────────────────────────────

def _validate_topology(
    graph: IRGraph,
    node_ids: set[str],
    errors: list[str],
    allow_cycles: bool,
) -> None:
    """Check for cycles and unreachable nodes."""
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}

    for edge in graph.edges:
        if edge.source in node_ids and edge.target in node_ids:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

    # Cycle detection via Kahn's algorithm (topological sort)
    if not allow_cycles:
        _check_cycles(adj, in_degree, node_ids, errors)

    # Reachability: every node must be reachable from a root (in_degree == 0)
    roots = {nid for nid, deg in in_degree.items() if deg == 0}
    if not roots and node_ids:
        # All nodes have incoming edges — either a cycle or all are mid-chain
        errors.append("No root nodes found (every node has incoming edges)")
        return

    reachable: set[str] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(adj.get(current, []))

    unreachable = node_ids - reachable
    if unreachable:
        errors.append(f"Unreachable nodes (not connected to any root): {sorted(unreachable)}")


def _check_cycles(
    adj: dict[str, list[str]],
    in_degree: dict[str, int],
    node_ids: set[str],
    errors: list[str],
) -> None:
    """Detect cycles using Kahn's topological sort algorithm."""
    in_deg = dict(in_degree)  # copy
    queue = [nid for nid, deg in in_deg.items() if deg == 0]
    visited_count = 0

    while queue:
        node = queue.pop(0)
        visited_count += 1
        for neighbor in adj.get(node, []):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    if visited_count != len(node_ids):
        cycle_nodes = sorted(nid for nid, deg in in_deg.items() if deg > 0)
        errors.append(f"Cycle detected involving nodes: {cycle_nodes}")


# ─── Pass 3: Semantics ───────────────────────────────────────────

def _validate_semantics(graph: IRGraph, errors: list[str]) -> None:
    """Validate each node's type and parameters against the schema."""
    for node in graph.nodes:
        if node.type not in SUPPORTED_LAYER_TYPES:
            errors.append(f"Node '{node.id}': unsupported layer type '{node.type}'")
            continue

        schema = _PARAM_SCHEMA.get(node.type, {})
        _check_required_params(node.id, node.type, node.params, schema, errors)
        _check_unknown_params(node.id, node.type, node.params, schema, errors)
        _check_param_types(node.id, node.params, schema, errors)


def _check_required_params(
    node_id: str,
    layer_type: str,
    params: dict[str, Any],
    schema: dict[str, tuple[tuple[type, ...], bool]],
    errors: list[str],
) -> None:
    """Ensure all required parameters are present."""
    for param_name, (_, required) in schema.items():
        if required and param_name not in params:
            errors.append(
                f"Node '{node_id}' ({layer_type}): missing required parameter '{param_name}'"
            )


def _check_unknown_params(
    node_id: str,
    layer_type: str,
    params: dict[str, Any],
    schema: dict[str, tuple[tuple[type, ...], bool]],
    errors: list[str],
) -> None:
    """Warn about parameters not defined in the schema."""
    for param_name in params:
        if param_name not in schema:
            errors.append(
                f"Node '{node_id}' ({layer_type}): unknown parameter '{param_name}'"
            )


def _check_param_types(
    node_id: str,
    params: dict[str, Any],
    schema: dict[str, tuple[tuple[type, ...], bool]],
    errors: list[str],
) -> None:
    """Check that parameter values match expected types."""
    for param_name, value in params.items():
        if param_name not in schema:
            continue
        allowed_types, _ = schema[param_name]
        if not isinstance(value, allowed_types):
            errors.append(
                f"Node '{node_id}': parameter '{param_name}' must be "
                f"{' or '.join(t.__name__ for t in allowed_types)}, got {type(value).__name__}"
            )


# ─── Helper ──────────────────────────────────────────────────────

def _pick_exception_type(errors: list[str]) -> type[ValidationError]:
    """Pick the most specific exception subclass from the error messages."""
    text = " ".join(errors).lower()
    if "cycle" in text:
        return CycleDetectedError
    if "unreachable" in text or "no root" in text:
        return UnreachableNodeError
    if "non-existent" in text:
        return InvalidEdgeError
    if "duplicate" in text:
        return DuplicateNodeError
    if "parameter" in text or "missing required" in text:
        return InvalidParameterError
    return ValidationError
