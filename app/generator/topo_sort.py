"""Topological sort for IR graphs using Kahn's algorithm."""

from collections import defaultdict

from app.ir.models import IRGraph


def topological_sort(graph: IRGraph) -> list[str]:
    """Return node IDs in topological order.

    Args:
        graph: A validated (acyclic) IRGraph.

    Returns:
        List of node IDs sorted so that every node appears after its predecessors.

    Raises:
        ValueError: If the graph contains a cycle (should be caught by validator first).
    """
    node_ids = {n.id for n in graph.nodes}
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}

    for edge in graph.edges:
        adj[edge.source].append(edge.target)
        in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

    queue = [nid for nid in in_degree if in_degree[nid] == 0]
    # Sort the initial queue for deterministic output order
    queue.sort()
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in sorted(adj.get(node, [])):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(node_ids):
        raise ValueError("Graph contains a cycle — cannot topologically sort")

    return result
