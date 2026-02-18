"""Intermediate Representation (IR) data models.

The IR is the central data structure that all parsers produce
and all generators consume. It is a directed graph of typed nodes.
"""

from typing import Any

from pydantic import BaseModel, Field


# Maps each layer type to its primary parameter name (used for shorthand notation).
# Layers not listed here have no primary parameter.
PRIMARY_PARAM: dict[str, str] = {
    "Input": "shape",
    "Dense": "units",
    "Linear": "out_features",
    "Conv2D": "out_channels",
    "MaxPool2D": "kernel_size",
    "Dropout": "p",
    "BatchNorm": "num_features",
    "Softmax": "dim",
}

SUPPORTED_LAYER_TYPES: set[str] = {
    "Input",
    "Dense",
    "Linear",
    "Conv2D",
    "MaxPool2D",
    "Flatten",
    "Dropout",
    "BatchNorm",
    "ReLU",
    "Sigmoid",
    "Softmax",
}


class IRNode(BaseModel):
    """A single layer node in the computation graph."""

    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Layer type (e.g. Linear, Conv2D, ReLU)")
    params: dict[str, Any] = Field(default_factory=dict, description="Layer parameters")


class IREdge(BaseModel):
    """A directed edge connecting two nodes."""

    source: str = Field(..., alias="from", description="Source node id")
    target: str = Field(..., alias="to", description="Target node id")

    model_config = {"populate_by_name": True}


class IRGraph(BaseModel):
    """The complete intermediate representation of a neural network."""

    nodes: list[IRNode] = Field(default_factory=list)
    edges: list[IREdge] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical JSON dict with 'from'/'to' keys."""
        return {
            "nodes": [node.model_dump() for node in self.nodes],
            "edges": [edge.model_dump(by_alias=True) for edge in self.edges],
        }

    def node_ids(self) -> set[str]:
        """Return the set of all node ids."""
        return {node.id for node in self.nodes}
