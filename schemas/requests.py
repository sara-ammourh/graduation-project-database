from pydantic import BaseModel
from typing import List, Tuple


class ExpressionRequest(BaseModel):
    expression: str


class ClusterRequest(BaseModel):
    algorithm: str
    points: List[Tuple[float, float]]
    k: int = 1
    eps: float = 0.5
    minSamples: int = 5


class GraphNodeResponse(BaseModel):
    id: int
    label: str
    pos: List[float]
    neighbors: List[int]


class GraphResponse(BaseModel):
    nodes: List[GraphNodeResponse]
    node_count: int
    edge_count: int
