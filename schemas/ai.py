from typing import List, Tuple

from pydantic import BaseModel


class GraphNodeSchema(BaseModel):
    id: int
    label: str
    pos: Tuple[float, float]
    neighbors: List[int]


class GraphPredictionResponse(BaseModel):
    graph: List[GraphNodeSchema]
