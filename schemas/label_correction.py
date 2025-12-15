from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import date


class NodeCorrection(BaseModel):
    """Single node label correction."""

    node_id: int
    predicted_label: str
    corrected_label: str


class LabelCorrectionRequest(BaseModel):
    """Request to save label corrections for an image."""

    image_path: str  # e.g., graph_images/4/graph_1702643200.jpg
    corrections: Dict[int, str]  # {node_id: corrected_label}
    predicted_labels: Dict[int, str]  # {node_id: predicted_label}
    data_structure_type: str = "graph_nodes"


class LabelCorrectionResponse(BaseModel):
    """Response with label correction details."""

    image_path: str
    data_structure_type: str
    wrong_label: Optional[Dict[str, Any]]  # predicted labels
    correct_label: Dict[str, Any]  # user corrections
    created_at: date
    user_id: int

    class Config:
        from_attributes = True
