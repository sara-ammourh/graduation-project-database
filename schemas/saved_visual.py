from pydantic import BaseModel
from typing import Dict, Any
from datetime import date


class SavedVisualRequest(BaseModel):
    """Request model for creating/updating a saved visual."""

    saved_visual: Dict[str, Any]
    type: str  # e.g., "graph", "plot", "visualization"


class SavedVisualResponse(BaseModel):
    """Response model for a saved visual."""

    id: int
    saved_visual: Dict[str, Any]
    type: str
    updated_at: date
    user_id: int

    class Config:
        from_attributes = True
