"""Pydantic schemas for prediction operations."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PredictionCreate(BaseModel):
    """Schema for creating a new prediction."""

    input_data: dict[str, Any] = Field(..., description="Input data for inference")
    request_id: Optional[str] = Field(None, description="Optional request tracking ID")


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    id: str
    model_id: str
    input_data: dict[str, Any]
    output_data: Optional[dict[str, Any]] = None
    inference_time_ms: Optional[float] = None
    cached: bool = False
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class PredictionListResponse(BaseModel):
    """Schema for listing predictions."""

    items: list[PredictionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
