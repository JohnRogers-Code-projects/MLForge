"""Pydantic schemas for ML model operations."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.ml_model import ModelStatus


class ModelBase(BaseModel):
    """Base schema for ML model."""

    name: str = Field(..., min_length=1, max_length=255, description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    version: str = Field(default="1.0.0", max_length=50, description="Model version")


class ModelCreate(ModelBase):
    """Schema for creating a new model."""

    model_metadata: Optional[dict[str, Any]] = Field(
        None,
        description="Additional metadata for the model",
    )

    model_config = {"protected_namespaces": ()}


class ModelUpdate(BaseModel):
    """Schema for updating a model."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    version: Optional[str] = Field(None, max_length=50)
    status: Optional[ModelStatus] = None
    model_metadata: Optional[dict[str, Any]] = None

    model_config = {"protected_namespaces": ()}


class ModelResponse(ModelBase):
    """Schema for model response."""

    id: str
    status: ModelStatus
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    input_schema: Optional[dict[str, Any]] = None
    output_schema: Optional[dict[str, Any]] = None
    model_metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class ModelListResponse(BaseModel):
    """Schema for listing models."""

    items: list[ModelResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ModelUploadResponse(BaseModel):
    """Schema for model file upload response."""

    id: str
    file_path: str
    file_size_bytes: int
    file_hash: str
    status: ModelStatus
    message: str = "File uploaded successfully"

    model_config = {"protected_namespaces": ()}
