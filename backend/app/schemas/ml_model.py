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


class InputSchemaItem(BaseModel):
    """Schema for a single model input."""

    name: str
    shape: list[int | str | None]
    dtype: str


class OutputSchemaItem(BaseModel):
    """Schema for a single model output."""

    name: str
    shape: list[int | str | None]
    dtype: str


class ModelCreate(ModelBase):
    """Schema for creating a new model."""

    metadata: Optional[dict[str, Any]] = Field(
        None,
        description="Additional metadata for the model",
    )


class ModelUpdate(BaseModel):
    """Schema for updating a model."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    version: Optional[str] = Field(None, max_length=50)
    status: Optional[ModelStatus] = None
    metadata: Optional[dict[str, Any]] = None


class ModelResponse(ModelBase):
    """Schema for model response."""

    id: str
    status: ModelStatus
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    input_schema: Optional[dict[str, Any]] = None
    output_schema: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelListResponse(BaseModel):
    """Schema for listing models."""

    items: list[ModelResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ModelUploadResponse(BaseModel):
    """Response after successful model upload and validation."""

    id: str
    name: str
    version: str
    status: ModelStatus
    file_path: str
    file_size_bytes: int
    file_hash: str
    input_schema: list[InputSchemaItem]
    output_schema: list[OutputSchemaItem]
    message: str = "Model uploaded and validated successfully"

    model_config = {"from_attributes": True}


class ModelValidationError(BaseModel):
    """Response when model validation fails."""

    error: str
    detail: Optional[str] = None
