"""Pydantic schemas for ML model operations."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.ml_model import ModelStatus


class ModelBase(BaseModel):
    """Base schema for ML model."""

    name: str = Field(..., min_length=1, max_length=255, description="Model name")
    description: str | None = Field(None, description="Model description")
    version: str = Field(default="1.0.0", max_length=50, description="Model version")


class ModelCreate(ModelBase):
    """Schema for creating a new model."""

    model_metadata: dict[str, Any] | None = Field(
        None,
        description="Additional metadata for the model",
    )

    model_config = {"protected_namespaces": ()}


class ModelUpdate(BaseModel):
    """Schema for updating a model."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    version: str | None = Field(None, max_length=50)
    status: ModelStatus | None = None
    model_metadata: dict[str, Any] | None = None

    model_config = {"protected_namespaces": ()}


class ModelResponse(ModelBase):
    """Schema for model response."""

    id: str
    status: ModelStatus
    file_path: str | None = None
    file_size_bytes: int | None = None
    input_schema: list[dict[str, Any]] | None = None
    output_schema: list[dict[str, Any]] | None = None
    model_metadata: dict[str, Any] | None = None
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


class TensorSchemaResponse(BaseModel):
    """Schema for input/output tensor information."""

    name: str
    dtype: str
    shape: list[int | None]


class ModelValidateResponse(BaseModel):
    """Schema for model validation response."""

    id: str
    valid: bool
    status: ModelStatus
    input_schema: list[TensorSchemaResponse] | None = None
    output_schema: list[TensorSchemaResponse] | None = None
    model_metadata: dict[str, Any] | None = None
    error_message: str | None = None
    message: str = "Model validated successfully"

    model_config = {"protected_namespaces": ()}


class ModelVersionSummary(BaseModel):
    """Summary of a model version."""

    id: str
    version: str
    status: ModelStatus
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class ModelVersionsResponse(BaseModel):
    """Response for listing model versions."""

    name: str
    versions: list[ModelVersionSummary]
    total: int
    latest_version: str | None = None
