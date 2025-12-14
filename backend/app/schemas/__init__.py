"""Pydantic schemas for API validation."""

from app.schemas.ml_model import (
    ModelCreate,
    ModelUpdate,
    ModelResponse,
    ModelListResponse,
    ModelUploadResponse,
    ModelValidationError,
    InputSchemaItem,
    OutputSchemaItem,
)
from app.schemas.prediction import (
    PredictionCreate,
    PredictionResponse,
)
from app.schemas.job import (
    JobCreate,
    JobResponse,
    JobStatusUpdate,
)
from app.schemas.common import (
    HealthResponse,
    PaginationParams,
    PaginatedResponse,
)

__all__ = [
    "ModelCreate",
    "ModelUpdate",
    "ModelResponse",
    "ModelListResponse",
    "ModelUploadResponse",
    "ModelValidationError",
    "InputSchemaItem",
    "OutputSchemaItem",
    "PredictionCreate",
    "PredictionResponse",
    "JobCreate",
    "JobResponse",
    "JobStatusUpdate",
    "HealthResponse",
    "PaginationParams",
    "PaginatedResponse",
]
