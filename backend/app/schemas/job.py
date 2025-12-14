"""Pydantic schemas for job operations."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.job import JobPriority, JobStatus


class JobCreate(BaseModel):
    """Schema for creating a new job."""

    model_id: str = Field(..., description="ID of the model to use for inference")
    input_data: dict[str, Any] = Field(..., description="Input data for inference")
    priority: JobPriority = Field(
        default=JobPriority.NORMAL,
        description="Job priority",
    )


class JobStatusUpdate(BaseModel):
    """Schema for updating job status."""

    status: JobStatus
    error_message: Optional[str] = None


class JobResponse(BaseModel):
    """Schema for job response."""

    id: str
    model_id: str
    status: JobStatus
    priority: JobPriority
    input_data: dict[str, Any]
    output_data: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    inference_time_ms: Optional[float] = None
    queue_time_ms: Optional[float] = None
    retries: int = 0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    """Schema for listing jobs."""

    items: list[JobResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
