"""Pydantic schemas for job operations."""

from datetime import datetime
from typing import Any

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

    model_config = {"protected_namespaces": ()}


class JobStatusUpdate(BaseModel):
    """Schema for updating job status."""

    status: JobStatus
    error_message: str | None = None


class JobResponse(BaseModel):
    """Schema for job response."""

    id: str
    model_id: str
    status: JobStatus
    priority: JobPriority
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None = None
    celery_task_id: str | None = None
    worker_id: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    inference_time_ms: float | None = None
    queue_time_ms: float | None = None
    retries: int = 0
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class JobListResponse(BaseModel):
    """Schema for listing jobs."""

    items: list[JobResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class JobResultResponse(BaseModel):
    """Schema for job result endpoint response.

    Returns the result of a completed job, or error details if failed.
    """

    job_id: str
    status: JobStatus
    result: dict[str, Any] | None = Field(
        None,
        description="Inference result if job completed successfully",
    )
    error_message: str | None = Field(
        None,
        description="Error message if job failed",
    )
    error_traceback: str | None = Field(
        None,
        description="Full error traceback if job failed (for debugging)",
    )
    inference_time_ms: float | None = Field(
        None,
        description="Time taken for inference in milliseconds",
    )
    completed_at: datetime | None = Field(
        None,
        description="Timestamp when job completed",
    )
