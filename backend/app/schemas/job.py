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

    model_config = {"protected_namespaces": ()}


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
    celery_task_id: Optional[str] = None
    worker_id: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    inference_time_ms: Optional[float] = None
    queue_time_ms: Optional[float] = None
    retries: int = 0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

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
    result: Optional[dict[str, Any]] = Field(
        None,
        description="Inference result if job completed successfully",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if job failed",
    )
    error_traceback: Optional[str] = Field(
        None,
        description="Full error traceback if job failed (for debugging)",
    )
    inference_time_ms: Optional[float] = Field(
        None,
        description="Time taken for inference in milliseconds",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when job completed",
    )
