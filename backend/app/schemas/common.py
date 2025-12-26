"""Common schemas used across the API."""

from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=_utc_now)
    database: str = "connected"
    redis: str = "connected"
    celery: str = "unknown"


class CeleryHealthResponse(BaseModel):
    """Celery worker health check response."""

    status: str  # "connected", "no_workers", "error"
    broker_connected: bool = False
    workers: dict = Field(default_factory=dict)  # worker_name -> stats
    queues: list[str] = Field(default_factory=list)
    error: str | None = None
    timestamp: datetime = Field(default_factory=_utc_now)


class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        total_pages = (total + page_size - 1) // page_size
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: str | None = None
    timestamp: datetime = Field(default_factory=_utc_now)
