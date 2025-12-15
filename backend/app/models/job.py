"""Job database model for async inference tasks."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.ml_model import MLModel


class JobStatus(str, enum.Enum):
    """Status of an async job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, enum.Enum):
    """Priority levels for jobs."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Job(Base):
    """Represents an async inference job."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    model_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Job configuration
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
    )
    priority: Mapped[JobPriority] = mapped_column(
        Enum(JobPriority),
        nullable=False,
        default=JobPriority.NORMAL,
    )

    # Input/Output
    input_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    output_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Execution details
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    worker_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Performance
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    queue_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    model: Mapped["MLModel"] = relationship("MLModel", back_populates="jobs")

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, status={self.status})>"
