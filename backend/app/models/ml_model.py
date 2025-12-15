"""MLModel database model for storing model metadata."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import DateTime, Enum, Integer, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.prediction import Prediction
    from app.models.job import Job


class ModelStatus(str, enum.Enum):
    """Status of an ML model."""

    PENDING = "pending"
    VALIDATING = "validating"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class MLModel(Base):
    """Represents an ML model in the system."""

    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus),
        nullable=False,
        default=ModelStatus.PENDING,
    )

    # Model file information
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # ONNX metadata (populated after validation)
    input_schema: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output_schema: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Note: Named 'model_metadata' to avoid conflict with SQLAlchemy's reserved 'metadata' attribute
    model_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction",
        back_populates="model",
        cascade="all, delete-orphan",
    )
    jobs: Mapped[list["Job"]] = relationship(
        "Job",
        back_populates="model",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<MLModel(id={self.id}, name={self.name}, version={self.version})>"
