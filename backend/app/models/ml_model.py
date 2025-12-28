"""MLModel database model for storing model metadata."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    Enum,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.job import Job
    from app.models.prediction import Prediction


class ModelStatus(str, enum.Enum):
    """Status of an ML model."""

    PENDING = "pending"
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class MLModel(Base):
    """Represents an ML model in the system.

    Pipeline Commitment Boundary
    ============================

    The validate endpoint (/models/{model_id}/validate) is THE commitment point.
    Crossing this boundary is one-way. There is no uncommit.

    PRE-BOUNDARY (status: PENDING, UPLOADED, VALIDATING, ERROR)
    -----------------------------------------------------------
    - Model is experimental
    - Metadata may change freely
    - File may be re-uploaded
    - No downstream operations depend on this model
    - No inference allowed

    POST-BOUNDARY (status: READY)
    -----------------------------
    The following invariants are assumed to hold. Code may rely on them.
    Violations should be treated as corruption, not expected failures.

    Invariants:
    - file_path is set and points to a valid ONNX file
    - file_hash is set and matches the file content
    - input_schema is set and describes model inputs
    - output_schema is set and describes model outputs
    - model_metadata contains ONNX runtime metadata
    - The file can be loaded by ONNX Runtime without error

    Code that operates post-boundary (predictions, jobs) assumes these hold.
    If they do not hold, the system is in a corrupt state.

    This is NOT a global canonical context. Each model commits independently.
    """

    __tablename__ = "ml_models"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
    )

    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus),
        nullable=False,
        default=ModelStatus.PENDING,
    )

    # Model file information
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ONNX metadata (populated after validation)
    input_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Note: Named 'model_metadata' to avoid conflict with SQLAlchemy's reserved 'metadata' attribute
    model_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

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

    # =========================================================================
    # PIPELINE COMMITMENT BOUNDARY
    # =========================================================================
    # The methods below relate to the commitment boundary.
    # Post-boundary code should call assert_committed() as its first operation.
    # This is documentation-as-code: making the boundary crossing explicit.

    def is_committed(self) -> bool:
        """Return True if this model has crossed the pipeline commitment boundary.

        A committed model (status=READY) has locked-in invariants.
        See class docstring for what invariants are assumed to hold.
        """
        return self.status == ModelStatus.READY

    def assert_committed(self) -> None:
        """Assert that this model has crossed the commitment boundary.

        Call this at the start of any post-boundary operation.
        If this fails, the caller is attempting to operate on an uncommitted model.

        This is the only runtime check for boundary enforcement.
        It makes the boundary explicit rather than relying on implicit status checks.
        """
        if not self.is_committed():
            raise ValueError(
                f"Model {self.id} has not crossed the pipeline commitment boundary. "
                f"Current status: {self.status.value}. "
                f"The model must be validated before inference operations."
            )
