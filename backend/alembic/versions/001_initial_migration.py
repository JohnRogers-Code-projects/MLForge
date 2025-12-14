"""Initial migration - create all tables

Revision ID: 001
Revises:
Create Date: 2025-12-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create ml_models table
    op.create_table(
        "ml_models",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column(
            "status",
            sa.Enum("PENDING", "VALIDATING", "READY", "ERROR", "ARCHIVED", name="modelstatus"),
            nullable=False,
        ),
        sa.Column("file_path", sa.String(500), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("file_hash", sa.String(64), nullable=True),
        sa.Column("input_schema", postgresql.JSON(), nullable=True),
        sa.Column("output_schema", postgresql.JSON(), nullable=True),
        sa.Column("metadata", postgresql.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_ml_models_name"), "ml_models", ["name"], unique=False)

    # Create predictions table
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("model_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("input_data", postgresql.JSON(), nullable=False),
        sa.Column("output_data", postgresql.JSON(), nullable=True),
        sa.Column("inference_time_ms", sa.Float(), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=False, default=False),
        sa.Column("request_id", sa.String(100), nullable=True),
        sa.Column("client_ip", sa.String(45), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["model_id"], ["ml_models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_predictions_model_id"), "predictions", ["model_id"], unique=False)

    # Create jobs table
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("model_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED",
                name="jobstatus"
            ),
            nullable=False,
        ),
        sa.Column(
            "priority",
            sa.Enum("LOW", "NORMAL", "HIGH", name="jobpriority"),
            nullable=False,
        ),
        sa.Column("input_data", postgresql.JSON(), nullable=False),
        sa.Column("output_data", postgresql.JSON(), nullable=True),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("worker_id", sa.String(255), nullable=True),
        sa.Column("retries", sa.Integer(), nullable=False, default=0),
        sa.Column("max_retries", sa.Integer(), nullable=False, default=3),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_traceback", sa.Text(), nullable=True),
        sa.Column("inference_time_ms", sa.Float(), nullable=True),
        sa.Column("queue_time_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["ml_models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_model_id"), "jobs", ["model_id"], unique=False)
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_jobs_status"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_model_id"), table_name="jobs")
    op.drop_table("jobs")
    op.drop_index(op.f("ix_predictions_model_id"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_index(op.f("ix_ml_models_name"), table_name="ml_models")
    op.drop_table("ml_models")
    op.execute("DROP TYPE IF EXISTS jobpriority")
    op.execute("DROP TYPE IF EXISTS jobstatus")
    op.execute("DROP TYPE IF EXISTS modelstatus")
