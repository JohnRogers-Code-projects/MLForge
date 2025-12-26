"""Rename metadata column to model_metadata

Revision ID: 002
Revises: 001
Create Date: 2025-12-15

This migration renames the 'metadata' column to 'model_metadata' in the ml_models table.
The rename is necessary because 'metadata' is a reserved attribute name in SQLAlchemy's
declarative base class, causing conflicts when accessing the column.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "ml_models",
        "metadata",
        new_column_name="model_metadata",
    )


def downgrade() -> None:
    op.alter_column(
        "ml_models",
        "model_metadata",
        new_column_name="metadata",
    )
