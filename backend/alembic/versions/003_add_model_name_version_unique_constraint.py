"""Add unique constraint on model name and version

Revision ID: 003
Revises: 002
Create Date: 2025-12-16

This migration adds a unique constraint on (name, version) combination in the ml_models table.
This ensures that each model version is unique - you cannot have two models with the same
name and version.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_model_name_version",
        "ml_models",
        ["name", "version"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_model_name_version",
        "ml_models",
        type_="unique",
    )
