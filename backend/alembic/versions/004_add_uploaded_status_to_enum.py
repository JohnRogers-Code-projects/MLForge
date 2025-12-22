"""Add UPLOADED status to modelstatus enum

Revision ID: 004
Revises: 003
Create Date: 2025-12-22
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add UPLOADED to the modelstatus enum
    # PostgreSQL requires ALTER TYPE to add new enum values
    op.execute("ALTER TYPE modelstatus ADD VALUE IF NOT EXISTS 'UPLOADED'")


def downgrade() -> None:
    # PostgreSQL doesn't support removing enum values directly
    # Would need to recreate the type, which is complex
    # For now, leave the value in place on downgrade
    pass
# Force rebuild 22 Dec 2025 22:10:10
