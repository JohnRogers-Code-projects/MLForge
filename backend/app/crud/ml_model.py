"""CRUD operations for ML models."""

import re
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.ml_model import MLModel, ModelStatus
from app.schemas.ml_model import ModelCreate, ModelUpdate


def parse_semver(version: str) -> tuple[int, int, int, str]:
    """Parse a semantic version string into comparable tuple.

    Supports formats like: 1.0.0, 1.2.3, 1.0.0-beta, 2.1.0-rc.1
    Returns (major, minor, patch, prerelease) tuple.
    Non-semver strings are treated as (0, 0, 0, original_string).
    """
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$", version)
    if match:
        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        prerelease = match.group(4) or ""
        return (major, minor, patch, prerelease)
    # Non-semver: sort alphabetically after all semver versions
    return (0, 0, 0, version)


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings.

    Returns:
        -1 if v1 < v2
        0 if v1 == v2
        1 if v1 > v2
    """
    parsed1 = parse_semver(v1)
    parsed2 = parse_semver(v2)

    # Compare major, minor, patch
    for i in range(3):
        if parsed1[i] < parsed2[i]:
            return -1
        if parsed1[i] > parsed2[i]:
            return 1

    # If equal so far, compare prerelease
    # Empty prerelease (stable) > any prerelease
    pre1, pre2 = parsed1[3], parsed2[3]
    if pre1 == pre2:
        return 0
    if pre1 == "":
        return 1  # stable > prerelease
    if pre2 == "":
        return -1  # prerelease < stable
    # Both have prerelease, compare alphabetically
    if pre1 < pre2:
        return -1
    return 1


class CRUDModel(CRUDBase[MLModel, ModelCreate, ModelUpdate]):
    """CRUD operations for MLModel."""

    async def get_by_name(
        self,
        db: AsyncSession,
        *,
        name: str,
    ) -> Optional[MLModel]:
        """Get a model by name."""
        result = await db.execute(
            select(MLModel).where(MLModel.name == name)
        )
        return result.scalar_one_or_none()

    async def get_by_name_and_version(
        self,
        db: AsyncSession,
        *,
        name: str,
        version: str,
    ) -> Optional[MLModel]:
        """Get a model by name and version."""
        result = await db.execute(
            select(MLModel).where(
                MLModel.name == name,
                MLModel.version == version,
            )
        )
        return result.scalar_one_or_none()

    async def get_ready_models(
        self,
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[MLModel]:
        """Get all models that are ready for inference."""
        result = await db.execute(
            select(MLModel)
            .where(MLModel.status == ModelStatus.READY)
            .offset(offset)
            .limit(limit)
            .order_by(MLModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        db: AsyncSession,
        *,
        model_id: str,
        status: ModelStatus,
    ) -> Optional[MLModel]:
        """Update model status."""
        model = await self.get(db, model_id)
        if model:
            model.status = status
            await db.flush()
            await db.refresh(model)
        return model

    async def get_versions_by_name(
        self,
        db: AsyncSession,
        *,
        name: str,
    ) -> list[MLModel]:
        """Get all versions of a model by name, sorted by version descending.

        Uses semantic version comparison to sort versions correctly
        (e.g., 2.0.0 > 1.10.0 > 1.9.0).
        """
        result = await db.execute(
            select(MLModel)
            .where(MLModel.name == name)
        )
        models = list(result.scalars().all())
        # Sort by semantic version (newest first)
        models.sort(key=lambda m: parse_semver(m.version), reverse=True)
        return models

    async def get_latest_by_name(
        self,
        db: AsyncSession,
        *,
        name: str,
        ready_only: bool = False,
    ) -> Optional[MLModel]:
        """Get the latest version of a model by name.

        Args:
            db: Database session
            name: Model name to search for
            ready_only: If True, only consider models with READY status

        Returns:
            The model with the highest semantic version, or None if not found.
        """
        query = select(MLModel).where(MLModel.name == name)
        if ready_only:
            query = query.where(MLModel.status == ModelStatus.READY)

        result = await db.execute(query)
        models = list(result.scalars().all())

        if not models:
            return None

        # Find the model with the highest version
        return max(models, key=lambda m: parse_semver(m.version))

    async def count_versions_by_name(
        self,
        db: AsyncSession,
        *,
        name: str,
    ) -> int:
        """Count all versions of a model by name."""
        result = await db.execute(
            select(func.count())
            .select_from(MLModel)
            .where(MLModel.name == name)
        )
        return result.scalar() or 0

    async def get_unique_model_names(
        self,
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[str]:
        """Get list of unique model names."""
        result = await db.execute(
            select(MLModel.name)
            .distinct()
            .order_by(MLModel.name)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())


model_crud = CRUDModel(MLModel)
