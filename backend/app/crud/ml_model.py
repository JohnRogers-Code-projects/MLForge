"""CRUD operations for ML models."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.ml_model import MLModel, ModelStatus
from app.schemas.ml_model import ModelCreate, ModelUpdate


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


model_crud = CRUDModel(MLModel)
