"""CRUD operations for predictions."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.prediction import Prediction
from app.schemas.prediction import PredictionCreate


class CRUDPrediction(CRUDBase[Prediction, PredictionCreate, PredictionCreate]):
    """CRUD operations for Prediction."""

    async def get_by_model(
        self,
        db: AsyncSession,
        *,
        model_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Prediction]:
        """Get predictions for a specific model."""
        result = await db.execute(
            select(Prediction)
            .where(Prediction.model_id == model_id)
            .offset(offset)
            .limit(limit)
            .order_by(Prediction.created_at.desc())
        )
        return list(result.scalars().all())

    async def count_by_model(
        self,
        db: AsyncSession,
        *,
        model_id: str,
    ) -> int:
        """Count predictions for a specific model."""
        result = await db.execute(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.model_id == model_id)
        )
        return result.scalar() or 0

    async def create_with_model(
        self,
        db: AsyncSession,
        *,
        obj_in: PredictionCreate,
        model_id: str,
    ) -> Prediction:
        """Create a prediction for a specific model."""
        db_obj = Prediction(
            model_id=model_id,
            input_data=obj_in.input_data,
            request_id=obj_in.request_id,
        )
        db.add(db_obj)
        await db.flush()
        await db.refresh(db_obj)
        return db_obj


prediction_crud = CRUDPrediction(Prediction)
