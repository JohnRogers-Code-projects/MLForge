"""CRUD operations for predictions."""

from typing import Any

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
        """Create a prediction for a specific model (without inference)."""
        db_obj = Prediction(
            model_id=model_id,
            input_data=obj_in.input_data,
            request_id=obj_in.request_id,
        )
        db.add(db_obj)
        await db.flush()
        await db.refresh(db_obj)
        return db_obj

    async def create_with_results(
        self,
        db: AsyncSession,
        *,
        model_id: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        inference_time_ms: float,
        request_id: str | None = None,
        client_ip: str | None = None,
        cached: bool = False,
    ) -> Prediction:
        """Create a prediction with inference results.

        Args:
            db: Database session
            model_id: ID of the model used for inference
            input_data: Input data sent to the model
            output_data: Output data from inference
            inference_time_ms: Time taken for inference in milliseconds
            request_id: Optional request tracking ID
            client_ip: Optional client IP address
            cached: Whether the result was served from cache

        Returns:
            Created Prediction record
        """
        db_obj = Prediction(
            model_id=model_id,
            input_data=input_data,
            output_data=output_data,
            inference_time_ms=inference_time_ms,
            request_id=request_id,
            client_ip=client_ip,
            cached=cached,
        )
        db.add(db_obj)
        await db.flush()
        await db.refresh(db_obj)
        return db_obj


prediction_crud = CRUDPrediction(Prediction)
