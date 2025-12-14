"""API routes for predictions."""

from fastapi import APIRouter, Query, status

from app.api.deps import DBSession, ModelDep
from app.crud import prediction_crud
from app.schemas.prediction import (
    PredictionCreate,
    PredictionListResponse,
    PredictionResponse,
)

router = APIRouter()


@router.post(
    "/models/{model_id}/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_prediction(
    model: ModelDep,
    prediction_in: PredictionCreate,
    db: DBSession,
) -> PredictionResponse:
    """
    Create a synchronous prediction.

    Note: Actual inference will be implemented in Phase 2.
    This endpoint currently stores the prediction request.
    """
    prediction = await prediction_crud.create_with_model(
        db,
        obj_in=prediction_in,
        model_id=model.id,
    )
    return PredictionResponse.model_validate(prediction)


@router.get("/models/{model_id}/predictions", response_model=PredictionListResponse)
async def list_predictions(
    model: ModelDep,
    db: DBSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> PredictionListResponse:
    """List predictions for a specific model."""
    offset = (page - 1) * page_size
    predictions = await prediction_crud.get_by_model(
        db,
        model_id=model.id,
        offset=offset,
        limit=page_size,
    )
    total = await prediction_crud.count_by_model(db, model_id=model.id)
    total_pages = (total + page_size - 1) // page_size

    return PredictionListResponse(
        items=[PredictionResponse.model_validate(p) for p in predictions],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
