"""API routes for predictions."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.api.deps import DBSession, ModelDep
from app.crud import prediction_crud
from app.models.ml_model import ModelStatus
from app.schemas.prediction import (
    PredictionCreate,
    PredictionListResponse,
    PredictionResponse,
)
from app.services.onnx_service import ONNXInferenceError, onnx_service

logger = logging.getLogger(__name__)

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
    request: Request,
) -> PredictionResponse:
    """
    Run synchronous inference on a model.

    The model must be in READY status with a valid file path.
    Input data must match the model's input schema.
    """
    # Validate model is ready for inference
    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model is not ready for inference (status: {model.status.value})",
        )

    if not model.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no file path - please upload the model first",
        )

    # Run inference
    try:
        output_data, inference_time_ms = onnx_service.predict(
            model_id=model.id,
            input_data=prediction_in.input_data,
            file_path=model.file_path,
        )
    except ONNXInferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference failed: {e}",
        )

    # Get client IP for logging
    client_ip = request.client.host if request.client else None

    # Store prediction record
    prediction = await prediction_crud.create_with_model(
        db,
        obj_in=prediction_in,
        model_id=model.id,
    )

    # Update prediction with output data and timing
    prediction.output_data = output_data
    prediction.inference_time_ms = inference_time_ms
    prediction.cached = False  # Will be True when Redis caching is implemented
    prediction.client_ip = client_ip
    await db.commit()
    await db.refresh(prediction)

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
