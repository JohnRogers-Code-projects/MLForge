"""API routes for predictions."""

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.api.deps import CacheDep, DBSession, ModelDep, ONNXDep, StorageDep
from app.crud import prediction_crud
from app.models.ml_model import ModelStatus
from app.schemas.prediction import (
    PredictionCreate,
    PredictionListResponse,
    PredictionResponse,
)
from app.services.onnx import ONNXInferenceError, ONNXInputError, ONNXLoadError
from app.services.prediction_cache import PredictionCache

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
    storage: StorageDep,
    onnx_service: ONNXDep,
    cache: CacheDep,
    request: Request,
    response: Response,
) -> PredictionResponse:
    """Run synchronous inference on a model.

    Validates the model is ready, runs inference using ONNX Runtime,
    and stores the prediction result in the database.

    Supports prediction caching: if the same input was recently predicted,
    returns the cached result. Use skip_cache=true to bypass the cache.
    """
    # Check model status
    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model is not ready for inference. Current status: {model.status.value}. "
            f"Please validate the model first.",
        )

    # Check model has a file
    if not model.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model does not have an uploaded file.",
        )

    # Initialize prediction cache
    prediction_cache = PredictionCache(cache)
    cached = False
    output_data = None
    inference_time_ms = None

    # Check cache (unless skip_cache is set)
    if not prediction_in.skip_cache:
        cache_result = await prediction_cache.get_prediction(
            model.id, prediction_in.input_data
        )
        if cache_result.hit:
            cached = True
            output_data = cache_result.output_data
            inference_time_ms = cache_result.inference_time_ms
            response.headers["X-Cache"] = "HIT"

    # Cache miss or skip_cache - run inference
    if not cached:
        # Get the file path from storage
        try:
            file_path = await storage.get_path(model.file_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to access model file: {e}",
            )

        # Run inference
        try:
            result = onnx_service.run_inference(file_path, prediction_in.input_data)
            output_data = result.outputs
            inference_time_ms = result.inference_time_ms
        except ONNXInputError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except ONNXLoadError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {e}",
            )
        except ONNXInferenceError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed: {e}",
            )

        # Cache the result for future requests
        await prediction_cache.set_prediction(
            model.id,
            prediction_in.input_data,
            output_data,
            inference_time_ms,
        )
        response.headers["X-Cache"] = "MISS"

    # Get client IP for logging
    client_ip = request.client.host if request.client else None

    # Create prediction record with results (always, even on cache hit for audit)
    prediction = await prediction_crud.create_with_results(
        db,
        model_id=model.id,
        input_data=prediction_in.input_data,
        output_data=output_data,
        inference_time_ms=inference_time_ms,
        request_id=prediction_in.request_id,
        client_ip=client_ip,
        cached=cached,
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
