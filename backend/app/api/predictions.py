"""API routes for predictions.

=============================================================================
POST-BOUNDARY CODE
=============================================================================

This module operates AFTER the pipeline commitment boundary.

All code here assumes the model has been validated and is in READY status.
The following invariants are assumed to hold:
- model.file_path points to a valid ONNX file
- model.input_schema and model.output_schema are authoritative
- The ONNX file is loadable by ONNX Runtime

If these invariants do not hold, the system is in a corrupt state.
This module does NOT tolerate pre-boundary models.
"""

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.api.deps import CacheDep, DBSession, ModelDep, ONNXDep, StorageDep
from app.crud import prediction_crud
from app.schemas.prediction import (
    PredictionCreate,
    PredictionListResponse,
    PredictionResponse,
)
from app.services.onnx import (
    ONNXInferenceError,
    ONNXInputError,
    ONNXLoadError,
    PostCommitmentInvariantViolation,
)
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

    This is POST-BOUNDARY code. It requires a committed model.

    Decision Authority
    ------------------
    All decisions about WHETHER to proceed are made in this function.
    Execution code (ONNXService) makes no policy decisions.

    To add policy (retries, fallbacks, confidence thresholds), you must
    modify this function. That is intentional. Policy changes should be
    visible in orchestration code, not hidden in execution code.
    """
    # =========================================================================
    # PHASE 1: DECISIONS
    # All decisions are made here. Each decision is named and explicit.
    # Adding policy requires adding a decision here.
    # =========================================================================

    # DECISION 1: Is the model committed?
    # Authority: Pipeline commitment boundary
    # If NO: Reject with 400
    try:
        model.assert_committed()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # DECISION 2: Does the model have a file path?
    # Authority: Post-commitment invariant
    # If NO: This is a contract violation, not a user error
    if not model.file_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="POST-COMMITMENT INVARIANT VIOLATED. "
            "Invariant: committed model has file_path set. "
            "Observed: file_path is None. "
            "The pipeline contract is broken. Execution cannot continue.",
        )

    # DECISION 3: Should we use a cached result?
    # Authority: Caller (via skip_cache parameter)
    # If YES: Return cached result, skip inference
    prediction_cache = PredictionCache(cache)
    use_cached_result = False
    cached_output = None
    cached_time = None

    if not prediction_in.skip_cache:
        cache_result = await prediction_cache.get_prediction(
            model.id, prediction_in.input_data
        )
        if cache_result.hit:
            use_cached_result = True
            cached_output = cache_result.output_data
            cached_time = cache_result.inference_time_ms

    # DECISION 4: Should we invoke inference?
    # Authority: This function (based on cache decision above)
    # This is the decision to invoke. It is explicit.
    should_invoke_inference = not use_cached_result

    # =========================================================================
    # PHASE 2: EXECUTION
    # Decisions have been made. Now execute based on those decisions.
    # Execution code does not make policy decisions.
    # =========================================================================

    if use_cached_result:
        # EXECUTE: Use cached result
        output_data = cached_output
        inference_time_ms = cached_time
        response.headers["X-Cache"] = "HIT"

    if should_invoke_inference:
        # EXECUTE: Run inference
        # This block contains NO decisions. Only execution.

        # Get file path
        try:
            file_path = await storage.get_path(model.file_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to access model file: {e}",
            ) from e

        # Invoke ONNXService (pure execution, no policy)
        try:
            result = onnx_service.run_inference(file_path, prediction_in.input_data)
            output_data = result.outputs
            inference_time_ms = result.inference_time_ms
        except PostCommitmentInvariantViolation:
            # Contract violation - not handled, execution stops
            raise
        except ONNXInputError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        except ONNXLoadError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {e}",
            ) from e
        except ONNXInferenceError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed: {e}",
            ) from e

        # Store in cache for future requests
        await prediction_cache.set_prediction(
            model.id,
            prediction_in.input_data,
            output_data,
            inference_time_ms,
        )
        response.headers["X-Cache"] = "MISS"

    # =========================================================================
    # PHASE 3: RECORD
    # Execution complete. Record the result.
    # =========================================================================

    if output_data is None or inference_time_ms is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference did not produce results.",
        )

    client_ip = request.client.host if request.client else None

    prediction = await prediction_crud.create_with_results(
        db,
        model_id=model.id,
        input_data=prediction_in.input_data,
        output_data=output_data,
        inference_time_ms=inference_time_ms,
        request_id=prediction_in.request_id,
        client_ip=client_ip,
        cached=use_cached_result,
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
