"""API routes for ML model management."""

from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, status

from app.api.deps import CacheDep, DBSession, ModelDep, ONNXDep, StorageDep
from app.config import settings
from app.crud import model_crud
from app.models.ml_model import ModelStatus
from app.schemas.ml_model import (
    ModelCreate,
    ModelListResponse,
    ModelResponse,
    ModelUpdate,
    ModelUploadResponse,
    ModelValidateResponse,
    ModelVersionsResponse,
    ModelVersionSummary,
    TensorSchemaResponse,
)
from app.services.model_cache import ModelCache, model_to_cache_dict
from app.services.prediction_cache import PredictionCache
from app.services.storage import StorageError, StorageFullError

router = APIRouter()


@router.post(
    "",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_model(
    model_in: ModelCreate,
    db: DBSession,
) -> ModelResponse:
    """Create a new ML model."""
    existing = await model_crud.get_by_name_and_version(
        db,
        name=model_in.name,
        version=model_in.version,
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_in.name}' version '{model_in.version}' already exists",
        )

    model = await model_crud.create(db, obj_in=model_in)
    return ModelResponse.model_validate(model)


@router.get("", response_model=ModelListResponse)
async def list_models(
    db: DBSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> ModelListResponse:
    """List all ML models with pagination."""
    offset = (page - 1) * page_size
    models = await model_crud.get_multi(db, offset=offset, limit=page_size)
    total = await model_crud.count(db)
    total_pages = (total + page_size - 1) // page_size

    return ModelListResponse(
        items=[ModelResponse.model_validate(m) for m in models],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/by-name/{name}/versions", response_model=ModelVersionsResponse)
async def list_model_versions(
    name: str,
    db: DBSession,
) -> ModelVersionsResponse:
    """List all versions of a model by name.

    Returns all versions sorted by semantic version (highest first).
    """
    versions = await model_crud.get_versions_by_name(db, name=name)

    if not versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model found with name '{name}'",
        )

    # The latest version is the first in the sorted list
    latest_version = versions[0].version if versions else None

    return ModelVersionsResponse(
        name=name,
        versions=[
            ModelVersionSummary(
                id=v.id,
                version=v.version,
                status=v.status,
                created_at=v.created_at,
            )
            for v in versions
        ],
        total=len(versions),
        latest_version=latest_version,
    )


@router.get("/by-name/{name}/latest", response_model=ModelResponse)
async def get_latest_model_version(
    name: str,
    db: DBSession,
    ready_only: bool = Query(
        False,
        description="If true, only return the latest READY version",
    ),
) -> ModelResponse:
    """Get the latest version of a model by name.

    Uses semantic versioning to determine the latest version.
    Optionally filter to only return READY models.
    """
    model = await model_crud.get_latest_by_name(db, name=name, ready_only=ready_only)

    if not model:
        if ready_only:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No ready model found with name '{name}'",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model found with name '{name}'",
        )

    return ModelResponse.model_validate(model)


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    response: Response,
    db: DBSession,
    cache: CacheDep,
) -> ModelResponse:
    """Get a specific ML model by ID.

    Results are cached for improved performance. Cache is automatically
    invalidated when the model is updated or deleted.
    """
    model_cache = ModelCache(cache)

    # Try cache first
    cached = await model_cache.get_model(model_id)
    if cached:
        # Add cache headers indicating a cache hit
        response.headers["X-Cache"] = "HIT"
        response.headers["Cache-Control"] = f"max-age={settings.cache_model_ttl}"
        return ModelResponse.model_validate(cached)

    # Cache miss - fetch from database
    model = await model_crud.get(db, model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found",
        )

    # Cache the result
    cache_data = model_to_cache_dict(model)
    await model_cache.set_model(model_id, cache_data)

    # Add cache headers indicating a cache miss
    response.headers["X-Cache"] = "MISS"
    response.headers["Cache-Control"] = f"max-age={settings.cache_model_ttl}"

    return ModelResponse.model_validate(model)


@router.patch("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    model_in: ModelUpdate,
    db: DBSession,
    cache: CacheDep,
) -> ModelResponse:
    """Update an ML model.

    Automatically invalidates the cache for this model.
    """
    # Get the model first
    model = await model_crud.get(db, model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found",
        )

    # Store old values before update for cache invalidation
    old_name = model.name
    old_version = model.version

    # Update in database
    updated = await model_crud.update(db, db_obj=model, obj_in=model_in)

    # Invalidate cache (using updated values, with old values if changed)
    model_cache = ModelCache(cache)
    await model_cache.invalidate_model(
        model_id=updated.id,
        name=updated.name,
        version=updated.version,
        old_name=old_name if old_name != updated.name else None,
        old_version=old_version if old_version != updated.version else None,
    )

    return ModelResponse.model_validate(updated)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    db: DBSession,
    cache: CacheDep,
) -> None:
    """Delete an ML model.

    Automatically invalidates the cache for this model.
    """
    # Get the model first to get name/version for cache invalidation
    model = await model_crud.get(db, model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found",
        )

    # Store name/version before deletion
    model_name = model.name
    model_version = model.version

    # Delete from database
    deleted = await model_crud.delete(db, id=model_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found",
        )

    # Invalidate caches
    model_cache = ModelCache(cache)
    await model_cache.invalidate_model(model_id, model_name, model_version)
    prediction_cache = PredictionCache(cache)
    await prediction_cache.invalidate_model_predictions(model_id)


@router.post("/{model_id}/upload", response_model=ModelUploadResponse)
async def upload_model_file(
    model: ModelDep,
    file: UploadFile,
    db: DBSession,
    storage: StorageDep,
    cache: CacheDep,
) -> ModelUploadResponse:
    """Upload an ONNX model file.

    Validates the file extension and size, stores it via the storage service,
    and updates the model record with file metadata.
    Automatically invalidates the cache for this model.
    """
    # Validate file extension
    allowed_extensions = {".onnx"}
    if file.filename:
        file_ext = (
            "." + file.filename.rsplit(".", 1)[-1].lower()
            if "." in file.filename
            else ""
        )
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    # Check if model already has a file
    if model.file_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Model already has an uploaded file. Delete the model and create a new one to upload a different file.",
        )

    # Generate storage filename using model ID for uniqueness
    storage_filename = f"{model.id}.onnx"

    try:
        # Save file via storage service (handles size validation internally)
        file_path, file_size, file_hash = await storage.save(
            file=file.file,
            filename=storage_filename,
            max_size_bytes=settings.max_model_size_bytes,
        )
    except StorageFullError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        ) from e
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage error: {e}",
        ) from e

    # Update model record
    try:
        updated = await model_crud.update(
            db,
            db_obj=model,
            obj_in={
                "file_path": file_path,
                "file_size_bytes": file_size,
                "file_hash": file_hash,
                "status": ModelStatus.UPLOADED,
            },
        )
    except Exception as db_exc:
        # Attempt to clean up the saved file if DB update fails
        try:
            await storage.delete(file_path)
        except Exception:
            pass  # Optionally log this error
        raise db_exc

    # Invalidate caches
    model_cache = ModelCache(cache)
    await model_cache.invalidate_model(model.id, model.name, model.version)
    prediction_cache = PredictionCache(cache)
    await prediction_cache.invalidate_model_predictions(model.id)

    return ModelUploadResponse(
        id=updated.id,
        file_path=file_path,
        file_size_bytes=file_size,
        file_hash=file_hash,
        status=updated.status,
    )


@router.post("/{model_id}/validate", response_model=ModelValidateResponse)
async def validate_model(
    model: ModelDep,
    db: DBSession,
    storage: StorageDep,
    onnx_service: ONNXDep,
    cache: CacheDep,
) -> ModelValidateResponse:
    """Validate an uploaded ONNX model.

    Loads the model with ONNX Runtime, validates its structure,
    and extracts input/output schemas and metadata. Updates the
    model status to READY on success or ERROR on failure.
    Automatically invalidates the cache for this model.
    """
    # Check if model has an uploaded file
    if not model.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model does not have an uploaded file. Upload a file first.",
        )

    # Check model is in a state that can be validated
    if model.status not in (ModelStatus.UPLOADED, ModelStatus.ERROR):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model cannot be validated in '{model.status.value}' status. "
            f"Only models in 'uploaded' or 'error' status can be validated.",
        )

    # Set status to VALIDATING
    await model_crud.update_status(db, model_id=model.id, status=ModelStatus.VALIDATING)

    # Get the file path from storage
    try:
        file_path = await storage.get_path(model.file_path)
    except Exception as e:
        await model_crud.update(
            db,
            db_obj=model,
            obj_in={"status": ModelStatus.ERROR},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to access model file: {e}",
        ) from e

    # Validate the ONNX model
    result = onnx_service.validate(file_path)

    if result.valid:
        # Convert schemas to serializable format
        input_schema = [s.to_dict() for s in result.input_schema]
        output_schema = [s.to_dict() for s in result.output_schema]

        # Update model with validation results
        updated = await model_crud.update(
            db,
            db_obj=model,
            obj_in={
                "status": ModelStatus.READY,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "model_metadata": result.metadata,
            },
        )

        # Invalidate caches
        model_cache = ModelCache(cache)
        await model_cache.invalidate_model(model.id, model.name, model.version)
        prediction_cache = PredictionCache(cache)
        await prediction_cache.invalidate_model_predictions(model.id)

        return ModelValidateResponse(
            id=updated.id,
            valid=True,
            status=updated.status,
            input_schema=[TensorSchemaResponse(**s) for s in input_schema],
            output_schema=[TensorSchemaResponse(**s) for s in output_schema],
            model_metadata=result.metadata,
            message="Model validated successfully",
        )
    else:
        # Update model with error status
        updated = await model_crud.update(
            db,
            db_obj=model,
            obj_in={"status": ModelStatus.ERROR},
        )

        # Invalidate caches
        model_cache = ModelCache(cache)
        await model_cache.invalidate_model(model.id, model.name, model.version)
        prediction_cache = PredictionCache(cache)
        await prediction_cache.invalidate_model_predictions(model.id)

        return ModelValidateResponse(
            id=updated.id,
            valid=False,
            status=updated.status,
            error_message=result.error_message,
            message="Model validation failed",
        )
