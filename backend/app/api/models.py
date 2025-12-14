"""API routes for ML model management."""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from app.api.deps import DBSession, ModelDep
from app.crud import model_crud
from app.models.ml_model import ModelStatus
from app.schemas.ml_model import (
    ModelCreate,
    ModelListResponse,
    ModelResponse,
    ModelUpdate,
    ModelUploadResponse,
)
from app.services.onnx_service import ONNXValidationError, onnx_service
from app.services.storage_service import StorageError, storage_service

logger = logging.getLogger(__name__)

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
    """Create a new ML model metadata entry (without file upload)."""
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


@router.post(
    "/upload",
    response_model=ModelUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and register an ONNX model",
)
async def upload_model(
    db: DBSession,
    file: UploadFile = File(..., description="ONNX model file"),
    name: str = Form(..., min_length=1, max_length=255, description="Model name"),
    version: str = Form(default="1.0.0", max_length=50, description="Model version"),
    description: Optional[str] = Form(None, description="Model description"),
) -> ModelUploadResponse:
    """
    Upload an ONNX model file with validation.

    This endpoint:
    1. Checks for duplicate name+version
    2. Saves the file to storage
    3. Validates the ONNX format
    4. Extracts input/output schema
    5. Creates the model record with READY status
    """
    # Check for existing model with same name and version
    existing = await model_crud.get_by_name_and_version(db, name=name, version=version)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{name}' version '{version}' already exists",
        )

    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".onnx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an ONNX model (.onnx extension)",
        )

    # Create model record in PENDING status
    model_create = ModelCreate(name=name, version=version, description=description)
    model = await model_crud.create(db, obj_in=model_create)

    try:
        # Update status to VALIDATING
        await model_crud.update_status(db, model.id, ModelStatus.VALIDATING)

        # Save file to storage
        file_path, file_size = storage_service.save_model(
            model_id=model.id,
            file=file.file,
            filename=file.filename,
        )

        # Validate ONNX model and extract metadata
        validation_result = onnx_service.validate_model(file_path)

        # Update model with file info and schema
        update_data = ModelUpdate(
            status=ModelStatus.READY,
        )
        model = await model_crud.update(db, db_obj=model, obj_in=update_data)

        # Update file path and schema directly (not in Pydantic schema)
        model.file_path = file_path
        model.file_size_bytes = file_size
        model.file_hash = validation_result["file_hash"]
        model.input_schema = validation_result["input_schema"]
        model.output_schema = validation_result["output_schema"]
        model.metadata = validation_result.get("metadata", {})
        await db.commit()
        await db.refresh(model)

        # Warm up the model for faster first inference
        try:
            onnx_service.warmup_model(model.id, file_path)
        except Exception as e:
            logger.warning(f"Model warmup failed (non-fatal): {e}")

        return ModelUploadResponse(
            id=model.id,
            name=model.name,
            version=model.version,
            status=model.status,
            file_path=model.file_path,
            file_size_bytes=model.file_size_bytes,
            file_hash=model.file_hash,
            input_schema=validation_result["input_schema"],
            output_schema=validation_result["output_schema"],
        )

    except (StorageError, ONNXValidationError) as e:
        # Update model status to ERROR
        model.status = ModelStatus.ERROR
        model.metadata = {"error": str(e)}
        await db.commit()

        # Clean up stored file if it exists
        storage_service.delete_model(model.id)

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model validation failed: {e}",
        )

    except Exception as e:
        # Unexpected error - clean up and re-raise
        logger.exception(f"Unexpected error during model upload: {e}")
        await model_crud.delete(db, id=model.id)
        storage_service.delete_model(model.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during model upload",
        )


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


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model: ModelDep) -> ModelResponse:
    """Get a specific ML model by ID."""
    return ModelResponse.model_validate(model)


@router.patch("/{model_id}", response_model=ModelResponse)
async def update_model(
    model: ModelDep,
    model_in: ModelUpdate,
    db: DBSession,
) -> ModelResponse:
    """Update an ML model."""
    updated = await model_crud.update(db, db_obj=model, obj_in=model_in)
    return ModelResponse.model_validate(updated)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model: ModelDep,
    db: DBSession,
) -> None:
    """Delete an ML model and its associated files."""
    model_id = model.id

    # Delete from database
    deleted = await model_crud.delete(db, id=model_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with id {model_id} not found",
        )

    # Unload from ONNX cache
    try:
        onnx_service.unload_model(model_id)
    except Exception as e:
        logger.error(f"Failed to unload ONNX model {model_id}: {e}")

    # Delete files from storage
    try:
        storage_service.delete_model(model_id)
    except Exception as e:
        logger.error(f"Failed to delete storage for model {model_id}: {e}")


@router.get("/{model_id}/schema", summary="Get model input/output schema")
async def get_model_schema(model: ModelDep) -> dict:
    """Get the input and output schema for a model."""
    return {
        "model_id": model.id,
        "name": model.name,
        "version": model.version,
        "input_schema": model.input_schema,
        "output_schema": model.output_schema,
    }


@router.post("/{model_id}/warmup", summary="Warm up a model for inference")
async def warmup_model(model: ModelDep) -> dict:
    """
    Load a model into memory and run a warmup inference.

    This prepares the model for fast inference by:
    1. Loading the ONNX session
    2. Running a dummy inference to initialize all layers
    """
    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model is not ready (status: {model.status.value})",
        )

    if not model.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no file path",
        )

    try:
        warmup_time_ms = onnx_service.warmup_model(model.id, model.file_path)
        return {
            "model_id": model.id,
            "status": "warmed_up",
            "warmup_time_ms": warmup_time_ms,
            "loaded": onnx_service.is_loaded(model.id),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Warmup failed: {e}",
        )
