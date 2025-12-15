"""API routes for ML model management."""

from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from app.api.deps import DBSession, ModelDep, StorageDep
from app.config import settings
from app.crud import model_crud
from app.models.ml_model import ModelStatus
from app.schemas.ml_model import (
    ModelCreate,
    ModelListResponse,
    ModelResponse,
    ModelUpdate,
    ModelUploadResponse,
)
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
    """Delete an ML model."""
    deleted = await model_crud.delete(db, id=model.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with id {model.id} not found",
        )


@router.post("/{model_id}/upload", response_model=ModelUploadResponse)
async def upload_model_file(
    model: ModelDep,
    file: UploadFile,
    db: DBSession,
    storage: StorageDep,
) -> ModelUploadResponse:
    """Upload an ONNX model file.

    Validates the file extension and size, stores it via the storage service,
    and updates the model record with file metadata.
    """
    # Validate file extension
    allowed_extensions = {".onnx"}
    if file.filename:
        file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
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
        )
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage error: {e}",
        )

    # Update model record
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

    return ModelUploadResponse(
        id=updated.id,
        file_path=file_path,
        file_size_bytes=file_size,
        file_hash=file_hash,
        status=updated.status,
    )