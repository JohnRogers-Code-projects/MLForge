"""API routes for ML model management."""

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import DBSession, ModelDep
from app.crud import model_crud
from app.schemas.ml_model import (
    ModelCreate,
    ModelListResponse,
    ModelResponse,
    ModelUpdate,
)

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
