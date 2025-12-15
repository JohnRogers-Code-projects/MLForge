"""API dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.crud import model_crud
from app.models.ml_model import MLModel
from app.services.storage import StorageService, get_storage_service

# Database session dependency
DBSession = Annotated[AsyncSession, Depends(get_db)]

# Storage service dependency
StorageDep = Annotated[StorageService, Depends(get_storage_service)]


async def get_model_or_404(
    model_id: str,
    db: DBSession,
) -> MLModel:
    """Get a model by ID or raise 404."""
    model = await model_crud.get(db, model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found",
        )
    return model


ModelDep = Annotated[MLModel, Depends(get_model_or_404)]
