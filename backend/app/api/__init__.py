"""API routes package."""

from fastapi import APIRouter

from app.api import health, models, predictions, jobs

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
