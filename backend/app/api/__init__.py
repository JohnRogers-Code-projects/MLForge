"""API routes package."""

from fastapi import APIRouter

from app.api import cache, health, jobs, models, predictions

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(predictions.router, tags=["predictions"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(cache.router, tags=["cache"])
