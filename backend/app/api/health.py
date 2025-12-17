"""Health check endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.schemas.common import HealthResponse
from app.services.cache import CacheService, get_cache_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
) -> HealthResponse:
    """Check application health status."""
    db_status = "connected"

    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    # Check Redis health
    cache_health = await cache.health_check()
    redis_status = cache_health.get("status", "unknown")

    # Determine overall health
    # healthy = db connected, degraded = db down or redis down
    if db_status == "connected":
        overall_status = "healthy"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.utcnow(),
        database=db_status,
        redis=redis_status,
    )


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict:
    """Kubernetes readiness probe."""
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


@router.get("/live")
async def liveness_check() -> dict:
    """Kubernetes liveness probe."""
    return {"status": "alive"}
