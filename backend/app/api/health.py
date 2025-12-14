"""Health check endpoints."""

from datetime import datetime

import onnxruntime as ort
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.schemas.common import HealthResponse
from app.services.onnx_service import onnx_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Check application health status."""
    db_status = "connected"
    redis_status = "not_configured"  # Will be updated in Phase 3
    loaded_models = 0

    # Check database
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    # Check ONNX runtime
    try:
        # Verify ONNX runtime is available
        _ = ort.get_available_providers()
        onnx_status = "available"
        loaded_models = len(onnx_service.get_loaded_models())
    except Exception:
        onnx_status = "unavailable"

    # Determine overall status
    is_healthy = db_status == "connected" and onnx_status == "available"

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.utcnow(),
        database=db_status,
        redis=redis_status,
        onnx_runtime=onnx_status,
        loaded_models=loaded_models,
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
