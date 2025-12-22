"""Health check endpoints."""

import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.schemas.common import CeleryHealthResponse, HealthResponse
from app.services.cache import CacheService, get_cache_service
from app.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Track application start time for uptime calculation
_start_time = time.time()


def check_celery_health() -> dict:
    """Check Celery broker and worker health.

    Returns dict with status and details. This is synchronous because
    Celery's inspect API is synchronous.
    """
    try:
        from app.celery import celery_app

        # Check broker connection by pinging workers
        # timeout=1.0 keeps health checks fast
        inspect = celery_app.control.inspect(timeout=1.0)

        # Get active workers
        ping_response = inspect.ping()

        if ping_response is None:
            # No workers responded - broker connection succeeded but no workers available
            return {
                "status": "no_workers",
                "broker_connected": True,
                "workers": {},
                "queues": ["inference", "default"],
            }

        # Workers responded - fetch stats once for all workers
        workers = {}
        all_stats = inspect.stats() or {}
        for worker_name, response in ping_response.items():
            if response.get("ok") == "pong":
                worker_stats = all_stats.get(worker_name, {})
                workers[worker_name] = {
                    "status": "online",
                    "concurrency": worker_stats.get("pool", {}).get("max-concurrency"),
                    "processed": worker_stats.get("total", {}),
                }

        return {
            "status": "connected",
            "broker_connected": True,
            "workers": workers,
            "queues": ["inference", "default"],
        }

    except Exception as e:
        return {
            "status": "error",
            "broker_connected": False,
            "workers": {},
            "queues": [],
            "error": str(e),
        }


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

    # Check Celery health (quick check)
    celery_health = check_celery_health()
    celery_status = celery_health.get("status", "unknown")

    # Determine overall health
    # healthy = db connected, degraded = db down or redis down or celery down
    if db_status == "connected":
        overall_status = "healthy"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc),
        database=db_status,
        redis=redis_status,
        celery=celery_status,
    )


@router.get("/health/celery", response_model=CeleryHealthResponse)
async def celery_health_check() -> CeleryHealthResponse:
    """Detailed Celery worker health check.

    Returns information about:
    - Broker connectivity
    - Active workers and their status
    - Configured queues
    """
    health = check_celery_health()
    return CeleryHealthResponse(
        status=health.get("status", "unknown"),
        broker_connected=health.get("broker_connected", False),
        workers=health.get("workers", {}),
        queues=health.get("queues", []),
        error=health.get("error"),
        timestamp=datetime.now(timezone.utc),
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


@router.get("/metrics")
async def metrics(
    db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
) -> dict:
    """Application metrics for monitoring dashboards.

    Returns detailed metrics about the application state including:
    - Uptime and process info
    - Database connection status
    - Cache hit/miss rates
    - Celery worker status
    """
    # Calculate uptime
    uptime_seconds = time.time() - _start_time

    # Get cache metrics
    cache_metrics = await cache.get_metrics()

    # Get Celery status
    celery_health = check_celery_health()

    # Check database
    db_connected = True
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_connected = False

    metrics_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "application": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "uptime_seconds": round(uptime_seconds, 2),
            "pid": os.getpid(),
        },
        "database": {
            "connected": db_connected,
        },
        "cache": {
            "enabled": settings.redis_enabled,
            "connected": cache_metrics.get("connected", False),
            "hits": cache_metrics.get("hits", 0),
            "misses": cache_metrics.get("misses", 0),
            "hit_rate": cache_metrics.get("hit_rate", 0),
        },
        "celery": {
            "status": celery_health.get("status"),
            "broker_connected": celery_health.get("broker_connected", False),
            "worker_count": len(celery_health.get("workers", {})),
        },
    }

    logger.debug(
        "Metrics collected",
        extra={"extra_fields": {"uptime": uptime_seconds}},
    )

    return metrics_data
