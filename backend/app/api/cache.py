"""API routes for cache management and metrics."""

from typing import Any

from fastapi import APIRouter, status

from app.api.deps import CacheDep
from app.services.prediction_cache import PredictionCache

router = APIRouter()


@router.get(
    "/cache/metrics",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def get_cache_metrics(
    cache: CacheDep,
) -> dict[str, Any]:
    """Get prediction cache metrics.

    Returns hit/miss statistics for the prediction cache.
    """
    prediction_cache = PredictionCache(cache)
    metrics = await prediction_cache.get_metrics()

    return {
        "prediction_cache": metrics,
    }


@router.post(
    "/cache/metrics/reset",
    response_model=dict[str, str],
    status_code=status.HTTP_200_OK,
)
async def reset_cache_metrics(
    cache: CacheDep,
) -> dict[str, str]:
    """Reset prediction cache metrics.

    Clears hit/miss counters. Useful for testing or monitoring resets.
    """
    prediction_cache = PredictionCache(cache)
    success = await prediction_cache.reset_metrics()

    if success:
        return {"status": "reset"}
    return {"status": "failed"}
