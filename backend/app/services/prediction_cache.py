"""Prediction result caching utilities.

This module provides caching for prediction results to avoid redundant
inference calls for identical inputs. Uses deterministic hashing of
input data to generate cache keys.

Cache key pattern:
- prediction:{model_id}:{input_hash} - Prediction result by model and input

Design notes:
- Uses MD5 for input hashing (not for security, just for cache key generation)
- Input is JSON-serialized with sorted keys for deterministic hashing
- Only caches output_data and inference_time_ms, not the full prediction record
- DB records are still created on cache hits for audit trail
"""

import hashlib
import json
import logging
from typing import Any

from app.config import settings
from app.services.cache import CacheService

logger = logging.getLogger(__name__)


# Cache key patterns
PREDICTION_KEY = "prediction:{model_id}:{input_hash}"
PREDICTION_METRICS_HITS = "metrics:prediction:hits"
PREDICTION_METRICS_MISSES = "metrics:prediction:misses"


def hash_input(input_data: dict[str, Any]) -> str:
    """Generate a deterministic hash of input data for cache key.

    Uses MD5 because we're not using this for security, just for
    generating consistent cache keys. MD5 is faster than SHA-256
    and good enough for this use case.

    Args:
        input_data: Dictionary of input data

    Returns:
        16-character hex hash of the input data
    """
    # Serialize with sorted keys for determinism
    # Use separators without spaces to minimize size
    serialized = json.dumps(input_data, sort_keys=True, separators=(",", ":"))
    # MD5 is fine for non-security cache keys, take first 16 chars
    return hashlib.md5(serialized.encode()).hexdigest()[:16]


class PredictionCacheResult:
    """Result from prediction cache lookup.

    Attributes:
        hit: Whether the lookup was a cache hit
        output_data: Cached prediction output (if hit)
        inference_time_ms: Cached inference time (if hit)
    """

    def __init__(
        self,
        hit: bool,
        output_data: dict[str, Any] | None = None,
        inference_time_ms: float | None = None,
    ):
        self.hit = hit
        self.output_data = output_data
        self.inference_time_ms = inference_time_ms


class PredictionCache:
    """Helper class for prediction result caching.

    Provides methods for caching and retrieving prediction results,
    with automatic cache key generation from model ID and input hash.
    """

    def __init__(self, cache: CacheService):
        """Initialize with a cache service instance."""
        self.cache = cache
        self.prediction_ttl = settings.cache_prediction_ttl
        self.enabled = settings.cache_prediction_enabled

    def _prediction_key(self, model_id: str, input_hash: str) -> str:
        """Generate cache key for a prediction."""
        return PREDICTION_KEY.format(model_id=model_id, input_hash=input_hash)

    async def get_prediction(
        self, model_id: str, input_data: dict[str, Any]
    ) -> PredictionCacheResult:
        """Look up a cached prediction result.

        Args:
            model_id: Model UUID
            input_data: Input data for the prediction

        Returns:
            PredictionCacheResult with hit=True if found, hit=False otherwise.
        """
        if not self.enabled:
            return PredictionCacheResult(hit=False)

        input_hash = hash_input(input_data)
        key = self._prediction_key(model_id, input_hash)

        cached = await self.cache.get(key)
        if cached is not None:
            # Track hit metric
            await self._increment_hits()
            logger.debug(f"Prediction cache hit for model {model_id}")
            return PredictionCacheResult(
                hit=True,
                output_data=cached.get("output_data"),
                inference_time_ms=cached.get("inference_time_ms"),
            )

        # Track miss metric
        await self._increment_misses()
        return PredictionCacheResult(hit=False)

    async def set_prediction(
        self,
        model_id: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        inference_time_ms: float,
    ) -> bool:
        """Cache a prediction result.

        Args:
            model_id: Model UUID
            input_data: Input data for the prediction (used for key generation)
            output_data: Prediction output to cache
            inference_time_ms: Inference time to cache

        Returns:
            True if cached successfully, False otherwise.
        """
        if not self.enabled:
            return False

        input_hash = hash_input(input_data)
        key = self._prediction_key(model_id, input_hash)

        cache_value = {
            "output_data": output_data,
            "inference_time_ms": inference_time_ms,
        }

        result = await self.cache.set(key, cache_value, ttl=self.prediction_ttl)
        if result:
            logger.debug(f"Cached prediction for model {model_id}")
        return result

    async def invalidate_model_predictions(self, model_id: str) -> int:
        """Invalidate all cached predictions for a model.

        Should be called when a model file is updated or the model
        is re-validated.

        Args:
            model_id: Model UUID

        Returns:
            Number of cache entries deleted.
        """
        prefix = f"prediction:{model_id}:"
        count = await self.cache.clear_prefix(prefix)
        if count > 0:
            logger.info(f"Invalidated {count} cached predictions for model {model_id}")
        return count

    async def _increment_hits(self) -> None:
        """Increment the cache hits counter."""
        # Use public incr method - returns None on error/disconnected
        await self.cache.incr(PREDICTION_METRICS_HITS)

    async def _increment_misses(self) -> None:
        """Increment the cache misses counter."""
        # Use public incr method - returns None on error/disconnected
        await self.cache.incr(PREDICTION_METRICS_MISSES)

    async def get_metrics(self) -> dict[str, Any]:
        """Get prediction cache metrics.

        Returns:
            Dict with hits, misses, and hit_rate.
        """
        hits = 0
        misses = 0

        if self.cache.is_connected:
            hits_val = await self.cache.get_raw(PREDICTION_METRICS_HITS)
            misses_val = await self.cache.get_raw(PREDICTION_METRICS_MISSES)
            hits = int(hits_val) if hits_val else 0
            misses = int(misses_val) if misses_val else 0

        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0.0

        return {
            "hits": hits,
            "misses": misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "enabled": self.enabled,
            "ttl_seconds": self.prediction_ttl,
        }

    async def reset_metrics(self) -> bool:
        """Reset prediction cache metrics to zero.

        Returns:
            True if successful, False otherwise.
        """
        if not self.cache.is_connected:
            return False

        deleted = await self.cache.delete_keys(
            PREDICTION_METRICS_HITS,
            PREDICTION_METRICS_MISSES,
        )
        return deleted > 0
