"""Tests for prediction caching functionality.

Tests cover:
- hash_input function for deterministic hashing
- PredictionCache class operations
- Cache hit/miss behavior in predict endpoint
- skip_cache parameter
- Cache invalidation on model changes
- Cache metrics endpoint
"""

import io
import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient
import onnx

from app.services.cache import CacheService
from app.services.prediction_cache import (
    PredictionCache,
    PredictionCacheResult,
    hash_input,
)
from tests.conftest import create_simple_onnx_model


class TestHashInput:
    """Tests for the hash_input function."""

    def test_hash_input_deterministic(self):
        """Same input produces same hash."""
        data = {"input": [[1.0, 2.0, 3.0]]}
        hash1 = hash_input(data)
        hash2 = hash_input(data)
        assert hash1 == hash2

    def test_hash_input_different_values(self):
        """Different values produce different hashes."""
        data1 = {"input": [[1.0, 2.0, 3.0]]}
        data2 = {"input": [[1.0, 2.0, 4.0]]}  # Different last value
        assert hash_input(data1) != hash_input(data2)

    def test_hash_input_key_order_invariant(self):
        """Hash is independent of key order (sorted)."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        assert hash_input(data1) == hash_input(data2)

    def test_hash_input_length(self):
        """Hash is 16 characters (truncated MD5)."""
        data = {"input": [[1.0]]}
        h = hash_input(data)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_input_nested_dict(self):
        """Handles nested dictionaries."""
        data = {"input": {"nested": {"deep": [1, 2, 3]}}}
        h = hash_input(data)
        assert len(h) == 16

    def test_hash_input_empty_dict(self):
        """Handles empty dictionary."""
        h = hash_input({})
        assert len(h) == 16


class TestPredictionCacheResult:
    """Tests for PredictionCacheResult class."""

    def test_cache_hit_result(self):
        """Cache hit result has correct attributes."""
        result = PredictionCacheResult(
            hit=True,
            output_data={"output": [[1.0]]},
            inference_time_ms=5.0,
        )
        assert result.hit is True
        assert result.output_data == {"output": [[1.0]]}
        assert result.inference_time_ms == 5.0

    def test_cache_miss_result(self):
        """Cache miss result has hit=False."""
        result = PredictionCacheResult(hit=False)
        assert result.hit is False
        assert result.output_data is None
        assert result.inference_time_ms is None


class TestPredictionCacheDisabled:
    """Tests for PredictionCache when caching is disabled."""

    @pytest.fixture
    def disabled_cache(self):
        """Create a disabled cache service."""
        return CacheService(enabled=False)

    @pytest.mark.asyncio
    async def test_get_prediction_disabled(self, disabled_cache):
        """Get returns miss when cache is disabled."""
        from app.config import settings
        original = settings.cache_prediction_enabled
        settings.cache_prediction_enabled = False
        try:
            pred_cache = PredictionCache(disabled_cache)
            result = await pred_cache.get_prediction("model-id", {"input": [[1.0]]})
            assert result.hit is False
        finally:
            settings.cache_prediction_enabled = original

    @pytest.mark.asyncio
    async def test_set_prediction_disabled(self, disabled_cache):
        """Set returns False when cache is disabled."""
        from app.config import settings
        original = settings.cache_prediction_enabled
        settings.cache_prediction_enabled = False
        try:
            pred_cache = PredictionCache(disabled_cache)
            result = await pred_cache.set_prediction(
                "model-id", {"input": [[1.0]]}, {"output": [[2.0]]}, 5.0
            )
            assert result is False
        finally:
            settings.cache_prediction_enabled = original


class TestPredictionCacheWithMockedRedis:
    """Tests for PredictionCache with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=True)
        mock.incr = AsyncMock(return_value=1)
        mock.delete = AsyncMock(return_value=1)
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def mock_cache_service(self, mock_redis):
        """Create a mock cache service."""
        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis
        return cache

    @pytest.mark.asyncio
    async def test_get_prediction_cache_hit(self, mock_cache_service, mock_redis):
        """Cache hit returns stored prediction data."""
        cached_data = '{"output_data": {"output": [[2.0]]}, "inference_time_ms": 5.0}'
        mock_redis.get.return_value = cached_data

        pred_cache = PredictionCache(mock_cache_service)
        result = await pred_cache.get_prediction("model-123", {"input": [[1.0]]})

        assert result.hit is True
        assert result.output_data == {"output": [[2.0]]}
        assert result.inference_time_ms == 5.0

    @pytest.mark.asyncio
    async def test_get_prediction_cache_miss(self, mock_cache_service, mock_redis):
        """Cache miss returns hit=False."""
        mock_redis.get.return_value = None

        pred_cache = PredictionCache(mock_cache_service)
        result = await pred_cache.get_prediction("model-123", {"input": [[1.0]]})

        assert result.hit is False

    @pytest.mark.asyncio
    async def test_set_prediction_stores_data(self, mock_cache_service, mock_redis):
        """Set stores prediction data in cache."""
        pred_cache = PredictionCache(mock_cache_service)
        result = await pred_cache.set_prediction(
            model_id="model-123",
            input_data={"input": [[1.0]]},
            output_data={"output": [[2.0]]},
            inference_time_ms=5.0,
        )

        assert result is True
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_key_includes_model_id(self, mock_cache_service, mock_redis):
        """Cache key is based on model ID and input hash."""
        pred_cache = PredictionCache(mock_cache_service)
        await pred_cache.get_prediction("model-abc", {"input": [[1.0]]})

        # Verify the key format
        call_args = mock_redis.get.call_args
        key = call_args[0][0]
        assert "model-abc" in key
        assert key.startswith("test:prediction:")

    @pytest.mark.asyncio
    async def test_get_prediction_increments_hit_counter(
        self, mock_cache_service, mock_redis
    ):
        """Cache hit increments hits counter."""
        cached_data = '{"output_data": {}, "inference_time_ms": 1.0}'
        mock_redis.get.return_value = cached_data

        pred_cache = PredictionCache(mock_cache_service)
        await pred_cache.get_prediction("model-123", {"input": [[1.0]]})

        # Verify incr was called for hits
        incr_calls = [
            call for call in mock_redis.incr.call_args_list
            if "hits" in str(call)
        ]
        assert len(incr_calls) == 1

    @pytest.mark.asyncio
    async def test_get_prediction_increments_miss_counter(
        self, mock_cache_service, mock_redis
    ):
        """Cache miss increments misses counter."""
        mock_redis.get.return_value = None

        pred_cache = PredictionCache(mock_cache_service)
        await pred_cache.get_prediction("model-123", {"input": [[1.0]]})

        # Verify incr was called for misses
        incr_calls = [
            call for call in mock_redis.incr.call_args_list
            if "misses" in str(call)
        ]
        assert len(incr_calls) == 1

    @pytest.mark.asyncio
    async def test_invalidate_model_predictions(self, mock_cache_service, mock_redis):
        """Invalidate clears all predictions for a model."""
        # Mock scan_iter to return some keys
        async def mock_scan_iter(*args, **kwargs):
            for key in ["test:prediction:model-123:abc", "test:prediction:model-123:def"]:
                yield key

        mock_redis.scan_iter = mock_scan_iter

        pred_cache = PredictionCache(mock_cache_service)
        count = await pred_cache.invalidate_model_predictions("model-123")

        # Should have deleted the matching keys
        mock_redis.delete.assert_called()


class TestPredictionCacheMetrics:
    """Tests for cache metrics functionality."""

    @pytest.fixture
    def mock_redis_with_metrics(self):
        """Create a mock Redis with metrics values."""
        mock = AsyncMock()
        mock.get = AsyncMock(side_effect=lambda k: "10" if "hits" in k else "5")
        mock.delete = AsyncMock(return_value=2)
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def mock_cache_with_metrics(self, mock_redis_with_metrics):
        """Create a cache service with metrics."""
        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis_with_metrics
        return cache

    @pytest.mark.asyncio
    async def test_get_metrics_calculates_hit_rate(self, mock_cache_with_metrics):
        """Metrics includes hit rate calculation."""
        pred_cache = PredictionCache(mock_cache_with_metrics)
        metrics = await pred_cache.get_metrics()

        assert metrics["hits"] == 10
        assert metrics["misses"] == 5
        assert metrics["total_requests"] == 15
        # 10/15 = 66.67%
        assert abs(metrics["hit_rate_percent"] - 66.67) < 0.1

    @pytest.mark.asyncio
    async def test_get_metrics_zero_requests(self):
        """Metrics handles zero requests without division by zero."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        pred_cache = PredictionCache(cache)
        metrics = await pred_cache.get_metrics()

        assert metrics["hits"] == 0
        assert metrics["misses"] == 0
        assert metrics["total_requests"] == 0
        assert metrics["hit_rate_percent"] == 0.0

    @pytest.mark.asyncio
    async def test_reset_metrics(self, mock_cache_with_metrics):
        """Reset metrics clears counters."""
        pred_cache = PredictionCache(mock_cache_with_metrics)
        result = await pred_cache.reset_metrics()

        assert result is True
        mock_cache_with_metrics._client.delete.assert_called_once()


@pytest.fixture
def valid_onnx_file() -> io.BytesIO:
    """Create a valid ONNX model file for testing."""
    model = create_simple_onnx_model()
    buffer = io.BytesIO()
    onnx.save(model, buffer)
    buffer.seek(0)
    return buffer


async def setup_ready_model(client: AsyncClient, valid_onnx_file: io.BytesIO) -> str:
    """Helper to create, upload, and validate a model. Returns model_id."""
    valid_onnx_file.seek(0)
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "cache-test-model", "version": "1.0.0"},
    )
    assert create_response.status_code == 201
    model_id = create_response.json()["id"]

    files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
    upload_response = await client.post(f"/api/v1/models/{model_id}/upload", files=files)
    assert upload_response.status_code == 200

    validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
    assert validate_response.status_code == 200
    assert validate_response.json()["status"] == "ready"

    return model_id


class TestPredictionCachingIntegration:
    """Integration tests for prediction caching in the API."""

    @pytest.mark.asyncio
    async def test_predict_sets_cache_header_miss(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """First prediction returns X-Cache: MISS."""
        model_id = await setup_ready_model(client, valid_onnx_file)

        response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}},
        )

        assert response.status_code == 201
        # Note: Cache is disabled in tests, but header should still be set
        assert "X-Cache" in response.headers
        assert response.headers["X-Cache"] == "MISS"

    @pytest.mark.asyncio
    async def test_predict_with_skip_cache(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """skip_cache=true forces fresh inference."""
        model_id = await setup_ready_model(client, valid_onnx_file)

        # First prediction
        await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}},
        )

        # Second prediction with skip_cache
        response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}, "skip_cache": True},
        )

        assert response.status_code == 201
        # Should still be MISS because cache is skipped
        assert response.headers.get("X-Cache") == "MISS"

    @pytest.mark.asyncio
    async def test_predict_cached_field_is_false_on_miss(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Prediction record has cached=False on cache miss."""
        model_id = await setup_ready_model(client, valid_onnx_file)

        response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["cached"] is False


class TestCacheMetricsEndpoint:
    """Tests for the /cache/metrics endpoint."""

    @pytest.mark.asyncio
    async def test_get_cache_metrics(self, client: AsyncClient):
        """GET /cache/metrics returns metrics structure."""
        response = await client.get("/api/v1/cache/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "prediction_cache" in data
        assert "hits" in data["prediction_cache"]
        assert "misses" in data["prediction_cache"]
        assert "hit_rate_percent" in data["prediction_cache"]

    @pytest.mark.asyncio
    async def test_reset_cache_metrics(self, client: AsyncClient):
        """POST /cache/metrics/reset returns status."""
        response = await client.post("/api/v1/cache/metrics/reset")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
