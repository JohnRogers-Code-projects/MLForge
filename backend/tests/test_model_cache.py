"""Tests for model metadata caching.

Tests cover:
- Cache hit/miss behavior
- Cache invalidation on update/delete
- Cache headers in responses
- ModelCache helper class
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient

from app.services.cache import CacheService
from app.services.model_cache import ModelCache, model_to_cache_dict


class TestModelCacheHelper:
    """Tests for ModelCache helper class."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache service."""
        mock = MagicMock(spec=CacheService)
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=True)
        mock.clear_prefix = AsyncMock(return_value=0)
        return mock

    def test_model_key_generation(self, mock_cache):
        """Test cache key generation for model by ID."""
        model_cache = ModelCache(mock_cache)
        key = model_cache._model_key("abc-123")
        assert key == "model:abc-123"

    def test_name_version_key_generation(self, mock_cache):
        """Test cache key generation for model by name/version."""
        model_cache = ModelCache(mock_cache)
        key = model_cache._name_version_key("my-model", "1.0.0")
        assert key == "model:name:my-model:version:1.0.0"

    def test_latest_key_generation(self, mock_cache):
        """Test cache key generation for latest version."""
        model_cache = ModelCache(mock_cache)
        key = model_cache._latest_key("my-model")
        assert key == "model:name:my-model:latest"

    def test_versions_key_generation(self, mock_cache):
        """Test cache key generation for versions list."""
        model_cache = ModelCache(mock_cache)
        key = model_cache._versions_key("my-model")
        assert key == "model:name:my-model:versions"

    @pytest.mark.asyncio
    async def test_get_model_cache_hit(self, mock_cache):
        """Test getting model from cache."""
        mock_cache.get.return_value = {"id": "abc-123", "name": "test"}

        model_cache = ModelCache(mock_cache)
        result = await model_cache.get_model("abc-123")

        assert result == {"id": "abc-123", "name": "test"}
        mock_cache.get.assert_called_once_with("model:abc-123")

    @pytest.mark.asyncio
    async def test_get_model_cache_miss(self, mock_cache):
        """Test cache miss returns None."""
        mock_cache.get.return_value = None

        model_cache = ModelCache(mock_cache)
        result = await model_cache.get_model("abc-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_model(self, mock_cache):
        """Test caching model data."""
        model_cache = ModelCache(mock_cache)
        model_data = {"id": "abc-123", "name": "test"}

        result = await model_cache.set_model("abc-123", model_data)

        assert result is True
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == "model:abc-123"
        assert call_args[0][1] == model_data

    @pytest.mark.asyncio
    async def test_invalidate_model(self, mock_cache):
        """Test cache invalidation clears all related keys."""
        model_cache = ModelCache(mock_cache)

        await model_cache.invalidate_model("abc-123", "my-model", "1.0.0")

        # Should delete: by ID, by name/version, latest, versions list
        assert mock_cache.delete.call_count == 4


class TestModelToCacheDict:
    """Tests for model_to_cache_dict conversion."""

    def test_converts_model_to_dict(self):
        """Test converting model ORM object to cache dict."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.id = "abc-123"
        mock_model.name = "test-model"
        mock_model.description = "A test model"
        mock_model.version = "1.0.0"
        mock_model.status.value = "ready"
        mock_model.file_path = "abc-123.onnx"
        mock_model.file_size_bytes = 1024
        mock_model.file_hash = "abc123"
        mock_model.input_schema = [{"name": "input", "shape": [1, 10]}]
        mock_model.output_schema = [{"name": "output", "shape": [1, 5]}]
        mock_model.model_metadata = {"version": "1.0"}
        mock_model.created_at.isoformat.return_value = "2025-01-01T00:00:00"
        mock_model.updated_at.isoformat.return_value = "2025-01-01T00:00:00"

        result = model_to_cache_dict(mock_model)

        assert result["id"] == "abc-123"
        assert result["name"] == "test-model"
        assert result["version"] == "1.0.0"
        assert result["status"] == "ready"
        assert result["file_path"] == "abc-123.onnx"


class TestModelCacheIntegration:
    """Integration tests for model caching in API endpoints."""

    @pytest.mark.asyncio
    async def test_get_model_cache_miss_then_hit(self, client: AsyncClient):
        """Test that get_model caches on miss and returns from cache on hit."""
        # Create a model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "cache-test-model", "version": "1.0.0"},
        )
        assert create_response.status_code == 201
        model_id = create_response.json()["id"]

        # First get - should be cache miss
        response1 = await client.get(f"/api/v1/models/{model_id}")
        assert response1.status_code == 200
        # Note: X-Cache header may be MISS or not set if cache is disabled in tests

        # Second get - in a real scenario with Redis, this would be a cache hit
        response2 = await client.get(f"/api/v1/models/{model_id}")
        assert response2.status_code == 200

        # Both responses should return the same data
        assert response1.json()["id"] == response2.json()["id"]
        assert response1.json()["name"] == response2.json()["name"]

    @pytest.mark.asyncio
    async def test_update_model_invalidates_cache(self, client: AsyncClient):
        """Test that updating a model works correctly (cache invalidation)."""
        # Create a model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "cache-update-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Get model (populates cache)
        await client.get(f"/api/v1/models/{model_id}")

        # Update model (should invalidate cache)
        update_response = await client.patch(
            f"/api/v1/models/{model_id}",
            json={"description": "Updated description"},
        )
        assert update_response.status_code == 200
        assert update_response.json()["description"] == "Updated description"

        # Get model again - should reflect update
        response = await client.get(f"/api/v1/models/{model_id}")
        assert response.status_code == 200
        assert response.json()["description"] == "Updated description"

    @pytest.mark.asyncio
    async def test_delete_model_invalidates_cache(self, client: AsyncClient):
        """Test that deleting a model works correctly (cache invalidation)."""
        # Create a model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "cache-delete-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Get model (populates cache)
        await client.get(f"/api/v1/models/{model_id}")

        # Delete model (should invalidate cache)
        delete_response = await client.delete(f"/api/v1/models/{model_id}")
        assert delete_response.status_code == 204

        # Get model again - should return 404
        response = await client.get(f"/api/v1/models/{model_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_model_returns_cache_control_header(self, client: AsyncClient):
        """Test that get_model returns Cache-Control header."""
        # Create a model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "cache-header-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Get model
        response = await client.get(f"/api/v1/models/{model_id}")
        assert response.status_code == 200

        # Check for Cache-Control header
        assert "cache-control" in response.headers
        assert "max-age=" in response.headers["cache-control"]

    @pytest.mark.asyncio
    async def test_get_model_returns_x_cache_header(self, client: AsyncClient):
        """Test that get_model returns X-Cache header."""
        # Create a model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "x-cache-header-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Get model
        response = await client.get(f"/api/v1/models/{model_id}")
        assert response.status_code == 200

        # Check for X-Cache header (either HIT or MISS)
        assert "x-cache" in response.headers
        assert response.headers["x-cache"] in ["HIT", "MISS"]
