"""Tests for Redis cache service.

Tests cover:
- CacheService initialization and connection
- Get/set/delete operations
- Graceful degradation when Redis unavailable
- Health check functionality
- Key namespacing
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.cache import CacheService, set_cache_service


class TestCacheServiceDisabled:
    """Tests for cache service when disabled."""

    @pytest.mark.asyncio
    async def test_disabled_cache_connect_returns_false(self):
        """Disabled cache returns False on connect."""
        cache = CacheService(enabled=False)
        result = await cache.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disabled_cache_get_returns_none(self):
        """Disabled cache returns None on get."""
        cache = CacheService(enabled=False)
        result = await cache.get("any_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_cache_set_returns_false(self):
        """Disabled cache returns False on set."""
        cache = CacheService(enabled=False)
        result = await cache.set("any_key", "any_value")
        assert result is False

    @pytest.mark.asyncio
    async def test_disabled_cache_delete_returns_false(self):
        """Disabled cache returns False on delete."""
        cache = CacheService(enabled=False)
        result = await cache.delete("any_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_disabled_cache_exists_returns_false(self):
        """Disabled cache returns False on exists."""
        cache = CacheService(enabled=False)
        result = await cache.exists("any_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_disabled_cache_health_check(self):
        """Disabled cache health check returns disabled status."""
        cache = CacheService(enabled=False)
        health = await cache.health_check()
        assert health["status"] == "disabled"
        assert health["enabled"] is False
        assert health["connected"] is False


class TestCacheServiceKeyNamespacing:
    """Tests for key namespacing."""

    def test_make_key_adds_prefix(self):
        """Keys are prefixed with configured prefix."""
        cache = CacheService(prefix="test:", enabled=False)
        assert cache._make_key("mykey") == "test:mykey"

    def test_make_key_default_prefix(self):
        """Default prefix from settings is used."""
        cache = CacheService(enabled=False)
        # Default is "modelforge:" from settings
        key = cache._make_key("mykey")
        assert key.startswith("modelforge:")
        assert key.endswith("mykey")


class TestCacheServiceWithMockedRedis:
    """Tests with mocked Redis client."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        mock.exists = AsyncMock(return_value=1)
        mock.info = AsyncMock(return_value={"redis_version": "7.0.0", "uptime_in_seconds": 1000})
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        mock = MagicMock()
        mock.disconnect = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_get_returns_cached_value(self, mock_redis, mock_pool):
        """Get returns cached string value."""
        mock_redis.get.return_value = '"cached_value"'

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        result = await cache.get("mykey")

        assert result == "cached_value"
        mock_redis.get.assert_called_once_with("test:mykey")

    @pytest.mark.asyncio
    async def test_get_returns_cached_dict(self, mock_redis, mock_pool):
        """Get deserializes JSON objects."""
        mock_redis.get.return_value = '{"foo": "bar", "num": 42}'

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        result = await cache.get("mykey")

        assert result == {"foo": "bar", "num": 42}

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self, mock_redis, mock_pool):
        """Get returns None on cache miss."""
        mock_redis.get.return_value = None

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        result = await cache.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_stores_value(self, mock_redis, mock_pool):
        """Set stores value with TTL."""
        cache = CacheService(prefix="test:", default_ttl=300, enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        result = await cache.set("mykey", {"data": "value"})

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "test:mykey"
        assert '"data"' in call_args[0][1]
        assert call_args[1]["ex"] == 300

    @pytest.mark.asyncio
    async def test_set_custom_ttl(self, mock_redis, mock_pool):
        """Set uses custom TTL when provided."""
        cache = CacheService(prefix="test:", default_ttl=300, enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        await cache.set("mykey", "value", ttl=60)

        call_args = mock_redis.set.call_args
        assert call_args[1]["ex"] == 60

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, mock_redis, mock_pool):
        """Delete removes key and returns True."""
        mock_redis.delete.return_value = 1

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        result = await cache.delete("mykey")

        assert result is True
        mock_redis.delete.assert_called_once_with("test:mykey")

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_miss(self, mock_redis, mock_pool):
        """Delete returns False when key doesn't exist."""
        mock_redis.delete.return_value = 0

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        result = await cache.delete("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_returns_true(self, mock_redis, mock_pool):
        """Exists returns True when key exists."""
        mock_redis.exists.return_value = 1

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        result = await cache.exists("mykey")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false(self, mock_redis, mock_pool):
        """Exists returns False when key doesn't exist."""
        mock_redis.exists.return_value = 0

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        result = await cache.exists("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_redis, mock_pool):
        """Health check returns healthy status."""
        cache = CacheService(enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        health = await cache.health_check()

        assert health["status"] == "healthy"
        assert health["enabled"] is True
        assert health["connected"] is True
        assert health["redis_version"] == "7.0.0"

    @pytest.mark.asyncio
    async def test_disconnect_closes_connections(self, mock_redis, mock_pool):
        """Disconnect closes client and pool."""
        cache = CacheService(enabled=True)
        cache._connected = True
        cache._client = mock_redis
        cache._pool = mock_pool

        await cache.disconnect()

        mock_redis.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()
        assert cache._connected is False
        assert cache._client is None
        assert cache._pool is None


class TestCacheServiceGracefulDegradation:
    """Tests for graceful degradation on Redis errors."""

    @pytest.fixture
    def failing_redis(self):
        """Create a Redis mock that raises errors."""
        from redis.exceptions import RedisError

        mock = AsyncMock()
        mock.get = AsyncMock(side_effect=RedisError("Connection refused"))
        mock.set = AsyncMock(side_effect=RedisError("Connection refused"))
        mock.delete = AsyncMock(side_effect=RedisError("Connection refused"))
        mock.exists = AsyncMock(side_effect=RedisError("Connection refused"))
        mock.ping = AsyncMock(side_effect=RedisError("Connection refused"))
        mock.info = AsyncMock(side_effect=RedisError("Connection refused"))
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_get_returns_none_on_error(self, failing_redis):
        """Get returns None instead of raising on Redis error."""
        cache = CacheService(enabled=True)
        cache._connected = True
        cache._client = failing_redis

        result = await cache.get("mykey")

        assert result is None  # Graceful degradation

    @pytest.mark.asyncio
    async def test_set_returns_false_on_error(self, failing_redis):
        """Set returns False instead of raising on Redis error."""
        cache = CacheService(enabled=True)
        cache._connected = True
        cache._client = failing_redis

        result = await cache.set("mykey", "value")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_error(self, failing_redis):
        """Delete returns False instead of raising on Redis error."""
        cache = CacheService(enabled=True)
        cache._connected = True
        cache._client = failing_redis

        result = await cache.delete("mykey")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_error(self, failing_redis):
        """Exists returns False instead of raising on Redis error."""
        cache = CacheService(enabled=True)
        cache._connected = True
        cache._client = failing_redis

        result = await cache.exists("mykey")

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_error(self, failing_redis):
        """Health check returns unhealthy status on Redis error."""
        cache = CacheService(enabled=True)
        cache._connected = True
        cache._client = failing_redis

        health = await cache.health_check()

        assert health["status"] == "unhealthy"
        assert health["connected"] is False
        assert "error" in health


class TestCacheServiceGetOrSet:
    """Tests for get_or_set convenience method."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=True)
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_get_or_set_cache_hit(self, mock_redis):
        """Returns cached value on hit."""
        mock_redis.get.return_value = '"cached"'

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        async def factory():
            return "computed"

        result = await cache.get_or_set("mykey", factory)

        assert result == "cached"
        mock_redis.set.assert_not_called()  # Factory not called

    @pytest.mark.asyncio
    async def test_get_or_set_cache_miss(self, mock_redis):
        """Computes and caches value on miss."""
        mock_redis.get.return_value = None

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        factory_called = False

        async def factory():
            nonlocal factory_called
            factory_called = True
            return "computed"

        result = await cache.get_or_set("mykey", factory)

        assert result == "computed"
        assert factory_called
        mock_redis.set.assert_called_once()


class TestCacheServiceNotConnected:
    """Tests for operations when not connected."""

    @pytest.mark.asyncio
    async def test_get_returns_none_when_not_connected(self):
        """Get returns None when not connected."""
        cache = CacheService(enabled=True)
        # Not connected - _connected is False by default
        result = await cache.get("mykey")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_returns_false_when_not_connected(self):
        """Set returns False when not connected."""
        cache = CacheService(enabled=True)
        result = await cache.set("mykey", "value")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self):
        """Health check returns disconnected status."""
        cache = CacheService(enabled=True)
        health = await cache.health_check()
        assert health["status"] == "disconnected"
        assert health["connected"] is False
