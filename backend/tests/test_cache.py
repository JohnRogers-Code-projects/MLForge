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


class TestCacheServiceNonJsonValues:
    """Tests for handling non-JSON values."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_get_returns_raw_string_on_json_decode_error(self, mock_redis):
        """Get returns raw string when JSON decode fails."""
        # This is not valid JSON
        mock_redis.get.return_value = "plain string without quotes"

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.get("mykey")

        # Should return the raw string value since it's not valid JSON
        assert result == "plain string without quotes"


class TestCacheServiceClearPrefix:
    """Tests for clear_prefix method."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_clear_prefix_deletes_matching_keys(self, mock_redis):
        """Clear prefix deletes all keys with matching prefix."""
        # Mock scan_iter to return some keys
        async def mock_scan_iter(**kwargs):
            for key in ["test:prefix:key1", "test:prefix:key2"]:
                yield key

        mock_redis.scan_iter = mock_scan_iter
        mock_redis.delete = AsyncMock(return_value=2)

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.clear_prefix("prefix:")

        assert result == 2
        mock_redis.delete.assert_called_once_with(
            "test:prefix:key1", "test:prefix:key2"
        )

    @pytest.mark.asyncio
    async def test_clear_prefix_no_matching_keys(self, mock_redis):
        """Clear prefix returns 0 when no keys match."""
        # Mock scan_iter to return no keys (empty async generator)
        async def mock_scan_iter(**kwargs):
            if False:
                yield  # Makes this an async generator that yields nothing

        mock_redis.scan_iter = mock_scan_iter

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.clear_prefix("nonexistent:")

        assert result == 0
        mock_redis.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_prefix_returns_zero_on_error(self, mock_redis):
        """Clear prefix returns 0 on Redis error."""
        from redis.exceptions import RedisError

        async def mock_scan_iter(**kwargs):
            if False:
                yield  # Makes this an async generator
            raise RedisError("Connection error")

        mock_redis.scan_iter = mock_scan_iter

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.clear_prefix("prefix:")

        assert result == 0

    @pytest.mark.asyncio
    async def test_clear_prefix_not_connected(self):
        """Clear prefix returns 0 when not connected."""
        cache = CacheService(enabled=True)
        result = await cache.clear_prefix("prefix:")
        assert result == 0


class TestCacheServiceIncr:
    """Tests for incr method."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_incr_increments_counter(self, mock_redis):
        """Incr increments counter and returns new value."""
        mock_redis.incr = AsyncMock(return_value=5)

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.incr("counter")

        assert result == 5
        mock_redis.incr.assert_called_once_with("test:counter")

    @pytest.mark.asyncio
    async def test_incr_returns_none_on_error(self, mock_redis):
        """Incr returns None on Redis error."""
        from redis.exceptions import RedisError

        mock_redis.incr = AsyncMock(side_effect=RedisError("Connection error"))

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.incr("counter")

        assert result is None

    @pytest.mark.asyncio
    async def test_incr_not_connected(self):
        """Incr returns None when not connected."""
        cache = CacheService(enabled=True)
        result = await cache.incr("counter")
        assert result is None


class TestCacheServiceGetRaw:
    """Tests for get_raw method."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_get_raw_returns_string(self, mock_redis):
        """Get raw returns raw string value."""
        mock_redis.get = AsyncMock(return_value="42")

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.get_raw("counter")

        assert result == "42"
        mock_redis.get.assert_called_once_with("test:counter")

    @pytest.mark.asyncio
    async def test_get_raw_returns_none_on_miss(self, mock_redis):
        """Get raw returns None on cache miss."""
        mock_redis.get = AsyncMock(return_value=None)

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.get_raw("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_raw_returns_none_on_error(self, mock_redis):
        """Get raw returns None on Redis error."""
        from redis.exceptions import RedisError

        mock_redis.get = AsyncMock(side_effect=RedisError("Connection error"))

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.get_raw("counter")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_raw_not_connected(self):
        """Get raw returns None when not connected."""
        cache = CacheService(enabled=True)
        result = await cache.get_raw("counter")
        assert result is None


class TestCacheServiceDeleteKeys:
    """Tests for delete_keys method."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_delete_keys_removes_multiple(self, mock_redis):
        """Delete keys removes multiple keys."""
        mock_redis.delete = AsyncMock(return_value=3)

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.delete_keys("key1", "key2", "key3")

        assert result == 3
        mock_redis.delete.assert_called_once_with(
            "test:key1", "test:key2", "test:key3"
        )

    @pytest.mark.asyncio
    async def test_delete_keys_empty_keys(self):
        """Delete keys returns 0 with no keys provided."""
        cache = CacheService(enabled=True)
        cache._connected = True
        result = await cache.delete_keys()
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_keys_returns_zero_on_error(self, mock_redis):
        """Delete keys returns 0 on Redis error."""
        from redis.exceptions import RedisError

        mock_redis.delete = AsyncMock(side_effect=RedisError("Connection error"))

        cache = CacheService(prefix="test:", enabled=True)
        cache._connected = True
        cache._client = mock_redis

        result = await cache.delete_keys("key1", "key2")

        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_keys_not_connected(self):
        """Delete keys returns 0 when not connected."""
        cache = CacheService(enabled=True)
        result = await cache.delete_keys("key1", "key2")
        assert result == 0


class TestCacheServiceConnect:
    """Tests for connect method with mocked Redis."""

    @pytest.mark.asyncio
    async def test_connect_already_connected(self):
        """Connect returns True if already connected."""
        cache = CacheService(enabled=True)
        cache._connected = True

        result = await cache.connect()

        assert result is True

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Connect establishes connection successfully."""
        from redis.asyncio import ConnectionPool, Redis

        with patch("app.services.cache.ConnectionPool") as mock_pool_class, \
             patch("app.services.cache.Redis") as mock_redis_class:
            mock_pool = MagicMock()
            mock_pool_class.from_url.return_value = mock_pool

            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_redis_class.return_value = mock_client

            cache = CacheService(enabled=True, redis_url="redis://localhost:6379/0")

            result = await cache.connect()

            assert result is True
            assert cache._connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Connect returns False on connection failure."""
        from redis.exceptions import RedisError

        with patch("app.services.cache.ConnectionPool") as mock_pool_class, \
             patch("app.services.cache.Redis") as mock_redis_class:
            mock_pool = MagicMock()
            mock_pool_class.from_url.return_value = mock_pool

            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(side_effect=RedisError("Connection refused"))
            mock_redis_class.return_value = mock_client

            cache = CacheService(enabled=True, redis_url="redis://localhost:6379/0")

            result = await cache.connect()

            assert result is False
            assert cache._connected is False
