"""Cache service for Redis-based caching.

This module provides a Redis cache service with:
- Connection pooling (managed by redis-py)
- Health checks
- Graceful degradation when Redis is unavailable
- Key namespacing with configurable prefix
- TTL support with configurable defaults

The service is designed to fail gracefully - cache misses or Redis errors
will not crash the application, just result in cache bypasses.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for cache operations."""

    pass


class CacheService:
    """Redis cache service with graceful degradation.

    This service wraps Redis operations and handles failures gracefully.
    When Redis is unavailable, operations return None/False rather than
    raising exceptions, allowing the application to continue without caching.

    Attributes:
        enabled: Whether caching is enabled
        prefix: Key prefix for namespacing
        default_ttl: Default TTL in seconds
    """

    def __init__(
        self,
        redis_url: str | None = None,
        prefix: str | None = None,
        default_ttl: int | None = None,
        enabled: bool | None = None,
    ):
        """Initialize cache service.

        Args:
            redis_url: Redis connection URL (default: from settings)
            prefix: Key prefix for namespacing (default: from settings)
            default_ttl: Default TTL in seconds (default: from settings)
            enabled: Whether caching is enabled (default: from settings)
        """
        self.enabled = enabled if enabled is not None else settings.redis_enabled
        self.prefix = prefix or settings.cache_key_prefix
        self.default_ttl = default_ttl or settings.cache_ttl

        self._redis_url = redis_url or settings.redis_url
        self._pool: ConnectionPool | None = None
        self._client: Redis | None = None
        self._connected = False

    async def connect(self) -> bool:
        """Initialize Redis connection pool.

        Returns:
            True if connection successful, False otherwise.
        """
        if not self.enabled:
            logger.info("Cache disabled by configuration")
            return False

        if self._connected:
            return True

        try:
            self._pool = ConnectionPool.from_url(
                self._redis_url,
                max_connections=settings.redis_max_connections,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_socket_connect_timeout,
                retry_on_timeout=settings.redis_retry_on_timeout,
                health_check_interval=settings.redis_health_check_interval,
                decode_responses=True,  # Return strings instead of bytes
            )
            self._client = Redis(connection_pool=self._pool)

            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info("Redis connection established")
            return True

        except RedisError as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        self._connected = False
        self._client = None
        self._pool = None
        logger.info("Redis connection closed")

    async def health_check(self) -> dict[str, Any]:
        """Check Redis connection health.

        Returns:
            Dict with health status information.
        """
        if not self.enabled:
            return {
                "status": "disabled",
                "enabled": False,
                "connected": False,
            }

        if not self._connected or not self._client:
            return {
                "status": "disconnected",
                "enabled": True,
                "connected": False,
            }

        try:
            await self._client.ping()
            info = await self._client.info("server")
            return {
                "status": "healthy",
                "enabled": True,
                "connected": True,
                "redis_version": info.get("redis_version"),
                "uptime_seconds": info.get("uptime_in_seconds"),
            }
        except RedisError as e:
            logger.warning(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "enabled": True,
                "connected": False,
                "error": str(e),
            }

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self._connected and self._client is not None

    def make_key(self, key: str) -> str:
        """Create a namespaced cache key.

        Args:
            key: Raw cache key

        Returns:
            Prefixed cache key
        """
        return f"{self.prefix}{key}"

    def _make_key(self, key: str) -> str:
        """Alias for make_key (deprecated, use make_key instead)."""
        return self.make_key(key)

    async def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/error.
        """
        if not self._connected or not self._client:
            return None

        try:
            value = await self._client.get(self.make_key(key))
            if value is None:
                return None

            # Try to deserialize JSON, fall back to raw string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        except RedisError as e:
            logger.warning(f"Cache get failed for key '{key}': {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized if not a string)
            ttl: TTL in seconds (default: use default_ttl)

        Returns:
            True if successful, False otherwise.
        """
        if not self._connected or not self._client:
            return False

        ttl = ttl if ttl is not None else self.default_ttl

        try:
            # Serialize non-string values as JSON
            if isinstance(value, str):
                serialized = value
            else:
                serialized = json.dumps(value)

            await self._client.set(
                self.make_key(key),
                serialized,
                ex=ttl,
            )
            return True

        except (RedisError, TypeError, json.JSONDecodeError) as e:
            logger.warning(f"Cache set failed for key '{key}': {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found/error.
        """
        if not self._connected or not self._client:
            return False

        try:
            result = await self._client.delete(self.make_key(key))
            return result > 0

        except RedisError as e:
            logger.warning(f"Cache delete failed for key '{key}': {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise.
        """
        if not self._connected or not self._client:
            return False

        try:
            result = await self._client.exists(self.make_key(key))
            return result > 0

        except RedisError as e:
            logger.warning(f"Cache exists check failed for key '{key}': {e}")
            return False

    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys matching a prefix.

        Useful for invalidating related cache entries.

        Args:
            prefix: Key prefix to match (will be added to global prefix)

        Returns:
            Number of keys deleted, or 0 on error.
        """
        if not self._connected or not self._client:
            return 0

        try:
            pattern = f"{self.make_key(prefix)}*"
            keys = []
            async for key in self._client.scan_iter(match=pattern, count=100):
                keys.append(key)

            if keys:
                return await self._client.delete(*keys)
            return 0

        except RedisError as e:
            logger.warning(f"Cache clear_prefix failed for '{prefix}': {e}")
            return 0

    async def incr(self, key: str) -> int | None:
        """Increment a counter in cache.

        Creates the key with value 1 if it doesn't exist.

        Args:
            key: Cache key

        Returns:
            New value after increment, or None on error.
        """
        if not self._connected or not self._client:
            return None

        try:
            return await self._client.incr(self.make_key(key))
        except RedisError as e:
            logger.warning(f"Cache incr failed for key '{key}': {e}")
            return None

    async def get_raw(self, key: str) -> str | None:
        """Get a raw string value from cache without JSON deserialization.

        Useful for counters and other non-JSON values.

        Args:
            key: Cache key

        Returns:
            Raw string value or None if not found/error.
        """
        if not self._connected or not self._client:
            return None

        try:
            return await self._client.get(self.make_key(key))
        except RedisError as e:
            logger.warning(f"Cache get_raw failed for key '{key}': {e}")
            return None

    async def delete_keys(self, *keys: str) -> int:
        """Delete multiple keys from cache.

        Args:
            *keys: Cache keys to delete

        Returns:
            Number of keys deleted, or 0 on error.
        """
        if not self._connected or not self._client or not keys:
            return 0

        try:
            full_keys = [self.make_key(k) for k in keys]
            return await self._client.delete(*full_keys)
        except RedisError as e:
            logger.warning(f"Cache delete_keys failed: {e}")
            return 0

    async def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics for monitoring.

        Returns:
            Dict with cache metrics including hit/miss counts and hit rate.
        """
        if not self.enabled:
            return {"connected": False, "enabled": False}

        if not self._connected or not self._client:
            return {"connected": False, "enabled": True}

        try:
            # Get cache hit/miss metrics from Redis INFO
            info = await self._client.info("stats")
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            hit_rate = ((hits / total) * 100) if total > 0 else 0

            return {
                "connected": True,
                "enabled": True,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hit_rate, 2),
            }
        except RedisError as e:
            logger.warning(f"Failed to get cache metrics: {e}")
            return {"connected": False, "enabled": True, "error": str(e)}

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[Any]],
        ttl: int | None = None,
    ) -> Any | None:
        """Get a value from cache, or compute and cache it.

        This is a convenience method that combines get and set.
        If the key is not in cache, the factory function is called
        to compute the value, which is then cached.

        Args:
            key: Cache key
            factory: Async callable that returns the value to cache
            ttl: TTL in seconds (default: use default_ttl)

        Returns:
            Cached or computed value, or None on error.
        """
        # Try cache first
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        try:
            value = await factory()
        except Exception as e:
            logger.error(f"Cache factory failed for key '{key}': {e}")
            return None

        # Cache it (ignore failures)
        await self.set(key, value, ttl)
        return value


# Singleton instance for dependency injection
_cache_service: CacheService | None = None


async def get_cache_service() -> CacheService:
    """Get the cache service instance.

    Returns a singleton CacheService. Initializes connection on first call.
    """
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.connect()
    return _cache_service


async def close_cache_service() -> None:
    """Close the cache service connection."""
    global _cache_service
    if _cache_service is not None:
        await _cache_service.disconnect()
        _cache_service = None


def set_cache_service(service: CacheService) -> None:
    """Set the cache service instance (for testing)."""
    global _cache_service
    _cache_service = service
