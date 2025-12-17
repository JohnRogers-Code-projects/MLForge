"""Model metadata caching utilities.

This module provides caching for model metadata to reduce database load
on frequently accessed models. Cache is automatically invalidated on
model updates and deletes.

Cache key patterns:
- model:{id} - Single model by ID
- model:name:{name}:version:{version} - Model by name and version
- model:name:{name}:latest - Latest version of a model
- model:name:{name}:versions - List of all versions
"""

import asyncio
import logging
from typing import Any, Optional

from app.config import settings
from app.services.cache import CacheService

logger = logging.getLogger(__name__)


# Cache key patterns
MODEL_KEY = "model:{id}"
MODEL_NAME_VERSION_KEY = "model:name:{name}:version:{version}"
MODEL_LATEST_KEY = "model:name:{name}:latest"
MODEL_VERSIONS_KEY = "model:name:{name}:versions"


class ModelCache:
    """Helper class for model-specific caching operations.

    Provides methods for caching and retrieving model metadata,
    with automatic key generation and TTL management.
    """

    def __init__(self, cache: CacheService):
        """Initialize with a cache service instance."""
        self.cache = cache
        self.model_ttl = settings.cache_model_ttl
        # list_ttl reserved for future use (PR 3.3 - caching version lists)
        self.list_ttl = settings.cache_model_list_ttl

    def _model_key(self, model_id: str) -> str:
        """Generate cache key for a model by ID."""
        return MODEL_KEY.format(id=model_id)

    def _name_version_key(self, name: str, version: str) -> str:
        """Generate cache key for a model by name and version."""
        return MODEL_NAME_VERSION_KEY.format(name=name, version=version)

    def _latest_key(self, name: str) -> str:
        """Generate cache key for latest version of a model."""
        return MODEL_LATEST_KEY.format(name=name)

    def _versions_key(self, name: str) -> str:
        """Generate cache key for versions list of a model."""
        return MODEL_VERSIONS_KEY.format(name=name)

    async def get_model(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get cached model by ID.

        Args:
            model_id: Model UUID

        Returns:
            Cached model data as dict, or None if not cached.
        """
        return await self.cache.get(self._model_key(model_id))

    async def set_model(self, model_id: str, model_data: dict[str, Any]) -> bool:
        """Cache model data.

        Args:
            model_id: Model UUID
            model_data: Model data to cache (should be JSON-serializable)

        Returns:
            True if cached successfully, False otherwise.
        """
        return await self.cache.set(
            self._model_key(model_id),
            model_data,
            ttl=self.model_ttl,
        )

    async def get_by_name_version(
        self, name: str, version: str
    ) -> Optional[dict[str, Any]]:
        """Get cached model by name and version.

        Args:
            name: Model name
            version: Model version

        Returns:
            Cached model data as dict, or None if not cached.
        """
        return await self.cache.get(self._name_version_key(name, version))

    async def set_by_name_version(
        self, name: str, version: str, model_data: dict[str, Any]
    ) -> bool:
        """Cache model data by name and version.

        Args:
            name: Model name
            version: Model version
            model_data: Model data to cache

        Returns:
            True if cached successfully, False otherwise.
        """
        return await self.cache.set(
            self._name_version_key(name, version),
            model_data,
            ttl=self.model_ttl,
        )

    async def invalidate_model(
        self,
        model_id: str,
        name: str,
        version: str,
        old_name: str | None = None,
        old_version: str | None = None,
    ) -> None:
        """Invalidate all cache entries for a model.

        Should be called when a model is updated or deleted.

        Args:
            model_id: Model UUID
            name: Current model name
            version: Current model version
            old_name: Previous model name (if changed during update)
            old_version: Previous model version (if changed during update)
        """
        # Build list of keys to invalidate
        keys_to_delete = [
            self._model_key(model_id),
            self._name_version_key(name, version),
            self._latest_key(name),
            self._versions_key(name),
        ]

        # If name or version changed, also invalidate old keys
        if old_name and old_name != name:
            keys_to_delete.extend([
                self._name_version_key(old_name, old_version or version),
                self._latest_key(old_name),
                self._versions_key(old_name),
            ])
        elif old_version and old_version != version:
            keys_to_delete.append(self._name_version_key(name, old_version))

        # Delete all keys in parallel for better performance
        await asyncio.gather(*[self.cache.delete(key) for key in keys_to_delete])

        logger.debug(f"Invalidated cache for model {model_id} ({name}:{version})")

    async def invalidate_by_name(self, name: str) -> None:
        """Invalidate all cache entries for a model name.

        Useful when any version of a model changes.

        Args:
            name: Model name
        """
        # Clear latest and versions list
        await self.cache.delete(self._latest_key(name))
        await self.cache.delete(self._versions_key(name))

        # Clear all versions - use pattern matching
        await self.cache.clear_prefix(f"model:name:{name}:")

        logger.debug(f"Invalidated all cache entries for model name '{name}'")


def model_to_cache_dict(model: Any) -> dict[str, Any]:
    """Convert a model ORM object to a cache-friendly dictionary.

    Args:
        model: SQLAlchemy model instance

    Returns:
        Dictionary representation suitable for JSON serialization and caching.
    """
    # Handle potentially None datetime fields defensively
    created_at = None
    if model.created_at is not None:
        created_at = model.created_at.isoformat()

    updated_at = None
    if model.updated_at is not None:
        updated_at = model.updated_at.isoformat()

    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "version": model.version,
        "status": model.status.value if hasattr(model.status, "value") else model.status,
        "file_path": model.file_path,
        "file_size_bytes": model.file_size_bytes,
        "file_hash": model.file_hash,
        "input_schema": model.input_schema,
        "output_schema": model.output_schema,
        "model_metadata": model.model_metadata,
        "created_at": created_at,
        "updated_at": updated_at,
    }
