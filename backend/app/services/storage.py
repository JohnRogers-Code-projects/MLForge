"""Storage service for managing model files.

This module provides an abstract interface for file storage operations
and a concrete implementation for local filesystem storage. The abstraction
allows for easy swapping to cloud storage (S3, GCS, Azure Blob) in the future.
"""

import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional

from app.config import settings


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class FileNotFoundError(StorageError):
    """Raised when a requested file doesn't exist."""

    pass


class StorageFullError(StorageError):
    """Raised when storage capacity is exceeded."""

    pass


class StorageService(ABC):
    """Abstract base class for storage operations.

    This interface defines the contract for all storage implementations.
    Implementations must handle:
    - File upload with size validation
    - File retrieval
    - File deletion
    - File existence checks
    - Hash computation for integrity verification
    """

    @abstractmethod
    async def save(
        self,
        file: BinaryIO,
        filename: str,
        max_size_bytes: Optional[int] = None,
    ) -> tuple[str, int, str]:
        """Save a file to storage.

        Args:
            file: File-like object to save
            filename: Name to save the file as
            max_size_bytes: Maximum allowed file size (None = use default)

        Returns:
            Tuple of (storage_path, file_size_bytes, file_hash)

        Raises:
            StorageFullError: If file exceeds size limit
            StorageError: If save operation fails
        """
        pass

    @abstractmethod
    async def get(self, path: str) -> bytes:
        """Retrieve a file from storage.

        Args:
            path: Storage path returned from save()

        Returns:
            File contents as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
            StorageError: If retrieval fails
        """
        pass

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete a file from storage.

        Args:
            path: Storage path to delete

        Returns:
            True if deleted, False if file didn't exist

        Raises:
            StorageError: If deletion fails
        """
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if a file exists in storage.

        Args:
            path: Storage path to check

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_path(self, path: str) -> Path:
        """Get the absolute filesystem path for a stored file.

        This is useful for operations that need direct file access,
        such as loading ONNX models into the runtime.

        Args:
            path: Storage path returned from save()

        Returns:
            Absolute Path object to the file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        pass

    @staticmethod
    def compute_hash(data: bytes) -> str:
        """Compute SHA-256 hash of data.

        Args:
            data: Bytes to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(data).hexdigest()


class LocalStorageService(StorageService):
    """Local filesystem storage implementation.

    Stores files in a configurable directory on the local filesystem.
    Files are organized by a simple flat structure with the original filename.

    Attributes:
        base_path: Root directory for file storage
        max_size_bytes: Default maximum file size in bytes
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        max_size_mb: Optional[int] = None,
    ):
        """Initialize local storage service.

        Args:
            base_path: Root directory for storage (default: from settings)
            max_size_mb: Default max file size in MB (default: from settings)
        """
        self.base_path = Path(base_path or settings.model_storage_path).resolve()
        self.max_size_bytes = (max_size_mb or settings.max_model_size_mb) * 1024 * 1024

        # Ensure storage directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        file: BinaryIO,
        filename: str,
        max_size_bytes: Optional[int] = None,
    ) -> tuple[str, int, str]:
        """Save a file to local filesystem.

        Reads the file in chunks to handle large files efficiently
        and compute hash during upload.
        """
        max_size = max_size_bytes or self.max_size_bytes

        # Sanitize filename to prevent directory traversal
        safe_filename = Path(filename).name
        if not safe_filename:
            raise StorageError("Invalid filename")

        file_path = self.base_path / safe_filename

        # Read file and compute hash simultaneously
        chunks = []
        total_size = 0
        hasher = hashlib.sha256()

        chunk_size = 8192  # 8KB chunks
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > max_size:
                raise StorageFullError(
                    f"File exceeds maximum size of {max_size / (1024*1024):.1f}MB"
                )

            chunks.append(chunk)
            hasher.update(chunk)

        file_hash = hasher.hexdigest()
        content = b"".join(chunks)

        # Write to filesystem
        try:
            file_path.write_bytes(content)
        except OSError as e:
            raise StorageError(f"Failed to write file: {e}") from e

        # Return relative path from base for portability
        return safe_filename, total_size, file_hash

    async def get(self, path: str) -> bytes:
        """Retrieve a file from local filesystem."""
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            return file_path.read_bytes()
        except OSError as e:
            raise StorageError(f"Failed to read file: {e}") from e

    async def delete(self, path: str) -> bool:
        """Delete a file from local filesystem."""
        file_path = self._resolve_path(path)

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            return True
        except OSError as e:
            raise StorageError(f"Failed to delete file: {e}") from e

    async def exists(self, path: str) -> bool:
        """Check if a file exists on local filesystem."""
        file_path = self._resolve_path(path)
        return file_path.exists() and file_path.is_file()

    async def get_path(self, path: str) -> Path:
        """Get the absolute path for a stored file."""
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return file_path

    def _resolve_path(self, path: str) -> Path:
        """Resolve a storage path to an absolute filesystem path.

        Includes security check to prevent directory traversal attacks.
        """
        # Sanitize to prevent directory traversal
        safe_path = Path(path).name
        resolved = (self.base_path / safe_path).resolve()

        # Security: ensure resolved path is within base_path
        if not str(resolved).startswith(str(self.base_path)):
            raise StorageError("Invalid path: directory traversal detected")

        return resolved


# Singleton instance for dependency injection
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get the storage service instance.

    Returns a singleton LocalStorageService by default.
    Can be overridden for testing or alternative implementations.
    """
    global _storage_service
    if _storage_service is None:
        _storage_service = LocalStorageService()
    return _storage_service


def set_storage_service(service: StorageService) -> None:
    """Set the storage service instance (for testing)."""
    global _storage_service
    _storage_service = service
