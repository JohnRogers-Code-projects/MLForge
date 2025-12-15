"""Storage service for model file handling."""

import logging
import shutil
from pathlib import Path
from typing import BinaryIO

from app.config import settings

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when storage operations fail."""

    pass


class StorageService:
    """
    Service for handling model file storage.

    Manages file uploads, storage, and retrieval for ONNX models.
    Uses local filesystem storage with organized directory structure.
    """

    def __init__(self, base_path: str | None = None, max_size_bytes: int | None = None) -> None:
        """
        Initialize storage service.

        Args:
            base_path: Base directory for model storage.
                       Defaults to settings.model_storage_path.
            max_size_bytes: Optional custom maximum file size in bytes (for testing).
        """
        self._base_path = Path(base_path or settings.model_storage_path)
        if max_size_bytes is not None:
            self._max_size_bytes = max_size_bytes
        else:
            self._max_size_bytes = settings.max_model_size_mb * 1024 * 1024

    def initialize(self) -> None:
        """Create storage directory if it doesn't exist."""
        self._base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage initialized at {self._base_path}")

    def save_model(
        self,
        model_id: str,
        file: BinaryIO,
        filename: str,
    ) -> tuple[str, int]:
        """
        Save an uploaded model file.

        Args:
            model_id: Unique identifier for the model
            file: File-like object containing the model data
            filename: Original filename (used for extension)

        Returns:
            Tuple of (file_path, file_size_bytes)

        Raises:
            StorageError: If the file cannot be saved or exceeds size limit
        """
        # Ensure storage directory exists
        self.initialize()

        # Create model-specific directory
        model_dir = self._base_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename and create destination path
        safe_filename = self._sanitize_filename(filename)
        if not safe_filename.endswith(".onnx"):
            safe_filename = "model.onnx"

        dest_path = model_dir / safe_filename

        # Read and validate file size
        try:
            # Read file in chunks to handle large files
            total_size = 0
            temp_path = model_dir / f".{safe_filename}.tmp"

            with open(temp_path, "wb") as dest:
                while chunk := file.read(8192):
                    total_size += len(chunk)
                    if total_size > self._max_size_bytes:
                        temp_path.unlink(missing_ok=True)
                        raise StorageError(
                            f"File exceeds maximum size of {settings.max_model_size_mb}MB"
                        )
                    dest.write(chunk)

            # Move temp file to final location
            temp_path.rename(dest_path)
            logger.info(f"Saved model {model_id} to {dest_path} ({total_size} bytes)")

            return str(dest_path), total_size

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to save model file: {e}") from e

    def get_model_path(self, model_id: str, filename: str = "model.onnx") -> str | None:
        """
        Get the full path to a model file.

        Args:
            model_id: The model ID
            filename: The filename (default: model.onnx)

        Returns:
            Full path if file exists, None otherwise
        """
        path = self._base_path / model_id / filename
        return str(path) if path.exists() else None

    def get_model_path_from_full_path(self, file_path: str) -> str | None:
        """
        Validate and return a full file path if it exists.

        Args:
            file_path: Full path to check

        Returns:
            The path if it exists, None otherwise
        """
        path = Path(file_path)
        return str(path) if path.exists() else None

    def delete_model(self, model_id: str) -> bool:
        """
        Delete all files for a model.

        Args:
            model_id: The model ID to delete

        Returns:
            True if deleted, False if not found
        """
        model_dir = self._base_path / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model directory {model_dir}")
            return True
        return False

    def model_exists(self, model_id: str) -> bool:
        """Check if a model directory exists."""
        return (self._base_path / model_id).exists()

    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with model_count, total_size_bytes, storage_path
        """
        if not self._base_path.exists():
            return {
                "model_count": 0,
                "total_size_bytes": 0,
                "storage_path": str(self._base_path),
            }

        model_count = 0
        total_size = 0

        for model_dir in self._base_path.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                model_count += 1
                for file in model_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

        return {
            "model_count": model_count,
            "total_size_bytes": total_size,
            "storage_path": str(self._base_path),
        }

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to prevent path traversal attacks."""
        # Remove directory components
        filename = Path(filename).name
        # Remove dangerous characters
        safe_chars = "".join(c for c in filename if c.isalnum() or c in "._-")
        return safe_chars or "model.onnx"


# Singleton instance
storage_service = StorageService()
