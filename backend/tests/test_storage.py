"""Tests for the storage service."""

import io
from pathlib import Path

import pytest

from app.services.storage_service import StorageError, StorageService


class TestStorageService:
    """Tests for the storage service."""

    def test_initialize_creates_directory(self, temp_storage_dir: Path):
        """Test that initialize creates the storage directory."""
        service = StorageService(str(temp_storage_dir / "new_dir"))
        service.initialize()
        assert (temp_storage_dir / "new_dir").exists()

    def test_save_model(self, temp_storage_dir: Path):
        """Test saving a model file."""
        service = StorageService(str(temp_storage_dir))
        model_id = "test-model-1"
        content = b"fake model content"

        file_path, file_size = service.save_model(
            model_id=model_id,
            file=io.BytesIO(content),
            filename="model.onnx",
        )

        assert Path(file_path).exists()
        assert file_size == len(content)
        assert model_id in file_path

    def test_save_model_sanitizes_filename(self, temp_storage_dir: Path):
        """Test that dangerous filenames are sanitized."""
        service = StorageService(str(temp_storage_dir))
        model_id = "test-sanitize"
        content = b"content"

        # Try path traversal attack
        file_path, _ = service.save_model(
            model_id=model_id,
            file=io.BytesIO(content),
            filename="../../../etc/passwd.onnx",
        )

        # Should be safely stored in model directory
        assert model_id in file_path
        assert ".." not in file_path

    def test_save_model_enforces_onnx_extension(self, temp_storage_dir: Path):
        """Test that non-ONNX filenames get default name."""
        service = StorageService(str(temp_storage_dir))
        model_id = "test-extension"
        content = b"content"

        file_path, _ = service.save_model(
            model_id=model_id,
            file=io.BytesIO(content),
            filename="model.txt",
        )

        assert file_path.endswith("model.onnx")

    def test_get_model_path(self, temp_storage_dir: Path):
        """Test getting model path."""
        service = StorageService(str(temp_storage_dir))
        model_id = "test-get-path"
        content = b"content"

        # Save model
        saved_path, _ = service.save_model(
            model_id=model_id,
            file=io.BytesIO(content),
            filename="model.onnx",
        )

        # Get path
        retrieved_path = service.get_model_path(model_id, "model.onnx")
        assert retrieved_path == saved_path

        # Non-existent model returns None
        none_path = service.get_model_path("nonexistent")
        assert none_path is None

    def test_delete_model(self, temp_storage_dir: Path):
        """Test deleting a model."""
        service = StorageService(str(temp_storage_dir))
        model_id = "test-delete"
        content = b"content"

        # Save model
        service.save_model(
            model_id=model_id,
            file=io.BytesIO(content),
            filename="model.onnx",
        )

        # Delete
        result = service.delete_model(model_id)
        assert result is True
        assert not service.model_exists(model_id)

        # Delete again returns False
        result = service.delete_model(model_id)
        assert result is False

    def test_model_exists(self, temp_storage_dir: Path):
        """Test checking if model exists."""
        service = StorageService(str(temp_storage_dir))
        model_id = "test-exists"

        assert not service.model_exists(model_id)

        service.save_model(
            model_id=model_id,
            file=io.BytesIO(b"content"),
            filename="model.onnx",
        )

        assert service.model_exists(model_id)

    def test_get_storage_stats(self, temp_storage_dir: Path):
        """Test getting storage statistics."""
        service = StorageService(str(temp_storage_dir))

        # Initially empty
        stats = service.get_storage_stats()
        assert stats["model_count"] == 0
        assert stats["total_size_bytes"] == 0

        # Add some models
        service.save_model("model-1", io.BytesIO(b"a" * 100), "m.onnx")
        service.save_model("model-2", io.BytesIO(b"b" * 200), "m.onnx")

        stats = service.get_storage_stats()
        assert stats["model_count"] == 2
        assert stats["total_size_bytes"] == 300

    def test_save_model_size_limit(self, temp_storage_dir: Path):
        """Test that size limit is enforced."""
        # Create service with 1KB limit
        service = StorageService(str(temp_storage_dir))
        service._max_size_bytes = 1024

        # Try to save file larger than limit
        large_content = b"x" * 2048

        with pytest.raises(StorageError, match="exceeds maximum size"):
            service.save_model(
                model_id="too-large",
                file=io.BytesIO(large_content),
                filename="model.onnx",
            )
