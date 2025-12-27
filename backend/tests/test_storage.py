"""Tests for storage service."""

import io
from pathlib import Path

import pytest

from app.services.storage import (
    FileNotFoundError as StorageFileNotFoundError,
)
from app.services.storage import (
    LocalStorageService,
    StorageError,
    StorageFullError,
    StorageService,
)


@pytest.fixture
def storage_service(tmp_path: Path) -> LocalStorageService:
    """Create a storage service with a temporary directory."""
    return LocalStorageService(base_path=str(tmp_path), max_size_mb=1)


@pytest.fixture
def sample_file() -> io.BytesIO:
    """Create a sample file for testing."""
    content = b"Hello, this is test content for the storage service."
    return io.BytesIO(content)


@pytest.fixture
def large_file() -> io.BytesIO:
    """Create a file larger than 1MB for size limit testing."""
    # 1.5MB of data
    content = b"x" * (1024 * 1024 + 512 * 1024)
    return io.BytesIO(content)


class TestLocalStorageService:
    """Tests for LocalStorageService."""

    @pytest.mark.asyncio
    async def test_save_file(
        self, storage_service: LocalStorageService, sample_file: io.BytesIO
    ):
        """Test saving a file returns path, size, and hash."""
        path, size, file_hash = await storage_service.save(sample_file, "test.onnx")

        assert path == "test.onnx"
        assert size == len(sample_file.getvalue())  # Length of sample content
        assert len(file_hash) == 64  # SHA-256 hex length
        assert await storage_service.exists(path)

    @pytest.mark.asyncio
    async def test_save_file_size_limit(
        self, storage_service: LocalStorageService, large_file: io.BytesIO
    ):
        """Test that saving a file exceeding size limit raises error."""
        with pytest.raises(StorageFullError) as exc_info:
            await storage_service.save(large_file, "large.onnx")

        assert "exceeds maximum size" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_save_file_custom_size_limit(
        self, storage_service: LocalStorageService, sample_file: io.BytesIO
    ):
        """Test custom size limit per upload."""
        with pytest.raises(StorageFullError):
            # Set limit to 10 bytes, sample is 52 bytes
            await storage_service.save(sample_file, "test.onnx", max_size_bytes=10)

    @pytest.mark.asyncio
    async def test_save_sanitizes_filename(
        self, storage_service: LocalStorageService, sample_file: io.BytesIO
    ):
        """Test that directory traversal in filename is prevented."""
        # Attempt directory traversal
        path, _, _ = await storage_service.save(sample_file, "../../../etc/passwd")

        # Should be sanitized to just "passwd"
        assert path == "passwd"
        assert await storage_service.exists("passwd")

    @pytest.mark.asyncio
    async def test_save_invalid_filename(
        self, storage_service: LocalStorageService, sample_file: io.BytesIO
    ):
        """Test that empty filename raises error."""
        with pytest.raises(StorageError) as exc_info:
            await storage_service.save(sample_file, "")

        assert "Invalid filename" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_file(
        self, storage_service: LocalStorageService, sample_file: io.BytesIO
    ):
        """Test retrieving a saved file."""
        original_content = sample_file.getvalue()
        await storage_service.save(sample_file, "test.onnx")

        content = await storage_service.get("test.onnx")

        assert content == original_content

    @pytest.mark.asyncio
    async def test_get_nonexistent_file(self, storage_service: LocalStorageService):
        """Test that getting a nonexistent file raises error."""
        with pytest.raises(StorageFileNotFoundError):
            await storage_service.get("nonexistent.onnx")

    @pytest.mark.asyncio
    async def test_delete_file(
        self, storage_service: LocalStorageService, sample_file: io.BytesIO
    ):
        """Test deleting a file."""
        await storage_service.save(sample_file, "test.onnx")
        assert await storage_service.exists("test.onnx")

        result = await storage_service.delete("test.onnx")

        assert result is True
        assert not await storage_service.exists("test.onnx")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file(self, storage_service: LocalStorageService):
        """Test deleting a nonexistent file returns False."""
        result = await storage_service.delete("nonexistent.onnx")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(
        self, storage_service: LocalStorageService, sample_file: io.BytesIO
    ):
        """Test exists returns True for existing file."""
        await storage_service.save(sample_file, "test.onnx")

        assert await storage_service.exists("test.onnx")

    @pytest.mark.asyncio
    async def test_exists_false(self, storage_service: LocalStorageService):
        """Test exists returns False for nonexistent file."""
        assert not await storage_service.exists("nonexistent.onnx")

    @pytest.mark.asyncio
    async def test_get_path(
        self,
        storage_service: LocalStorageService,
        sample_file: io.BytesIO,
        tmp_path: Path,
    ):
        """Test getting the filesystem path of a stored file."""
        await storage_service.save(sample_file, "test.onnx")

        path = await storage_service.get_path("test.onnx")

        assert path.exists()
        assert path.is_absolute()
        assert path.name == "test.onnx"
        assert path.parent == tmp_path

    @pytest.mark.asyncio
    async def test_get_path_nonexistent(self, storage_service: LocalStorageService):
        """Test getting path for nonexistent file raises error."""
        with pytest.raises(StorageFileNotFoundError):
            await storage_service.get_path("nonexistent.onnx")

    @pytest.mark.asyncio
    async def test_directory_traversal_blocked(
        self, storage_service: LocalStorageService
    ):
        """Test that directory traversal attacks are blocked on retrieval."""
        # Even if someone tried to access outside base_path
        result = await storage_service.exists("../../../etc/passwd")

        # Should be sanitized and return False (file doesn't exist in base_path)
        assert result is False

    def test_compute_hash(self):
        """Test hash computation is deterministic."""
        data = b"test data for hashing"

        hash1 = LocalStorageService.compute_hash(data)
        hash2 = LocalStorageService.compute_hash(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars

    def test_compute_hash_different_data(self):
        """Test different data produces different hashes."""
        hash1 = LocalStorageService.compute_hash(b"data1")
        hash2 = LocalStorageService.compute_hash(b"data2")

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_save_computes_correct_hash(
        self, storage_service: LocalStorageService, sample_file: io.BytesIO
    ):
        """Test that saved file hash matches computed hash."""
        original_content = sample_file.getvalue()
        expected_hash = LocalStorageService.compute_hash(original_content)

        _, _, actual_hash = await storage_service.save(sample_file, "test.onnx")

        assert actual_hash == expected_hash

    @pytest.mark.asyncio
    async def test_creates_storage_directory(self, tmp_path: Path):
        """Test that storage service creates the base directory if it doesn't exist."""
        new_path = tmp_path / "new" / "nested" / "storage"
        assert not new_path.exists()

        LocalStorageService(base_path=str(new_path))

        assert new_path.exists()
        assert new_path.is_dir()


class TestStorageServiceInterface:
    """Tests to verify the interface contract."""

    def test_local_storage_implements_interface(self, tmp_path: Path):
        """Test that LocalStorageService properly implements StorageService."""
        service = LocalStorageService(base_path=str(tmp_path))

        assert isinstance(service, StorageService)
        assert hasattr(service, "save")
        assert hasattr(service, "get")
        assert hasattr(service, "delete")
        assert hasattr(service, "exists")
        assert hasattr(service, "get_path")
