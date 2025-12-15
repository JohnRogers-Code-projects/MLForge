"""Tests for ML model endpoints."""

import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_model(client: AsyncClient):
    """Test creating a new model."""
    model_data = {
        "name": "test-model",
        "description": "A test model",
        "version": "1.0.0",
    }
    response = await client.post("/api/v1/models", json=model_data)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == model_data["name"]
    assert data["description"] == model_data["description"]
    assert data["version"] == model_data["version"]
    assert data["status"] == "pending"
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_duplicate_model(client: AsyncClient):
    """Test creating a duplicate model returns conflict."""
    model_data = {
        "name": "duplicate-model",
        "version": "1.0.0",
    }
    # Create first model
    response = await client.post("/api/v1/models", json=model_data)
    assert response.status_code == 201

    # Try to create duplicate
    response = await client.post("/api/v1/models", json=model_data)
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_list_models(client: AsyncClient):
    """Test listing models with pagination."""
    # Create a few models
    for i in range(3):
        await client.post(
            "/api/v1/models",
            json={"name": f"list-test-model-{i}", "version": "1.0.0"},
        )

    response = await client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert len(data["items"]) >= 3


@pytest.mark.asyncio
async def test_get_model(client: AsyncClient):
    """Test getting a specific model."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "get-test-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Get the model
    response = await client.get(f"/api/v1/models/{model_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == model_id
    assert data["name"] == "get-test-model"


@pytest.mark.asyncio
async def test_get_nonexistent_model(client: AsyncClient):
    """Test getting a model that doesn't exist."""
    response = await client.get("/api/v1/models/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_model(client: AsyncClient):
    """Test updating a model."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "update-test-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Update the model
    update_data = {"description": "Updated description"}
    response = await client.patch(f"/api/v1/models/{model_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "Updated description"


@pytest.mark.asyncio
async def test_delete_model(client: AsyncClient):
    """Test deleting a model."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "delete-test-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Delete the model
    response = await client.delete(f"/api/v1/models/{model_id}")
    assert response.status_code == 204

    # Verify it's deleted
    get_response = await client.get(f"/api/v1/models/{model_id}")
    assert get_response.status_code == 404


# Upload endpoint tests


@pytest.fixture
def sample_onnx_file() -> io.BytesIO:
    """Create a sample ONNX file for testing.

    Note: This is not a valid ONNX model, just test bytes.
    ONNX validation will be added in PR 2.3.
    """
    content = b"fake-onnx-model-content-for-testing"
    return io.BytesIO(content)


@pytest.mark.asyncio
async def test_upload_model_file_success(client: AsyncClient, sample_onnx_file: io.BytesIO):
    """Test successful model file upload."""
    # Create a model first
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "upload-test-model", "version": "1.0.0"},
    )
    assert create_response.status_code == 201
    model_id = create_response.json()["id"]

    # Upload file
    files = {"file": ("model.onnx", sample_onnx_file, "application/octet-stream")}
    response = await client.post(f"/api/v1/models/{model_id}/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == model_id
    assert data["file_path"] == f"{model_id}.onnx"
    assert data["file_size_bytes"] == len(b"fake-onnx-model-content-for-testing")
    assert len(data["file_hash"]) == 64  # SHA-256 hex length
    assert data["status"] == "uploaded"
    assert data["message"] == "File uploaded successfully"


@pytest.mark.asyncio
async def test_upload_model_file_invalid_extension(client: AsyncClient):
    """Test upload with invalid file extension."""
    # Create a model first
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "invalid-ext-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Upload file with wrong extension
    content = io.BytesIO(b"not an onnx file")
    files = {"file": ("model.pkl", content, "application/octet-stream")}
    response = await client.post(f"/api/v1/models/{model_id}/upload", files=files)

    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_model_file_no_filename(client: AsyncClient):
    """Test upload without filename.

    Note: FastAPI/Starlette returns 422 (Unprocessable Entity) when the
    filename is empty, as it fails request validation before reaching
    our endpoint logic.
    """
    # Create a model first
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "no-filename-model", "version": "1.0.0"},
    )
    assert create_response.status_code == 201
    model_id = create_response.json()["id"]

    # Upload file without filename - FastAPI validates this as 422
    content = io.BytesIO(b"some content")
    files = {"file": ("", content, "application/octet-stream")}
    response = await client.post(f"/api/v1/models/{model_id}/upload", files=files)

    # FastAPI returns 422 for empty filename (request validation failure)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_model_file_already_uploaded(
    client: AsyncClient, sample_onnx_file: io.BytesIO
):
    """Test upload when model already has a file."""
    # Create a model first
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "duplicate-upload-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Upload file first time
    files = {"file": ("model.onnx", sample_onnx_file, "application/octet-stream")}
    response = await client.post(f"/api/v1/models/{model_id}/upload", files=files)
    assert response.status_code == 200

    # Try to upload again
    sample_onnx_file.seek(0)  # Reset stream position
    files = {"file": ("model2.onnx", sample_onnx_file, "application/octet-stream")}
    response = await client.post(f"/api/v1/models/{model_id}/upload", files=files)

    assert response.status_code == 409
    assert "already has an uploaded file" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_model_file_nonexistent_model(client: AsyncClient):
    """Test upload to nonexistent model."""
    content = io.BytesIO(b"some content")
    files = {"file": ("model.onnx", content, "application/octet-stream")}
    response = await client.post("/api/v1/models/nonexistent-id/upload", files=files)

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_updates_model_record(
    client: AsyncClient, sample_onnx_file: io.BytesIO
):
    """Test that upload updates the model record correctly."""
    # Create a model first
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "record-update-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]
    assert create_response.json()["status"] == "pending"
    assert create_response.json()["file_path"] is None

    # Upload file
    files = {"file": ("model.onnx", sample_onnx_file, "application/octet-stream")}
    await client.post(f"/api/v1/models/{model_id}/upload", files=files)

    # Verify model record was updated
    get_response = await client.get(f"/api/v1/models/{model_id}")
    data = get_response.json()

    assert data["status"] == "uploaded"
    assert data["file_path"] == f"{model_id}.onnx"
    assert data["file_size_bytes"] == len(b"fake-onnx-model-content-for-testing")


@pytest.mark.asyncio
async def test_upload_case_insensitive_extension(client: AsyncClient):
    """Test that file extension check is case insensitive."""
    # Create a model first
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "case-insensitive-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Upload file with uppercase extension
    content = io.BytesIO(b"fake content")
    files = {"file": ("model.ONNX", content, "application/octet-stream")}
    response = await client.post(f"/api/v1/models/{model_id}/upload", files=files)

    assert response.status_code == 200
