"""Tests for ML model endpoints."""

import io

import onnx
import pytest
from httpx import AsyncClient

from tests.conftest import create_simple_onnx_model


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
    f = io.BytesIO(content)
    f.seek(0)
    return f


@pytest.mark.asyncio
async def test_upload_model_file_success(
    client: AsyncClient, sample_onnx_file: io.BytesIO
):
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
    assert create_response.status_code == 201
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


# Validation endpoint tests


@pytest.fixture
def valid_onnx_file() -> io.BytesIO:
    """Create a valid ONNX model file for testing."""
    model = create_simple_onnx_model()
    buffer = io.BytesIO()
    onnx.save(model, buffer)
    buffer.seek(0)
    return buffer


@pytest.mark.asyncio
async def test_validate_model_success(client: AsyncClient, valid_onnx_file: io.BytesIO):
    """Test successful model validation."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "validate-test-model", "version": "1.0.0"},
    )
    assert create_response.status_code == 201
    model_id = create_response.json()["id"]

    # Upload valid ONNX file
    files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
    upload_response = await client.post(
        f"/api/v1/models/{model_id}/upload", files=files
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["status"] == "uploaded"

    # Validate the model
    response = await client.post(f"/api/v1/models/{model_id}/validate")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == model_id
    assert data["valid"] is True
    assert data["status"] == "ready"
    assert data["error_message"] is None
    assert data["message"] == "Model validated successfully"

    # Check input schema
    assert len(data["input_schema"]) == 1
    assert data["input_schema"][0]["name"] == "input"
    assert data["input_schema"][0]["dtype"] == "float32"
    assert data["input_schema"][0]["shape"] == [None, 10]

    # Check output schema
    assert len(data["output_schema"]) == 1
    assert data["output_schema"][0]["name"] == "output"
    assert data["output_schema"][0]["dtype"] == "float32"

    # Check metadata
    assert data["model_metadata"] is not None
    assert "providers" in data["model_metadata"]


@pytest.mark.asyncio
async def test_validate_model_updates_record(
    client: AsyncClient, valid_onnx_file: io.BytesIO
):
    """Test that validation updates the model record correctly."""
    # Create and upload model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "validate-update-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
    await client.post(f"/api/v1/models/{model_id}/upload", files=files)

    # Validate
    await client.post(f"/api/v1/models/{model_id}/validate")

    # Get model and verify record was updated
    get_response = await client.get(f"/api/v1/models/{model_id}")
    data = get_response.json()

    assert data["status"] == "ready"
    assert data["input_schema"] is not None
    assert data["output_schema"] is not None
    assert data["model_metadata"] is not None


@pytest.mark.asyncio
async def test_validate_model_invalid_onnx(client: AsyncClient):
    """Test validation with invalid ONNX file."""
    # Create and upload model with invalid ONNX content
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "invalid-validate-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Upload invalid file (not a real ONNX model)
    content = io.BytesIO(b"this is not a valid onnx model")
    files = {"file": ("model.onnx", content, "application/octet-stream")}
    await client.post(f"/api/v1/models/{model_id}/upload", files=files)

    # Validate - should fail
    response = await client.post(f"/api/v1/models/{model_id}/validate")
    assert response.status_code == 200

    data = response.json()
    assert data["valid"] is False
    assert data["status"] == "error"
    assert data["error_message"] is not None
    assert data["message"] == "Model validation failed"


@pytest.mark.asyncio
async def test_validate_model_no_file(client: AsyncClient):
    """Test validation when model has no uploaded file."""
    # Create model without uploading file
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "no-file-validate-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Try to validate without uploading
    response = await client.post(f"/api/v1/models/{model_id}/validate")
    assert response.status_code == 400
    assert "Upload a file first" in response.json()["detail"]


@pytest.mark.asyncio
async def test_validate_model_wrong_status(
    client: AsyncClient, valid_onnx_file: io.BytesIO
):
    """Test validation when model is already validated (READY status)."""
    # Create, upload, and validate model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "already-valid-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
    await client.post(f"/api/v1/models/{model_id}/upload", files=files)
    await client.post(f"/api/v1/models/{model_id}/validate")

    # Try to validate again
    response = await client.post(f"/api/v1/models/{model_id}/validate")
    assert response.status_code == 409
    assert "cannot be validated" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_validate_model_can_revalidate_after_error(client: AsyncClient):
    """Test that models in ERROR status can be revalidated."""
    # Create and upload invalid model to get ERROR status
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "revalidate-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    content = io.BytesIO(b"invalid onnx")
    files = {"file": ("model.onnx", content, "application/octet-stream")}
    await client.post(f"/api/v1/models/{model_id}/upload", files=files)
    await client.post(f"/api/v1/models/{model_id}/validate")

    # Verify status is ERROR
    get_response = await client.get(f"/api/v1/models/{model_id}")
    assert get_response.json()["status"] == "error"

    # Try to revalidate - should be allowed (even though it will fail again)
    response = await client.post(f"/api/v1/models/{model_id}/validate")
    # Should return 200 (not 409) because ERROR status allows revalidation
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_validate_nonexistent_model(client: AsyncClient):
    """Test validation of nonexistent model."""
    response = await client.post("/api/v1/models/nonexistent-id/validate")
    assert response.status_code == 404


# CRUD operation tests


class TestModelCRUDOperations:
    """Direct unit tests for MLModel CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_by_name(self, client: AsyncClient):
        """Test getting a model by name."""
        from app.crud import model_crud
        from app.database import get_db

        # Create a model
        await client.post(
            "/api/v1/models",
            json={"name": "crud-get-by-name", "version": "1.0.0"},
        )

        async for session in client._transport.app.dependency_overrides[get_db]():
            model = await model_crud.get_by_name(session, name="crud-get-by-name")
            assert model is not None
            assert model.name == "crud-get-by-name"
            break

    @pytest.mark.asyncio
    async def test_get_by_name_not_found(self, client: AsyncClient):
        """Test getting a nonexistent model by name."""
        from app.crud import model_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            model = await model_crud.get_by_name(session, name="nonexistent-model-name")
            assert model is None
            break

    @pytest.mark.asyncio
    async def test_get_by_name_and_version(self, client: AsyncClient):
        """Test getting a model by name and version."""
        from app.crud import model_crud
        from app.database import get_db

        # Create models with different versions
        await client.post(
            "/api/v1/models",
            json={"name": "crud-by-name-version", "version": "1.0.0"},
        )
        await client.post(
            "/api/v1/models",
            json={"name": "crud-by-name-version", "version": "2.0.0"},
        )

        async for session in client._transport.app.dependency_overrides[get_db]():
            model = await model_crud.get_by_name_and_version(
                session, name="crud-by-name-version", version="2.0.0"
            )
            assert model is not None
            assert model.version == "2.0.0"
            break

    @pytest.mark.asyncio
    async def test_get_ready_models(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting models with READY status."""
        from app.crud import model_crud
        from app.database import get_db

        # Create and make a model ready
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "crud-ready-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Upload and validate to make it ready
        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        async for session in client._transport.app.dependency_overrides[get_db]():
            ready_models = await model_crud.get_ready_models(session)
            assert any(m.id == model_id for m in ready_models)
            break

    @pytest.mark.asyncio
    async def test_update_status(self, client: AsyncClient):
        """Test updating model status."""
        from app.crud import model_crud
        from app.database import get_db
        from app.models.ml_model import ModelStatus

        # Create a model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "crud-update-status", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        async for session in client._transport.app.dependency_overrides[get_db]():
            updated = await model_crud.update_status(
                session, model_id=model_id, status=ModelStatus.READY
            )
            assert updated is not None
            assert updated.status == ModelStatus.READY
            break

    @pytest.mark.asyncio
    async def test_update_status_nonexistent(self, client: AsyncClient):
        """Test updating status of nonexistent model returns None."""
        from app.crud import model_crud
        from app.database import get_db
        from app.models.ml_model import ModelStatus

        async for session in client._transport.app.dependency_overrides[get_db]():
            result = await model_crud.update_status(
                session,
                model_id="00000000-0000-0000-0000-000000000000",
                status=ModelStatus.READY,
            )
            assert result is None
            break

    @pytest.mark.asyncio
    async def test_get_versions_by_name(self, client: AsyncClient):
        """Test getting all versions of a model."""
        from app.crud import model_crud
        from app.database import get_db

        # Create multiple versions
        for version in ["1.0.0", "2.0.0", "1.5.0"]:
            await client.post(
                "/api/v1/models",
                json={"name": "crud-versions", "version": version},
            )

        async for session in client._transport.app.dependency_overrides[get_db]():
            versions = await model_crud.get_versions_by_name(
                session, name="crud-versions"
            )
            assert len(versions) == 3
            # Should be sorted newest first (2.0.0, 1.5.0, 1.0.0)
            assert versions[0].version == "2.0.0"
            assert versions[1].version == "1.5.0"
            assert versions[2].version == "1.0.0"
            break

    @pytest.mark.asyncio
    async def test_get_latest_by_name(self, client: AsyncClient):
        """Test getting the latest version of a model."""
        from app.crud import model_crud
        from app.database import get_db

        # Create multiple versions
        for version in ["1.0.0", "3.0.0", "2.0.0"]:
            await client.post(
                "/api/v1/models",
                json={"name": "crud-latest", "version": version},
            )

        async for session in client._transport.app.dependency_overrides[get_db]():
            latest = await model_crud.get_latest_by_name(session, name="crud-latest")
            assert latest is not None
            assert latest.version == "3.0.0"
            break

    @pytest.mark.asyncio
    async def test_get_latest_by_name_not_found(self, client: AsyncClient):
        """Test getting latest version of nonexistent model."""
        from app.crud import model_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            latest = await model_crud.get_latest_by_name(
                session, name="nonexistent-crud-latest"
            )
            assert latest is None
            break

    @pytest.mark.asyncio
    async def test_get_latest_by_name_ready_only(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting the latest READY version of a model."""
        from app.crud import model_crud
        from app.database import get_db

        # Create v1.0.0 and make it ready
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "crud-latest-ready", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]
        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Create v2.0.0 but leave it pending
        await client.post(
            "/api/v1/models",
            json={"name": "crud-latest-ready", "version": "2.0.0"},
        )

        async for session in client._transport.app.dependency_overrides[get_db]():
            # Without ready_only, should get 2.0.0
            latest = await model_crud.get_latest_by_name(
                session, name="crud-latest-ready"
            )
            assert latest.version == "2.0.0"

            # With ready_only, should get 1.0.0
            latest_ready = await model_crud.get_latest_by_name(
                session, name="crud-latest-ready", ready_only=True
            )
            assert latest_ready.version == "1.0.0"
            break

    @pytest.mark.asyncio
    async def test_count_versions_by_name(self, client: AsyncClient):
        """Test counting versions of a model."""
        from app.crud import model_crud
        from app.database import get_db

        # Create multiple versions
        for version in ["1.0.0", "2.0.0", "3.0.0"]:
            await client.post(
                "/api/v1/models",
                json={"name": "crud-count-versions", "version": version},
            )

        async for session in client._transport.app.dependency_overrides[get_db]():
            count = await model_crud.count_versions_by_name(
                session, name="crud-count-versions"
            )
            assert count == 3
            break

    @pytest.mark.asyncio
    async def test_get_unique_model_names(self, client: AsyncClient):
        """Test getting unique model names."""
        from app.crud import model_crud
        from app.database import get_db

        # Create models with different names
        for name in ["crud-unique-a", "crud-unique-b", "crud-unique-a"]:
            await client.post(
                "/api/v1/models",
                json={"name": name, "version": f"1.0.{hash(name) % 100}"},
            )

        async for session in client._transport.app.dependency_overrides[get_db]():
            names = await model_crud.get_unique_model_names(session)
            # Should contain both unique names
            assert "crud-unique-a" in names
            assert "crud-unique-b" in names
            break
