"""Tests for ONNX service and model upload/inference endpoints."""

import io
from pathlib import Path

import numpy as np
import pytest
from httpx import AsyncClient

from app.services.onnx_service import (
    ONNXInferenceError,
    ONNXService,
    ONNXValidationError,
)


class TestONNXService:
    """Tests for the ONNX service."""

    def test_validate_model_success(self, simple_onnx_model: Path, clean_onnx_service):
        """Test successful model validation."""
        service = ONNXService()
        result = service.validate_model(str(simple_onnx_model))

        assert "input_schema" in result
        assert "output_schema" in result
        assert "file_hash" in result
        assert "file_size_bytes" in result

        # Check input schema
        assert len(result["input_schema"]) == 2
        input_names = [i["name"] for i in result["input_schema"]]
        assert "input_a" in input_names
        assert "input_b" in input_names

        # Check output schema
        assert len(result["output_schema"]) == 1
        assert result["output_schema"][0]["name"] == "output"

    def test_validate_model_not_found(self, clean_onnx_service):
        """Test validation fails for non-existent file."""
        service = ONNXService()
        with pytest.raises(ONNXValidationError, match="not found"):
            service.validate_model("/nonexistent/path/model.onnx")

    def test_validate_model_invalid_extension(
        self, test_model_dir: Path, clean_onnx_service
    ):
        """Test validation fails for non-ONNX file."""
        # Create a file with wrong extension
        bad_file = test_model_dir / "model.txt"
        bad_file.write_text("not an onnx model")

        service = ONNXService()
        with pytest.raises(ONNXValidationError, match="Invalid file extension"):
            service.validate_model(str(bad_file))

    def test_load_and_predict(self, simple_onnx_model: Path, clean_onnx_service):
        """Test loading a model and running inference."""
        service = ONNXService()
        model_id = "test-model-1"

        # Load model
        service.load_model(model_id, str(simple_onnx_model))
        assert service.is_loaded(model_id)

        # Run inference
        input_data = {
            "input_a": [[1.0, 2.0, 3.0]],
            "input_b": [[4.0, 5.0, 6.0]],
        }
        output, inference_time = service.predict(model_id, input_data)

        # Check output
        assert "output" in output
        expected = [[5.0, 7.0, 9.0]]
        np.testing.assert_array_almost_equal(output["output"], expected)
        assert inference_time > 0

    def test_predict_missing_input(self, simple_onnx_model: Path, clean_onnx_service):
        """Test inference fails with missing input."""
        service = ONNXService()
        model_id = "test-model-2"
        service.load_model(model_id, str(simple_onnx_model))

        # Missing input_b
        input_data = {"input_a": [[1.0, 2.0, 3.0]]}
        with pytest.raises(ONNXInferenceError, match="Missing required input"):
            service.predict(model_id, input_data)

    def test_warmup_model(self, simple_onnx_model: Path, clean_onnx_service):
        """Test model warmup."""
        service = ONNXService()
        model_id = "test-warmup"

        warmup_time = service.warmup_model(model_id, str(simple_onnx_model))

        assert warmup_time > 0
        assert service.is_loaded(model_id)

    def test_unload_model(self, simple_onnx_model: Path, clean_onnx_service):
        """Test unloading a model."""
        service = ONNXService()
        model_id = "test-unload"

        service.load_model(model_id, str(simple_onnx_model))
        assert service.is_loaded(model_id)

        result = service.unload_model(model_id)
        assert result is True
        assert not service.is_loaded(model_id)

        # Unloading again returns False
        result = service.unload_model(model_id)
        assert result is False

    def test_get_loaded_models(self, simple_onnx_model: Path, clean_onnx_service):
        """Test getting list of loaded models."""
        service = ONNXService()

        assert len(service.get_loaded_models()) == 0

        service.load_model("model-1", str(simple_onnx_model))
        service.load_model("model-2", str(simple_onnx_model))

        loaded = service.get_loaded_models()
        assert len(loaded) == 2
        assert "model-1" in loaded
        assert "model-2" in loaded


@pytest.mark.asyncio
class TestModelUploadEndpoint:
    """Tests for the model upload endpoint."""

    async def test_upload_model_success(
        self,
        client: AsyncClient,
        simple_onnx_model: Path,
        temp_storage_dir: Path,
        clean_onnx_service,
    ):
        """Test successful model upload."""
        with open(simple_onnx_model, "rb") as f:
            model_content = f.read()

        response = await client.post(
            "/api/v1/models/upload",
            files={"file": ("test_model.onnx", io.BytesIO(model_content))},
            data={"name": "upload-test-model", "version": "1.0.0"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "upload-test-model"
        assert data["version"] == "1.0.0"
        assert data["status"] == "ready"
        assert "input_schema" in data
        assert "output_schema" in data
        assert data["file_size_bytes"] > 0

    async def test_upload_duplicate_model(
        self,
        client: AsyncClient,
        simple_onnx_model: Path,
        temp_storage_dir: Path,
        clean_onnx_service,
    ):
        """Test uploading a duplicate model returns conflict."""
        with open(simple_onnx_model, "rb") as f:
            model_content = f.read()

        # Upload first time
        response = await client.post(
            "/api/v1/models/upload",
            files={"file": ("test.onnx", io.BytesIO(model_content))},
            data={"name": "dup-model", "version": "1.0.0"},
        )
        assert response.status_code == 201

        # Upload again with same name and version
        response = await client.post(
            "/api/v1/models/upload",
            files={"file": ("test.onnx", io.BytesIO(model_content))},
            data={"name": "dup-model", "version": "1.0.0"},
        )
        assert response.status_code == 409

    async def test_upload_invalid_extension(
        self,
        client: AsyncClient,
        temp_storage_dir: Path,
    ):
        """Test uploading a file with invalid extension."""
        response = await client.post(
            "/api/v1/models/upload",
            files={"file": ("model.txt", io.BytesIO(b"not an onnx model"))},
            data={"name": "bad-ext-model", "version": "1.0.0"},
        )
        assert response.status_code == 400
        assert "onnx" in response.json()["detail"].lower()

    async def test_upload_invalid_onnx(
        self,
        client: AsyncClient,
        temp_storage_dir: Path,
        clean_onnx_service,
    ):
        """Test uploading an invalid ONNX file."""
        response = await client.post(
            "/api/v1/models/upload",
            files={"file": ("model.onnx", io.BytesIO(b"not valid onnx content"))},
            data={"name": "invalid-onnx", "version": "1.0.0"},
        )
        assert response.status_code == 422


@pytest.mark.asyncio
class TestInferenceEndpoint:
    """Tests for the inference endpoint."""

    async def test_inference_success(
        self,
        client: AsyncClient,
        simple_onnx_model: Path,
        temp_storage_dir: Path,
        clean_onnx_service,
    ):
        """Test successful inference."""
        # First upload a model
        with open(simple_onnx_model, "rb") as f:
            model_content = f.read()

        upload_response = await client.post(
            "/api/v1/models/upload",
            files={"file": ("test.onnx", io.BytesIO(model_content))},
            data={"name": "inference-test", "version": "1.0.0"},
        )
        assert upload_response.status_code == 201
        model_id = upload_response.json()["id"]

        # Run inference
        input_data = {
            "input_data": {
                "input_a": [[1.0, 2.0, 3.0]],
                "input_b": [[4.0, 5.0, 6.0]],
            }
        }
        response = await client.post(
            f"/api/v1/predictions/models/{model_id}/predict",
            json=input_data,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["model_id"] == model_id
        assert "output_data" in data
        assert data["inference_time_ms"] > 0

        # Verify output
        expected = [[5.0, 7.0, 9.0]]
        np.testing.assert_array_almost_equal(
            data["output_data"]["output"], expected
        )

    async def test_inference_model_not_ready(
        self,
        client: AsyncClient,
    ):
        """Test inference fails when model is not ready."""
        # Create a model without uploading (PENDING status)
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "not-ready-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Try to run inference
        response = await client.post(
            f"/api/v1/predictions/models/{model_id}/predict",
            json={"input_data": {}},
        )

        assert response.status_code == 400
        assert "not ready" in response.json()["detail"].lower()

    async def test_inference_missing_input(
        self,
        client: AsyncClient,
        simple_onnx_model: Path,
        temp_storage_dir: Path,
        clean_onnx_service,
    ):
        """Test inference fails with missing input."""
        # Upload model
        with open(simple_onnx_model, "rb") as f:
            model_content = f.read()

        upload_response = await client.post(
            "/api/v1/models/upload",
            files={"file": ("test.onnx", io.BytesIO(model_content))},
            data={"name": "missing-input-test", "version": "1.0.0"},
        )
        model_id = upload_response.json()["id"]

        # Run inference with missing input
        response = await client.post(
            f"/api/v1/predictions/models/{model_id}/predict",
            json={"input_data": {"input_a": [[1.0, 2.0, 3.0]]}},  # Missing input_b
        )

        assert response.status_code == 422


@pytest.mark.asyncio
class TestModelSchemaEndpoint:
    """Tests for the model schema endpoint."""

    async def test_get_model_schema(
        self,
        client: AsyncClient,
        simple_onnx_model: Path,
        temp_storage_dir: Path,
        clean_onnx_service,
    ):
        """Test getting model schema."""
        # Upload model
        with open(simple_onnx_model, "rb") as f:
            model_content = f.read()

        upload_response = await client.post(
            "/api/v1/models/upload",
            files={"file": ("test.onnx", io.BytesIO(model_content))},
            data={"name": "schema-test", "version": "1.0.0"},
        )
        model_id = upload_response.json()["id"]

        # Get schema
        response = await client.get(f"/api/v1/models/{model_id}/schema")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_id
        assert data["input_schema"] is not None
        assert data["output_schema"] is not None


@pytest.mark.asyncio
class TestWarmupEndpoint:
    """Tests for the model warmup endpoint."""

    async def test_warmup_model(
        self,
        client: AsyncClient,
        simple_onnx_model: Path,
        temp_storage_dir: Path,
        clean_onnx_service,
    ):
        """Test warming up a model."""
        # Upload model
        with open(simple_onnx_model, "rb") as f:
            model_content = f.read()

        upload_response = await client.post(
            "/api/v1/models/upload",
            files={"file": ("test.onnx", io.BytesIO(model_content))},
            data={"name": "warmup-test", "version": "1.0.0"},
        )
        model_id = upload_response.json()["id"]

        # Warmup
        response = await client.post(f"/api/v1/models/{model_id}/warmup")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "warmed_up"
        assert data["warmup_time_ms"] > 0
        assert data["loaded"] is True

    async def test_warmup_not_ready_model(
        self,
        client: AsyncClient,
    ):
        """Test warmup fails for model not in READY status."""
        # Create model without upload
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "warmup-not-ready", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        response = await client.post(f"/api/v1/models/{model_id}/warmup")
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_health_includes_onnx_status(client: AsyncClient):
    """Test health endpoint includes ONNX runtime status."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "onnx_runtime" in data
    assert data["onnx_runtime"] == "available"
    assert "loaded_models" in data
