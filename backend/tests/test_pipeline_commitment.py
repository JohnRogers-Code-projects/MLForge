"""Tests for pipeline commitment boundary enforcement.

=============================================================================
NEGATIVE TESTS
=============================================================================

These tests prove that misuse fails and invalid state cannot propagate.

The pipeline commitment boundary is THE architectural guarantee of MLForge:
- Before commitment (PENDING, UPLOADED, VALIDATING, ERROR): experimentation allowed
- After commitment (READY): invariants enforced, violations fail loudly

These tests verify:
1. Pre-boundary models cannot be used for inference
2. Error messages explicitly name the commitment boundary
3. Invalid artifacts cannot reach READY status
4. Violations are explicit, not silent
"""

import io

import onnx
import pytest
from httpx import AsyncClient

from tests.conftest import create_simple_onnx_model


@pytest.fixture
def valid_onnx_file() -> io.BytesIO:
    """Create a valid ONNX model file for testing."""
    model = create_simple_onnx_model()
    buffer = io.BytesIO()
    onnx.save(model, buffer)
    buffer.seek(0)
    return buffer


@pytest.fixture
def invalid_onnx_file() -> io.BytesIO:
    """Create an invalid ONNX file (random bytes)."""
    buffer = io.BytesIO(b"this is not a valid onnx model - just garbage bytes")
    return buffer


class TestPipelineCommitmentEnforcement:
    """Tests proving pre-boundary models cannot run inference.

    These are negative tests. They prove that the system correctly
    REJECTS operations that violate the pipeline commitment boundary.

    The commitment boundary is not optional. It is THE architectural
    guarantee that separates "experimentation" from "production-ready".
    """

    @pytest.mark.asyncio
    async def test_inference_on_pending_model_fails_with_commitment_message(
        self, client: AsyncClient
    ):
        """PENDING model cannot run inference. Error names commitment boundary.

        This proves:
        1. A model in PENDING status cannot be used for inference
        2. The error message explicitly mentions "commitment boundary"
        3. The failure is a 400 Bad Request (client error, not server error)
        """
        # Create model but don't upload or validate
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "pending-model", "version": "1.0.0"},
        )
        assert create_response.status_code == 201
        model_id = create_response.json()["id"]
        assert create_response.json()["status"] == "pending"

        # Attempt inference on PENDING model
        response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}},
        )

        # Must fail with 400
        assert response.status_code == 400

        # Error must explicitly name the commitment boundary
        detail = response.json()["detail"]
        assert "commitment" in detail.lower() or "boundary" in detail.lower(), (
            f"Error message must mention commitment boundary. Got: {detail}"
        )

    @pytest.mark.asyncio
    async def test_inference_on_uploaded_model_fails_with_commitment_message(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """UPLOADED model cannot run inference. Error names commitment boundary.

        This is the critical test. A model that has been uploaded but NOT
        validated has NOT crossed the commitment boundary.

        The validate endpoint is THE commitment point. Without it, the model
        is experimental and cannot be used for inference.
        """
        # Create and upload but DON'T validate
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "uploaded-not-validated", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Upload file
        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        upload_response = await client.post(
            f"/api/v1/models/{model_id}/upload", files=files
        )
        assert upload_response.status_code == 200
        assert upload_response.json()["status"] == "uploaded"

        # Attempt inference on UPLOADED (not READY) model
        response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}},
        )

        # Must fail with 400
        assert response.status_code == 400

        # Error must explicitly name the commitment boundary
        detail = response.json()["detail"]
        assert "commitment" in detail.lower() or "boundary" in detail.lower(), (
            f"Error message must mention commitment boundary. Got: {detail}"
        )
        # Should also mention current status
        assert "uploaded" in detail.lower(), (
            f"Error message should mention current status. Got: {detail}"
        )

    @pytest.mark.asyncio
    async def test_inference_on_error_model_fails_with_commitment_message(
        self, client: AsyncClient, invalid_onnx_file: io.BytesIO
    ):
        """ERROR status model cannot run inference.

        A model that failed validation is in ERROR status. It has NOT
        crossed the commitment boundary. It cannot be used for inference.
        """
        # Create model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "error-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Upload invalid file
        files = {"file": ("model.onnx", invalid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)

        # Attempt validation - should fail and put model in ERROR status
        validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
        assert validate_response.status_code == 400

        # Verify model is in ERROR status
        get_response = await client.get(f"/api/v1/models/{model_id}")
        assert get_response.json()["status"] == "error"

        # Attempt inference on ERROR model
        response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}},
        )

        # Must fail with 400
        assert response.status_code == 400

        # Error must mention commitment or boundary
        detail = response.json()["detail"]
        assert "commitment" in detail.lower() or "boundary" in detail.lower(), (
            f"Error message must mention commitment boundary. Got: {detail}"
        )

    @pytest.mark.asyncio
    async def test_async_job_on_uncommitted_model_fails(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Async job submission for uncommitted model fails.

        The async path (Celery jobs) must also enforce the commitment
        boundary. An uncommitted model cannot be used for async inference.

        Note: This tests job creation, not job execution. Job creation
        should fail immediately for uncommitted models.
        """
        # Create and upload but DON'T validate
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "async-test-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)

        # Attempt to create async job on UPLOADED model
        response = await client.post(
            "/api/v1/jobs",
            json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
        )

        # Must fail - uncommitted model cannot have jobs created
        assert response.status_code == 400

        # Error should indicate model is not ready
        detail = response.json()["detail"]
        assert (
            "commitment" in detail.lower()
            or "boundary" in detail.lower()
            or "ready" in detail.lower()
        ), f"Error message should indicate model not ready. Got: {detail}"


class TestInvalidStateCannotPropagate:
    """Tests proving invalid artifacts cannot reach READY status.

    The commitment boundary is enforced AT validation time.
    Invalid models must be rejected and cannot reach READY status.
    """

    @pytest.mark.asyncio
    async def test_invalid_onnx_file_stays_in_error_status(
        self, client: AsyncClient, invalid_onnx_file: io.BytesIO
    ):
        """Invalid ONNX file cannot reach READY status.

        This proves:
        1. Invalid ONNX file can be uploaded (pre-boundary, experimentation allowed)
        2. Validation fails with explicit error
        3. Model stays in ERROR status, never reaches READY
        4. The error is loud and explicit
        """
        # Create model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "invalid-onnx-test", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Upload invalid file - should succeed (pre-boundary)
        files = {"file": ("model.onnx", invalid_onnx_file, "application/octet-stream")}
        upload_response = await client.post(
            f"/api/v1/models/{model_id}/upload", files=files
        )
        assert upload_response.status_code == 200
        assert upload_response.json()["status"] == "uploaded"

        # Validation must fail
        validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
        assert validate_response.status_code == 400

        # Model must be in ERROR status, not READY
        get_response = await client.get(f"/api/v1/models/{model_id}")
        model_data = get_response.json()

        assert model_data["status"] == "error", (
            f"Invalid model must be in ERROR status, not {model_data['status']}"
        )
        assert model_data["status"] != "ready", "Invalid model must NEVER reach READY"

    @pytest.mark.asyncio
    async def test_empty_file_cannot_become_ready(self, client: AsyncClient):
        """Empty file cannot reach READY status.

        Even a zero-byte file cannot cross the commitment boundary.
        """
        # Create model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "empty-file-test", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Upload empty file
        empty_file = io.BytesIO(b"")
        files = {"file": ("model.onnx", empty_file, "application/octet-stream")}
        upload_response = await client.post(
            f"/api/v1/models/{model_id}/upload", files=files
        )

        # Empty file should be rejected at upload or validation
        if upload_response.status_code == 200:
            # If upload succeeded, validation must fail
            validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
            assert validate_response.status_code == 400

            # Verify not in READY status
            get_response = await client.get(f"/api/v1/models/{model_id}")
            assert get_response.json()["status"] != "ready"
        else:
            # Upload itself rejected empty file - also acceptable
            assert upload_response.status_code == 400

    @pytest.mark.asyncio
    async def test_validation_is_required_for_ready_status(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Model cannot be READY without explicit validation call.

        The validate endpoint is THE commitment point. There is no
        backdoor to READY status. You must call validate.
        """
        # Create model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "no-backdoor-test", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]
        assert create_response.json()["status"] == "pending"

        # Upload file
        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        upload_response = await client.post(
            f"/api/v1/models/{model_id}/upload", files=files
        )
        assert upload_response.status_code == 200
        assert upload_response.json()["status"] == "uploaded"

        # Model must NOT be READY without validation
        get_response = await client.get(f"/api/v1/models/{model_id}")
        assert get_response.json()["status"] != "ready", (
            "Model must not be READY without explicit validation call"
        )

        # Now validate
        validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
        assert validate_response.status_code == 200

        # NOW it should be READY
        get_response = await client.get(f"/api/v1/models/{model_id}")
        assert get_response.json()["status"] == "ready"


class TestCommitmentBoundaryExplicitness:
    """Tests proving the commitment boundary is explicit in system behavior.

    These tests verify that the commitment concept is not hidden or implicit.
    Users must be able to understand the boundary from error messages.
    """

    @pytest.mark.asyncio
    async def test_error_message_names_current_status(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Error messages must tell user the current model status.

        When inference is rejected, the user needs to know:
        1. The model's current status
        2. What status is required (READY)
        3. What concept they're violating (commitment boundary)
        """
        # Create and upload but don't validate
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "status-in-error-test", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)

        # Attempt inference
        response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}},
        )

        assert response.status_code == 400
        detail = response.json()["detail"].lower()

        # Must mention current status
        assert "uploaded" in detail, (
            f"Error should mention current status 'uploaded'. Got: {detail}"
        )

    @pytest.mark.asyncio
    async def test_validated_model_allows_inference(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Validated model (READY status) allows inference.

        This is the positive case proving the boundary works both ways:
        - Before boundary: inference rejected
        - After boundary: inference allowed
        """
        # Create model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "positive-case-test", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Upload
        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)

        # Validate - crosses commitment boundary
        validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
        assert validate_response.status_code == 200
        assert validate_response.json()["status"] == "ready"

        # Now inference should work
        response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}},
        )

        assert response.status_code == 201
        assert "output_data" in response.json()
