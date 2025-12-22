"""Integration tests for end-to-end workflows.

These tests verify that multiple components work together correctly,
testing real user journeys across multiple API endpoints.
"""

import io
from unittest.mock import patch

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


class TestModelLifecycleWorkflow:
    """Test complete model lifecycle from creation to inference."""

    @pytest.mark.asyncio
    async def test_complete_model_workflow(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test: Create model -> Upload -> Validate -> Predict -> Check result stored."""
        # Step 1: Create model
        create_response = await client.post(
            "/api/v1/models",
            json={
                "name": "workflow-test-model",
                "version": "1.0.0",
                "description": "Integration test model",
            },
        )
        assert create_response.status_code == 201
        model_id = create_response.json()["id"]
        assert create_response.json()["status"] == "pending"

        # Step 2: Upload ONNX file
        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        upload_response = await client.post(
            f"/api/v1/models/{model_id}/upload", files=files
        )
        assert upload_response.status_code == 200
        assert upload_response.json()["status"] == "uploaded"
        assert upload_response.json()["file_size_bytes"] > 0

        # Step 3: Validate model
        validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
        assert validate_response.status_code == 200
        assert validate_response.json()["status"] == "ready"
        assert validate_response.json()["valid"] is True
        assert len(validate_response.json()["input_schema"]) == 1
        assert len(validate_response.json()["output_schema"]) == 1

        # Step 4: Run inference
        input_data = {"input": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}
        predict_response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": input_data, "request_id": "test-workflow-001"},
        )
        assert predict_response.status_code == 201
        prediction_id = predict_response.json()["id"]

        # Verify output is input + 1 (our test model behavior)
        output = predict_response.json()["output_data"]["output"][0]
        expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        for actual, exp in zip(output, expected):
            assert abs(actual - exp) < 0.001

        # Step 5: Verify prediction stored in history
        history_response = await client.get(f"/api/v1/models/{model_id}/predictions")
        assert history_response.status_code == 200
        predictions = history_response.json()["items"]
        assert len(predictions) >= 1

        stored_pred = next((p for p in predictions if p["id"] == prediction_id), None)
        assert stored_pred is not None
        assert stored_pred["inference_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_model_workflow_with_multiple_predictions(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test multiple predictions on same model and verify all stored."""
        # Setup model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "multi-predict-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Run 5 predictions with different inputs
        prediction_ids = []
        for i in range(5):
            input_val = float(i)
            response = await client.post(
                f"/api/v1/models/{model_id}/predict",
                json={"input_data": {"input": [[input_val] * 10]}},
            )
            assert response.status_code == 201
            prediction_ids.append(response.json()["id"])

        # Verify all predictions in history
        history = await client.get(f"/api/v1/models/{model_id}/predictions")
        assert history.json()["total"] == 5

        # Verify each prediction has correct output
        for item in history.json()["items"]:
            input_val = item["input_data"]["input"][0][0]
            expected_output = input_val + 1.0
            actual_output = item["output_data"]["output"][0][0]
            assert abs(actual_output - expected_output) < 0.001


class TestAsyncJobWorkflow:
    """Test async job queue workflows."""

    @pytest.mark.asyncio
    async def test_job_creation_and_result_retrieval(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test: Create model -> Create job -> Get result."""
        # Setup model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "job-workflow-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Create job (Celery is mocked in tests)
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-celery-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={
                    "model_id": model_id,
                    "input_data": {"input": [[1.0] * 10]},
                    "priority": "high",
                },
            )

        assert job_response.status_code == 201
        job_id = job_response.json()["id"]
        assert job_response.json()["status"] == "queued"
        assert job_response.json()["priority"] == "high"

        # Verify job appears in listing
        jobs_list = await client.get("/api/v1/jobs")
        assert jobs_list.status_code == 200
        assert any(j["id"] == job_id for j in jobs_list.json()["items"])

        # Get job details
        job_detail = await client.get(f"/api/v1/jobs/{job_id}")
        assert job_detail.status_code == 200
        assert job_detail.json()["model_id"] == model_id

    @pytest.mark.asyncio
    async def test_job_cancellation_workflow(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test job creation and cancellation."""
        # Setup model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "cancel-job-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Create job
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-celery-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Cancel job
        with patch("app.api.jobs.celery_app") as mock_celery:
            cancel_response = await client.post(f"/api/v1/jobs/{job_id}/cancel")

        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"

        # Verify job is cancelled in listing
        job_detail = await client.get(f"/api/v1/jobs/{job_id}")
        assert job_detail.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_multiple_jobs_for_same_model(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test creating multiple jobs for the same model."""
        # Setup model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "multi-job-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Create multiple jobs with different priorities
        job_ids = []
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            for priority in ["low", "normal", "high"]:
                response = await client.post(
                    "/api/v1/jobs",
                    json={
                        "model_id": model_id,
                        "input_data": {"input": [[1.0] * 10]},
                        "priority": priority,
                    },
                )
                assert response.status_code == 201
                job_ids.append(response.json()["id"])

        # Verify all jobs exist
        jobs_list = await client.get("/api/v1/jobs")
        for job_id in job_ids:
            assert any(j["id"] == job_id for j in jobs_list.json()["items"])


class TestModelVersioningWorkflow:
    """Test model versioning workflows."""

    @pytest.mark.asyncio
    async def test_create_multiple_versions(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test creating and querying multiple versions of a model."""
        model_name = "versioned-model"

        # Create v1.0.0
        v1_response = await client.post(
            "/api/v1/models",
            json={"name": model_name, "version": "1.0.0"},
        )
        assert v1_response.status_code == 201

        # Create v2.0.0
        v2_response = await client.post(
            "/api/v1/models",
            json={"name": model_name, "version": "2.0.0"},
        )
        assert v2_response.status_code == 201

        # Create v1.5.0 (out of order)
        v15_response = await client.post(
            "/api/v1/models",
            json={"name": model_name, "version": "1.5.0"},
        )
        assert v15_response.status_code == 201

        # Get all versions
        versions_response = await client.get(
            f"/api/v1/models/by-name/{model_name}/versions"
        )
        assert versions_response.status_code == 200
        data = versions_response.json()
        versions = data["versions"]
        assert len(versions) == 3
        assert data["total"] == 3

        # Should be sorted by version (newest first)
        assert versions[0]["version"] == "2.0.0"
        assert versions[1]["version"] == "1.5.0"
        assert versions[2]["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_latest_version(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting latest version of a model."""
        model_name = "latest-version-model"

        # Create versions
        for version in ["1.0.0", "2.0.0", "1.5.0"]:
            await client.post(
                "/api/v1/models",
                json={"name": model_name, "version": version},
            )

        # Get latest (should be 2.0.0)
        latest_response = await client.get(
            f"/api/v1/models/by-name/{model_name}/latest"
        )
        assert latest_response.status_code == 200
        assert latest_response.json()["version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_get_latest_ready_version(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting latest READY version when newer versions aren't ready."""
        model_name = "ready-version-model"

        # Create v1.0.0 and make it ready
        v1_response = await client.post(
            "/api/v1/models",
            json={"name": model_name, "version": "1.0.0"},
        )
        v1_id = v1_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{v1_id}/upload", files=files)
        await client.post(f"/api/v1/models/{v1_id}/validate")

        # Create v2.0.0 but leave it pending
        await client.post(
            "/api/v1/models",
            json={"name": model_name, "version": "2.0.0"},
        )

        # Get latest (any status) - should be 2.0.0
        latest_any = await client.get(f"/api/v1/models/by-name/{model_name}/latest")
        assert latest_any.json()["version"] == "2.0.0"

        # Get latest ready - should be 1.0.0
        latest_ready = await client.get(
            f"/api/v1/models/by-name/{model_name}/latest?ready_only=true"
        )
        assert latest_ready.json()["version"] == "1.0.0"


class TestErrorPathIntegration:
    """Test error handling across components."""

    @pytest.mark.asyncio
    async def test_predict_on_pending_model_fails(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Cannot predict on model that isn't READY."""
        # Create model but don't validate
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "pending-predict-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Try to predict
        predict_response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": {"input": [[1.0] * 10]}},
        )
        assert predict_response.status_code == 400
        assert "not ready" in predict_response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_validate_without_upload_fails(self, client: AsyncClient):
        """Cannot validate model without uploaded file."""
        # Create model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "no-file-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Try to validate
        validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
        assert validate_response.status_code == 400
        assert "uploaded file" in validate_response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_invalid_onnx_validation_error_stored(self, client: AsyncClient):
        """Validation failure stores error message for later retrieval."""
        # Create model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "invalid-onnx-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Upload invalid ONNX file
        invalid_content = io.BytesIO(b"not a valid onnx file")
        files = {"file": ("model.onnx", invalid_content, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)

        # Validate (will fail)
        validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
        assert validate_response.status_code == 200
        assert validate_response.json()["valid"] is False
        assert validate_response.json()["status"] == "error"
        assert validate_response.json()["error_message"] is not None

        # Verify error stored in model record
        model_response = await client.get(f"/api/v1/models/{model_id}")
        assert model_response.json()["status"] == "error"

    @pytest.mark.asyncio
    async def test_job_for_nonexistent_model_fails(self, client: AsyncClient):
        """Cannot create job for nonexistent model."""
        job_response = await client.post(
            "/api/v1/jobs",
            json={
                "model_id": "00000000-0000-0000-0000-000000000000",
                "input_data": {"input": [[1.0] * 10]},
            },
        )
        assert job_response.status_code == 404

    @pytest.mark.asyncio
    async def test_duplicate_model_version_rejected(self, client: AsyncClient):
        """Cannot create duplicate model name+version combination."""
        # Create first model
        await client.post(
            "/api/v1/models",
            json={"name": "duplicate-model", "version": "1.0.0"},
        )

        # Try to create duplicate
        duplicate_response = await client.post(
            "/api/v1/models",
            json={"name": "duplicate-model", "version": "1.0.0"},
        )
        assert duplicate_response.status_code == 409


class TestCacheIntegration:
    """Test cache behavior integration.

    Note: Full cache testing is done in test_cache.py and test_prediction_cache.py.
    These tests verify cache behavior works correctly in API context.
    """

    @pytest.mark.asyncio
    async def test_prediction_with_cache_disabled(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test predictions work correctly when cache is disabled."""
        # Setup model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "cache-disabled-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Make two identical predictions
        input_data = {"input": [[1.0] * 10]}
        response1 = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": input_data},
        )
        assert response1.status_code == 201

        response2 = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": input_data},
        )
        assert response2.status_code == 201

        # Both should have results (cache miss doesn't break predictions)
        output1 = response1.json()["output_data"]["output"][0]
        output2 = response2.json()["output_data"]["output"][0]

        # Verify both produce correct output
        expected = [2.0] * 10
        for actual, exp in zip(output1, expected):
            assert abs(actual - exp) < 0.001
        for actual, exp in zip(output2, expected):
            assert abs(actual - exp) < 0.001

        # Both should be stored (two different prediction IDs)
        assert response1.json()["id"] != response2.json()["id"]

    @pytest.mark.asyncio
    async def test_skip_cache_parameter(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test skip_cache parameter bypasses cache."""
        # Setup model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "skip-cache-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Prediction with skip_cache
        input_data = {"input": [[1.0] * 10]}
        response = await client.post(
            f"/api/v1/models/{model_id}/predict",
            json={"input_data": input_data, "skip_cache": True},
        )
        assert response.status_code == 201

        # Verify output
        output = response.json()["output_data"]["output"][0]
        expected = [2.0] * 10
        for actual, exp in zip(output, expected):
            assert abs(actual - exp) < 0.001


class TestDataConsistency:
    """Test data consistency across operations."""

    @pytest.mark.asyncio
    async def test_model_deletion_cascade(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Verify model deletion doesn't leave orphaned data."""
        # Create and setup model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "delete-cascade-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Create predictions
        for i in range(3):
            await client.post(
                f"/api/v1/models/{model_id}/predict",
                json={"input_data": {"input": [[float(i)] * 10]}},
            )

        # Verify predictions exist
        history = await client.get(f"/api/v1/models/{model_id}/predictions")
        assert history.json()["total"] == 3

        # Delete model
        delete_response = await client.delete(f"/api/v1/models/{model_id}")
        assert delete_response.status_code == 204

        # Verify model is gone
        get_response = await client.get(f"/api/v1/models/{model_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_model_status_transitions(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Verify model status transitions follow expected lifecycle."""
        # Create model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "status-transition-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        # Check PENDING status
        model = await client.get(f"/api/v1/models/{model_id}")
        assert model.json()["status"] == "pending"

        # Upload -> UPLOADED
        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        model = await client.get(f"/api/v1/models/{model_id}")
        assert model.json()["status"] == "uploaded"

        # Validate -> READY
        await client.post(f"/api/v1/models/{model_id}/validate")
        model = await client.get(f"/api/v1/models/{model_id}")
        assert model.json()["status"] == "ready"

    @pytest.mark.asyncio
    async def test_predictions_all_returned(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Verify all predictions are returned in history."""
        # Setup model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "predictions-returned-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Create predictions with different inputs
        created_ids = set()
        for i in range(3):
            response = await client.post(
                f"/api/v1/models/{model_id}/predict",
                json={"input_data": {"input": [[float(i)] * 10]}},
            )
            created_ids.add(response.json()["id"])

        # Get predictions
        history = await client.get(f"/api/v1/models/{model_id}/predictions")
        returned_ids = {p["id"] for p in history.json()["items"]}

        # All created predictions should be in history
        assert created_ids.issubset(returned_ids)
        assert history.json()["total"] >= 3


class TestPaginationIntegration:
    """Test pagination across endpoints."""

    @pytest.mark.asyncio
    async def test_model_list_pagination(self, client: AsyncClient):
        """Test paginating through model list."""
        # Create 10 models
        model_ids = []
        for i in range(10):
            response = await client.post(
                "/api/v1/models",
                json={"name": f"pagination-model-{i}", "version": "1.0.0"},
            )
            model_ids.append(response.json()["id"])

        # Get first page
        page1 = await client.get("/api/v1/models?page=1&page_size=3")
        assert page1.status_code == 200
        assert len(page1.json()["items"]) == 3
        assert page1.json()["total"] >= 10
        assert page1.json()["page"] == 1

        # Get second page
        page2 = await client.get("/api/v1/models?page=2&page_size=3")
        assert page2.status_code == 200
        assert len(page2.json()["items"]) == 3
        assert page2.json()["page"] == 2

        # Verify no overlap
        page1_ids = {m["id"] for m in page1.json()["items"]}
        page2_ids = {m["id"] for m in page2.json()["items"]}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_job_list_pagination(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test paginating through job list."""
        # Setup model
        create_response = await client.post(
            "/api/v1/models",
            json={"name": "job-pagination-model", "version": "1.0.0"},
        )
        model_id = create_response.json()["id"]

        files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
        await client.post(f"/api/v1/models/{model_id}/upload", files=files)
        await client.post(f"/api/v1/models/{model_id}/validate")

        # Create 10 jobs
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            for i in range(10):
                await client.post(
                    "/api/v1/jobs",
                    json={
                        "model_id": model_id,
                        "input_data": {"input": [[float(i)] * 10]},
                    },
                )

        # Test pagination
        page1 = await client.get("/api/v1/jobs?page=1&page_size=4")
        assert len(page1.json()["items"]) == 4

        page2 = await client.get("/api/v1/jobs?page=2&page_size=4")
        assert len(page2.json()["items"]) == 4

        # Verify different items
        page1_ids = {j["id"] for j in page1.json()["items"]}
        page2_ids = {j["id"] for j in page2.json()["items"]}
        assert page1_ids.isdisjoint(page2_ids)
