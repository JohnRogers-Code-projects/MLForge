"""Tests for job endpoints.

Tests the async job API which requires models to be in READY state
and integrates with Celery for task queuing.
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


async def setup_ready_model(
    client: AsyncClient, valid_onnx_file: io.BytesIO, name: str = "job-test-model"
) -> str:
    """Helper to create, upload, and validate a model. Returns model_id."""
    # Reset file position
    valid_onnx_file.seek(0)

    # Create model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": name, "version": "1.0.0"},
    )
    assert create_response.status_code == 201
    model_id = create_response.json()["id"]

    # Upload ONNX file
    files = {"file": ("model.onnx", valid_onnx_file, "application/octet-stream")}
    upload_response = await client.post(f"/api/v1/models/{model_id}/upload", files=files)
    assert upload_response.status_code == 200

    # Validate model
    validate_response = await client.post(f"/api/v1/models/{model_id}/validate")
    assert validate_response.status_code == 200
    assert validate_response.json()["status"] == "ready"

    return model_id


class TestJobCreation:
    """Tests for job creation endpoint."""

    @pytest.mark.asyncio
    async def test_create_job_success(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test creating a new job for a READY model."""
        model_id = await setup_ready_model(client, valid_onnx_file, "create-job-model")

        # Create a job - mock Celery task to avoid Redis dependency
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"

            job_data = {
                "model_id": model_id,
                "input_data": {"input": [[1.0] * 10]},
                "priority": "normal",
            }
            response = await client.post("/api/v1/jobs", json=job_data)

        assert response.status_code == 201
        data = response.json()
        assert data["model_id"] == model_id
        # Status should be QUEUED after successful task queuing
        assert data["status"] == "queued"
        assert data["priority"] == "normal"
        assert data["celery_task_id"] == "mock-task-id"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_job_celery_failure_falls_back_to_pending(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test job stays PENDING if Celery queuing fails."""
        model_id = await setup_ready_model(client, valid_onnx_file, "celery-fail-model")

        # Mock Celery task to raise exception
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.side_effect = Exception("Redis connection refused")

            job_data = {
                "model_id": model_id,
                "input_data": {"input": [[1.0] * 10]},
            }
            response = await client.post("/api/v1/jobs", json=job_data)

        assert response.status_code == 201
        data = response.json()
        # Job should stay in PENDING status
        assert data["status"] == "pending"
        assert data["celery_task_id"] is None

    @pytest.mark.asyncio
    async def test_create_job_nonexistent_model(self, client: AsyncClient):
        """Test creating a job for a nonexistent model returns 404."""
        job_data = {
            "model_id": "00000000-0000-0000-0000-000000000000",
            "input_data": {"feature1": 1.0},
        }
        response = await client.post("/api/v1/jobs", json=job_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_create_job_model_not_ready(self, client: AsyncClient):
        """Test creating a job for a model that isn't READY returns 400."""
        # Create model but don't upload/validate
        model_response = await client.post(
            "/api/v1/models",
            json={"name": "not-ready-job-model", "version": "1.0.0"},
        )
        model_id = model_response.json()["id"]

        job_data = {
            "model_id": model_id,
            "input_data": {"feature1": 1.0},
        }
        response = await client.post("/api/v1/jobs", json=job_data)
        assert response.status_code == 400
        assert "not ready" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_create_job_with_priority(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test creating jobs with different priorities."""
        model_id = await setup_ready_model(client, valid_onnx_file, "priority-job-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"

            for priority in ["low", "normal", "high"]:
                job_data = {
                    "model_id": model_id,
                    "input_data": {"input": [[1.0] * 10]},
                    "priority": priority,
                }
                response = await client.post("/api/v1/jobs", json=job_data)
                assert response.status_code == 201
                assert response.json()["priority"] == priority


class TestJobListing:
    """Tests for job listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_jobs(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test listing jobs."""
        model_id = await setup_ready_model(client, valid_onnx_file, "list-jobs-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"

            for i in range(3):
                await client.post(
                    "/api/v1/jobs",
                    json={"model_id": model_id, "input_data": {"input": [[float(i)] * 10]}},
                )

        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 3
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data

    @pytest.mark.asyncio
    async def test_list_jobs_by_status(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test filtering jobs by status."""
        model_id = await setup_ready_model(client, valid_onnx_file, "status-filter-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )

        response = await client.get("/api/v1/jobs?status=queued")
        assert response.status_code == 200
        data = response.json()
        for job in data["items"]:
            assert job["status"] == "queued"

    @pytest.mark.asyncio
    async def test_list_jobs_pagination(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test pagination of job listing."""
        model_id = await setup_ready_model(client, valid_onnx_file, "paginate-jobs-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"

            for i in range(5):
                await client.post(
                    "/api/v1/jobs",
                    json={"model_id": model_id, "input_data": {"input": [[float(i)] * 10]}},
                )

        response = await client.get("/api/v1/jobs?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2


class TestJobRetrieval:
    """Tests for getting specific jobs."""

    @pytest.mark.asyncio
    async def test_get_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting a specific job by ID."""
        model_id = await setup_ready_model(client, valid_onnx_file, "get-job-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        response = await client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["model_id"] == model_id

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, client: AsyncClient):
        """Test getting a nonexistent job returns 404."""
        response = await client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestJobCancellation:
    """Tests for job cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_pending_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test cancelling a job in PENDING status."""
        model_id = await setup_ready_model(client, valid_onnx_file, "cancel-pending-model")

        # Create job that stays in PENDING (Celery fails)
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.side_effect = Exception("Redis unavailable")
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]
        assert job_response.json()["status"] == "pending"

        # Cancel the job
        response = await client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_queued_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test cancelling a job in QUEUED status."""
        model_id = await setup_ready_model(client, valid_onnx_file, "cancel-queued-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]
        assert job_response.json()["status"] == "queued"

        # Cancel the job
        response = await client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self, client: AsyncClient):
        """Test cancelling a nonexistent job returns 404."""
        response = await client.post(
            "/api/v1/jobs/00000000-0000-0000-0000-000000000000/cancel"
        )
        assert response.status_code == 404


class TestJobResults:
    """Tests for job result retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_get_result_completed_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting result of a successfully completed job."""
        model_id = await setup_ready_model(client, valid_onnx_file, "result-completed-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Simulate job completion by directly updating via CRUD
        from app.models.job import JobStatus
        from app.crud import job_crud
        from app.database import get_db

        # Get a fresh session from the client's dependency override
        async for session in client._transport.app.dependency_overrides[get_db]():
            job = await job_crud.get(session, job_id)
            await job_crud.update(
                session,
                db_obj=job,
                obj_in={"status": JobStatus.COMPLETED, "output_data": {"output": [[2.0] * 10]}},
            )
            await session.commit()
            break

        response = await client.get(f"/api/v1/jobs/{job_id}/result")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "completed"
        assert data["result"] == {"output": [[2.0] * 10]}
        assert data["error_message"] is None
        assert data["error_traceback"] is None

    @pytest.mark.asyncio
    async def test_get_result_failed_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting result of a failed job returns error details."""
        model_id = await setup_ready_model(client, valid_onnx_file, "result-failed-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Simulate job failure
        from app.models.job import JobStatus
        from app.crud import job_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            job = await job_crud.get(session, job_id)
            await job_crud.update(
                session,
                db_obj=job,
                obj_in={
                    "status": JobStatus.FAILED,
                    "error_message": "Model inference failed",
                    "error_traceback": "Traceback (most recent call last):\n  File ...\nRuntimeError: CUDA out of memory",
                },
            )
            await session.commit()
            break

        response = await client.get(f"/api/v1/jobs/{job_id}/result")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "failed"
        assert data["result"] is None
        assert data["error_message"] == "Model inference failed"
        assert "CUDA out of memory" in data["error_traceback"]

    @pytest.mark.asyncio
    async def test_get_result_processing_job_returns_202(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting result of a still-processing job returns 202."""
        model_id = await setup_ready_model(client, valid_onnx_file, "result-processing-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]
        # Job is in QUEUED status

        response = await client.get(f"/api/v1/jobs/{job_id}/result")
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "queued"
        assert data["message"] == "Job is still processing"

    @pytest.mark.asyncio
    async def test_get_result_pending_job_returns_202(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting result of a pending job returns 202."""
        model_id = await setup_ready_model(client, valid_onnx_file, "result-pending-model")

        # Create job that stays PENDING (Celery fails)
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.side_effect = Exception("Redis unavailable")
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]
        assert job_response.json()["status"] == "pending"

        response = await client.get(f"/api/v1/jobs/{job_id}/result")
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_result_cancelled_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting result of a cancelled job."""
        model_id = await setup_ready_model(client, valid_onnx_file, "result-cancelled-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Cancel the job
        await client.post(f"/api/v1/jobs/{job_id}/cancel")

        response = await client.get(f"/api/v1/jobs/{job_id}/result")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "cancelled"
        assert data["result"] is None
        assert data["error_message"] is None

    @pytest.mark.asyncio
    async def test_get_result_not_found(self, client: AsyncClient):
        """Test getting result of nonexistent job returns 404."""
        response = await client.get(
            "/api/v1/jobs/00000000-0000-0000-0000-000000000000/result"
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_result_with_wait_completes(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test wait parameter returns when job completes."""
        model_id = await setup_ready_model(client, valid_onnx_file, "result-wait-complete-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Complete the job before requesting result
        from app.models.job import JobStatus
        from app.crud import job_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            job = await job_crud.get(session, job_id)
            await job_crud.update(
                session,
                db_obj=job,
                obj_in={"status": JobStatus.COMPLETED, "output_data": {"output": [[2.0] * 10]}},
            )
            await session.commit()
            break

        # Request with wait - should return immediately since job is done
        response = await client.get(f"/api/v1/jobs/{job_id}/result?wait=5")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_result_with_wait_timeout(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test wait parameter times out if job doesn't complete."""
        model_id = await setup_ready_model(client, valid_onnx_file, "result-wait-timeout-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Don't complete the job - it stays QUEUED
        # Use a short wait to not slow down tests
        response = await client.get(f"/api/v1/jobs/{job_id}/result?wait=1")
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "queued"
        assert data["message"] == "Job is still processing"

    @pytest.mark.asyncio
    async def test_get_result_wait_exceeds_max(self, client: AsyncClient):
        """Test wait parameter validation rejects values exceeding max."""
        # Max wait is 30 seconds
        response = await client.get(
            "/api/v1/jobs/00000000-0000-0000-0000-000000000000/result?wait=60"
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_result_wait_negative(self, client: AsyncClient):
        """Test wait parameter validation rejects negative values."""
        response = await client.get(
            "/api/v1/jobs/00000000-0000-0000-0000-000000000000/result?wait=-1"
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_result_includes_timing_info(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test completed job result includes timing info."""
        model_id = await setup_ready_model(client, valid_onnx_file, "result-timing-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        from app.models.job import JobStatus
        from app.crud import job_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            job = await job_crud.get(session, job_id)
            await job_crud.update(
                session,
                db_obj=job,
                obj_in={
                    "status": JobStatus.COMPLETED,
                    "output_data": {"output": [[2.0] * 10]},
                    "inference_time_ms": 42.5,
                },
            )
            await session.commit()
            break

        response = await client.get(f"/api/v1/jobs/{job_id}/result")
        assert response.status_code == 200
        data = response.json()
        assert data["inference_time_ms"] == 42.5
