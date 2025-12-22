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


class TestJobCancellationWithRevoke:
    """Tests for job cancellation with Celery task revocation."""

    @pytest.mark.asyncio
    async def test_cancel_queued_job_revokes_celery_task(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test that cancelling a queued job revokes the Celery task."""
        model_id = await setup_ready_model(client, valid_onnx_file, "cancel-revoke-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id-to-revoke"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]
        assert job_response.json()["celery_task_id"] == "mock-task-id-to-revoke"

        # Cancel the job - should revoke the Celery task
        with patch("app.api.jobs.celery_app") as mock_celery:
            response = await client.post(f"/api/v1/jobs/{job_id}/cancel")
            assert response.status_code == 200
            # Verify revoke was called with terminate=True
            mock_celery.control.revoke.assert_called_once_with(
                "mock-task-id-to-revoke", terminate=True
            )

    @pytest.mark.asyncio
    async def test_cancel_running_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test cancelling a job in RUNNING status."""
        model_id = await setup_ready_model(client, valid_onnx_file, "cancel-running-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Update job to RUNNING status
        from app.models.job import JobStatus
        from app.crud import job_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            job = await job_crud.get(session, job_id)
            await job_crud.update(
                session,
                db_obj=job,
                obj_in={"status": JobStatus.RUNNING},
            )
            await session.commit()
            break

        # Cancel the running job
        with patch("app.api.jobs.celery_app"):
            response = await client.post(f"/api/v1/jobs/{job_id}/cancel")
            assert response.status_code == 200
            assert response.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_completed_job_fails(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test that cancelling a completed job returns 400."""
        model_id = await setup_ready_model(client, valid_onnx_file, "cancel-completed-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Update job to COMPLETED status
        from app.models.job import JobStatus
        from app.crud import job_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            job = await job_crud.get(session, job_id)
            await job_crud.update(
                session,
                db_obj=job,
                obj_in={"status": JobStatus.COMPLETED},
            )
            await session.commit()
            break

        # Try to cancel - should fail
        response = await client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert response.status_code == 400
        assert "Cannot cancel job" in response.json()["detail"]


class TestJobDeletion:
    """Tests for job deletion endpoint."""

    @pytest.mark.asyncio
    async def test_delete_completed_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test deleting a completed job."""
        model_id = await setup_ready_model(client, valid_onnx_file, "delete-completed-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Update job to COMPLETED status
        from app.models.job import JobStatus
        from app.crud import job_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            job = await job_crud.get(session, job_id)
            await job_crud.update(
                session,
                db_obj=job,
                obj_in={"status": JobStatus.COMPLETED},
            )
            await session.commit()
            break

        # Delete the job
        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 204

        # Verify job is gone
        get_response = await client.get(f"/api/v1/jobs/{job_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_failed_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test deleting a failed job."""
        model_id = await setup_ready_model(client, valid_onnx_file, "delete-failed-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Update job to FAILED status
        from app.models.job import JobStatus
        from app.crud import job_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            job = await job_crud.get(session, job_id)
            await job_crud.update(
                session,
                db_obj=job,
                obj_in={"status": JobStatus.FAILED, "error_message": "Test failure"},
            )
            await session.commit()
            break

        # Delete the job
        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_cancelled_job(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test deleting a cancelled job."""
        model_id = await setup_ready_model(client, valid_onnx_file, "delete-cancelled-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Cancel the job
        with patch("app.api.jobs.celery_app"):
            await client.post(f"/api/v1/jobs/{job_id}/cancel")

        # Delete the job
        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_running_job_fails(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test that deleting a running job returns 400."""
        model_id = await setup_ready_model(client, valid_onnx_file, "delete-running-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]

        # Update job to RUNNING status
        from app.models.job import JobStatus
        from app.crud import job_crud
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            job = await job_crud.get(session, job_id)
            await job_crud.update(
                session,
                db_obj=job,
                obj_in={"status": JobStatus.RUNNING},
            )
            await session.commit()
            break

        # Try to delete - should fail
        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 400
        assert "Cannot delete job" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_queued_job_fails(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test that deleting a queued job returns 400."""
        model_id = await setup_ready_model(client, valid_onnx_file, "delete-queued-model")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            job_response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = job_response.json()["id"]
        assert job_response.json()["status"] == "queued"

        # Try to delete - should fail
        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 400
        assert "Cancel it first" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_job(self, client: AsyncClient):
        """Test deleting a nonexistent job returns 404."""
        response = await client.delete(
            "/api/v1/jobs/00000000-0000-0000-0000-000000000000"
        )
        assert response.status_code == 404


class TestJobCRUDOperations:
    """Direct unit tests for Job CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_by_model(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting jobs by model ID."""
        from app.crud import job_crud
        from app.database import get_db

        model_id = await setup_ready_model(client, valid_onnx_file, "crud-by-model")

        # Create some jobs for this model
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            for i in range(3):
                await client.post(
                    "/api/v1/jobs",
                    json={"model_id": model_id, "input_data": {"input": [[float(i)] * 10]}},
                )

        # Get jobs using CRUD directly
        async for session in client._transport.app.dependency_overrides[get_db]():
            jobs = await job_crud.get_by_model(session, model_id=model_id)
            assert len(jobs) == 3
            for job in jobs:
                assert job.model_id == model_id
            break

    @pytest.mark.asyncio
    async def test_get_by_model_with_pagination(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting jobs by model ID with offset and limit."""
        from app.crud import job_crud
        from app.database import get_db

        model_id = await setup_ready_model(client, valid_onnx_file, "crud-by-model-paged")

        # Create 5 jobs
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            for i in range(5):
                await client.post(
                    "/api/v1/jobs",
                    json={"model_id": model_id, "input_data": {"input": [[float(i)] * 10]}},
                )

        # Test pagination
        async for session in client._transport.app.dependency_overrides[get_db]():
            # Get first 2 jobs
            jobs = await job_crud.get_by_model(session, model_id=model_id, offset=0, limit=2)
            assert len(jobs) == 2

            # Get next 2 jobs
            jobs = await job_crud.get_by_model(session, model_id=model_id, offset=2, limit=2)
            assert len(jobs) == 2
            break

    @pytest.mark.asyncio
    async def test_get_pending_jobs(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test getting pending jobs ordered by priority."""
        from app.crud import job_crud
        from app.database import get_db

        model_id = await setup_ready_model(client, valid_onnx_file, "crud-pending-jobs")

        # Create jobs with different priorities that stay in PENDING (Celery fails)
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.side_effect = Exception("Redis unavailable")

            for priority in ["low", "normal", "high"]:
                await client.post(
                    "/api/v1/jobs",
                    json={
                        "model_id": model_id,
                        "input_data": {"input": [[1.0] * 10]},
                        "priority": priority,
                    },
                )

        # Get pending jobs
        async for session in client._transport.app.dependency_overrides[get_db]():
            pending = await job_crud.get_pending_jobs(session, limit=10)
            # Should have 3 pending jobs (could have more from other tests)
            assert len(pending) >= 3
            # All should be pending
            for job in pending:
                assert job.status.value == "pending"
            break

    @pytest.mark.asyncio
    async def test_count_by_status(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test counting jobs by status."""
        from app.crud import job_crud
        from app.models.job import JobStatus
        from app.database import get_db

        model_id = await setup_ready_model(client, valid_onnx_file, "crud-count-status")

        # Create some queued jobs
        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            for i in range(3):
                await client.post(
                    "/api/v1/jobs",
                    json={"model_id": model_id, "input_data": {"input": [[float(i)] * 10]}},
                )

        async for session in client._transport.app.dependency_overrides[get_db]():
            queued_count = await job_crud.count_by_status(session, status=JobStatus.QUEUED)
            assert queued_count >= 3
            break

    @pytest.mark.asyncio
    async def test_update_status_to_running(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test updating job status to RUNNING sets started_at."""
        from app.crud import job_crud
        from app.models.job import JobStatus
        from app.database import get_db

        model_id = await setup_ready_model(client, valid_onnx_file, "crud-status-running")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = response.json()["id"]

        async for session in client._transport.app.dependency_overrides[get_db]():
            updated = await job_crud.update_status(session, job_id=job_id, status=JobStatus.RUNNING)
            assert updated is not None
            assert updated.status == JobStatus.RUNNING
            assert updated.started_at is not None
            break

    @pytest.mark.asyncio
    async def test_update_status_to_completed(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test updating job status to COMPLETED sets completed_at."""
        from app.crud import job_crud
        from app.models.job import JobStatus
        from app.database import get_db

        model_id = await setup_ready_model(client, valid_onnx_file, "crud-status-completed")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = response.json()["id"]

        async for session in client._transport.app.dependency_overrides[get_db]():
            updated = await job_crud.update_status(
                session, job_id=job_id, status=JobStatus.COMPLETED
            )
            assert updated is not None
            assert updated.status == JobStatus.COMPLETED
            assert updated.completed_at is not None
            break

    @pytest.mark.asyncio
    async def test_update_status_to_failed_with_error(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test updating job status to FAILED with error message."""
        from app.crud import job_crud
        from app.models.job import JobStatus
        from app.database import get_db

        model_id = await setup_ready_model(client, valid_onnx_file, "crud-status-failed")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = response.json()["id"]

        async for session in client._transport.app.dependency_overrides[get_db]():
            updated = await job_crud.update_status(
                session,
                job_id=job_id,
                status=JobStatus.FAILED,
                error_message="Inference failed due to OOM",
            )
            assert updated is not None
            assert updated.status == JobStatus.FAILED
            assert updated.error_message == "Inference failed due to OOM"
            assert updated.completed_at is not None
            break

    @pytest.mark.asyncio
    async def test_update_status_nonexistent_job(
        self, client: AsyncClient
    ):
        """Test updating status of nonexistent job returns None."""
        from app.crud import job_crud
        from app.models.job import JobStatus
        from app.database import get_db

        async for session in client._transport.app.dependency_overrides[get_db]():
            result = await job_crud.update_status(
                session,
                job_id="00000000-0000-0000-0000-000000000000",
                status=JobStatus.RUNNING,
            )
            assert result is None
            break

    @pytest.mark.asyncio
    async def test_update_status_to_cancelled(
        self, client: AsyncClient, valid_onnx_file: io.BytesIO
    ):
        """Test updating job status to CANCELLED sets completed_at."""
        from app.crud import job_crud
        from app.models.job import JobStatus
        from app.database import get_db

        model_id = await setup_ready_model(client, valid_onnx_file, "crud-status-cancelled")

        with patch("app.api.jobs.run_inference_task") as mock_task:
            mock_task.delay.return_value.id = "mock-task-id"
            response = await client.post(
                "/api/v1/jobs",
                json={"model_id": model_id, "input_data": {"input": [[1.0] * 10]}},
            )
        job_id = response.json()["id"]

        async for session in client._transport.app.dependency_overrides[get_db]():
            updated = await job_crud.update_status(
                session, job_id=job_id, status=JobStatus.CANCELLED
            )
            assert updated is not None
            assert updated.status == JobStatus.CANCELLED
            assert updated.completed_at is not None
            break
