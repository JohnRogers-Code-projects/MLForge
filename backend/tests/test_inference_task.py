"""Tests for Celery inference task.

These tests verify the run_inference_task Celery task behaves correctly,
including status transitions, timing metrics, error handling, and retries.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.models.job import Job, JobPriority, JobStatus
from app.models.ml_model import MLModel, ModelStatus
from app.services.onnx import InferenceResult, ONNXError


def create_mock_job():
    """Create a mock job for testing."""
    job = MagicMock(spec=Job)
    job.id = str(uuid4())
    job.model_id = str(uuid4())
    job.status = JobStatus.QUEUED
    job.priority = JobPriority.NORMAL
    job.input_data = {"input": [[1.0] * 10]}
    job.output_data = None
    job.celery_task_id = None
    job.worker_id = None
    job.retries = 0
    job.error_message = None
    job.error_traceback = None
    job.inference_time_ms = None
    job.queue_time_ms = None
    job.created_at = datetime.now(UTC)
    job.started_at = None
    job.completed_at = None
    return job


def create_mock_model(model_id):
    """Create a mock model for testing."""
    model = MagicMock(spec=MLModel)
    model.id = model_id
    model.name = "test-model"
    model.status = ModelStatus.READY
    model.file_path = "test-model.onnx"
    return model


class TestRunInferenceTask:
    """Tests for the run_inference_task Celery task."""

    def test_task_successful_inference(self):
        """Test successful inference updates job correctly."""
        from app.tasks.inference import run_inference_task

        mock_job = create_mock_job()
        mock_model = create_mock_model(mock_job.model_id)
        mock_inference_result = InferenceResult(
            outputs={"output": [[2.0] * 10]},
            inference_time_ms=15.5,
        )

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [
            mock_job,  # First call: get job
            mock_model,  # Second call: get model
        ]

        mock_onnx_service = MagicMock()
        mock_onnx_service.run_inference.return_value = mock_inference_result

        with (
            patch("app.tasks.inference._get_sync_session") as mock_session,
            patch("app.tasks.inference.ONNXService") as mock_onnx_class,
            patch("app.tasks.inference.settings") as mock_settings,
        ):
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_onnx_class.return_value = mock_onnx_service
            mock_settings.model_storage_path = "/models"
            mock_settings.job_max_retries = 3

            # Configure eager mode for testing
            from app.celery import celery_app

            celery_app.conf.task_always_eager = True

            # Run with apply() to execute synchronously
            result = run_inference_task.apply(args=[mock_job.id])

        # Verify result
        assert result.successful()
        result_data = result.result
        assert result_data["status"] == "completed"
        assert result_data["output_data"] == {"output": [[2.0] * 10]}

    def test_task_job_not_found(self):
        """Test task handles missing job gracefully."""
        from app.tasks.inference import run_inference_task

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = None

        with (
            patch("app.tasks.inference._get_sync_session") as mock_session,
        ):
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            from app.celery import celery_app

            celery_app.conf.task_always_eager = True

            result = run_inference_task.apply(args=["nonexistent-job-id"])

        assert result.successful()
        assert result.result["status"] == "error"
        assert result.result["error"] == "Job not found"

    def test_task_model_not_found_triggers_retry(self):
        """Test task triggers retry when model not found (may be transient)."""
        from celery.exceptions import Retry

        from app.tasks.inference import run_inference_task

        mock_job = create_mock_job()

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [
            mock_job,  # Job found
            None,  # Model not found
        ]

        with (
            patch("app.tasks.inference._get_sync_session") as mock_session,
            patch("app.tasks.inference.settings") as mock_settings,
        ):
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_settings.job_max_retries = 3

            from app.celery import celery_app

            celery_app.conf.task_always_eager = True
            celery_app.conf.task_eager_propagates = True

            # Unexpected errors trigger Celery retry mechanism
            with pytest.raises(Retry):
                run_inference_task.apply(args=[mock_job.id], throw=True)

    def test_task_model_not_ready_triggers_retry(self):
        """Test task triggers retry when model not ready (may be transient)."""
        from celery.exceptions import Retry

        from app.tasks.inference import run_inference_task

        mock_job = create_mock_job()
        mock_model = create_mock_model(mock_job.model_id)
        mock_model.status = ModelStatus.UPLOADED  # Not READY

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [
            mock_job,
            mock_model,
        ]

        with (
            patch("app.tasks.inference._get_sync_session") as mock_session,
            patch("app.tasks.inference.settings") as mock_settings,
        ):
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_settings.job_max_retries = 3

            from app.celery import celery_app

            celery_app.conf.task_always_eager = True
            celery_app.conf.task_eager_propagates = True

            # Unexpected errors trigger Celery retry mechanism
            with pytest.raises(Retry):
                run_inference_task.apply(args=[mock_job.id], throw=True)

    def test_task_model_no_file_triggers_retry(self):
        """Test task triggers retry when model has no file (may be transient)."""
        from celery.exceptions import Retry

        from app.tasks.inference import run_inference_task

        mock_job = create_mock_job()
        mock_model = create_mock_model(mock_job.model_id)
        mock_model.file_path = None  # No file

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [
            mock_job,
            mock_model,
        ]

        with (
            patch("app.tasks.inference._get_sync_session") as mock_session,
            patch("app.tasks.inference.settings") as mock_settings,
        ):
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_settings.job_max_retries = 3

            from app.celery import celery_app

            celery_app.conf.task_always_eager = True
            celery_app.conf.task_eager_propagates = True

            # Unexpected errors trigger Celery retry mechanism
            with pytest.raises(Retry):
                run_inference_task.apply(args=[mock_job.id], throw=True)

    def test_task_onnx_error_marks_job_failed(self):
        """Test ONNX errors mark job as FAILED without retry."""
        from app.tasks.inference import run_inference_task

        mock_job = create_mock_job()
        mock_model = create_mock_model(mock_job.model_id)

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [
            mock_job,
            mock_model,
        ]

        mock_onnx_service = MagicMock()
        mock_onnx_service.run_inference.side_effect = ONNXError("Invalid input shape")

        with (
            patch("app.tasks.inference._get_sync_session") as mock_session,
            patch("app.tasks.inference.ONNXService") as mock_onnx_class,
            patch("app.tasks.inference.settings") as mock_settings,
        ):
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_onnx_class.return_value = mock_onnx_service
            mock_settings.model_storage_path = "/models"
            mock_settings.job_max_retries = 3

            from app.celery import celery_app

            celery_app.conf.task_always_eager = True

            result = run_inference_task.apply(args=[mock_job.id])

        assert result.successful()
        assert result.result["status"] == "failed"
        assert "Invalid input shape" in result.result["error_message"]
        assert mock_job.status == JobStatus.FAILED

    def test_task_tracks_timing_metrics(self):
        """Test task tracks queue_time_ms and inference_time_ms."""
        from app.tasks.inference import run_inference_task

        mock_job = create_mock_job()
        mock_model = create_mock_model(mock_job.model_id)
        mock_inference_result = InferenceResult(
            outputs={"output": [[2.0] * 10]},
            inference_time_ms=15.5,
        )

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [
            mock_job,
            mock_model,
        ]

        mock_onnx_service = MagicMock()
        mock_onnx_service.run_inference.return_value = mock_inference_result

        with (
            patch("app.tasks.inference._get_sync_session") as mock_session,
            patch("app.tasks.inference.ONNXService") as mock_onnx_class,
            patch("app.tasks.inference.settings") as mock_settings,
        ):
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_onnx_class.return_value = mock_onnx_service
            mock_settings.model_storage_path = "/models"
            mock_settings.job_max_retries = 3

            from app.celery import celery_app

            celery_app.conf.task_always_eager = True

            result = run_inference_task.apply(args=[mock_job.id])

        # Check timing metrics in result
        result_data = result.result
        assert "queue_time_ms" in result_data
        assert result_data["queue_time_ms"] >= 0
        assert "inference_time_ms" in result_data
        assert result_data["inference_time_ms"] == 15.5

        # Check job has timing metrics
        assert mock_job.queue_time_ms is not None
        assert mock_job.inference_time_ms == 15.5

    def test_task_stores_worker_info(self):
        """Test task stores celery_task_id and worker_id."""
        from app.tasks.inference import run_inference_task

        mock_job = create_mock_job()
        mock_model = create_mock_model(mock_job.model_id)
        mock_inference_result = InferenceResult(
            outputs={"output": [[2.0] * 10]},
            inference_time_ms=15.5,
        )

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [
            mock_job,
            mock_model,
        ]

        mock_onnx_service = MagicMock()
        mock_onnx_service.run_inference.return_value = mock_inference_result

        with (
            patch("app.tasks.inference._get_sync_session") as mock_session,
            patch("app.tasks.inference.ONNXService") as mock_onnx_class,
            patch("app.tasks.inference.settings") as mock_settings,
        ):
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_onnx_class.return_value = mock_onnx_service
            mock_settings.model_storage_path = "/models"
            mock_settings.job_max_retries = 3

            from app.celery import celery_app

            celery_app.conf.task_always_eager = True

            run_inference_task.apply(args=[mock_job.id])

        # In eager mode, task_id and hostname are set
        assert mock_job.celery_task_id is not None
        # Worker ID might be None in eager mode
        assert mock_job.status == JobStatus.COMPLETED

    def test_task_stores_error_traceback(self):
        """Test task stores error traceback on failure."""
        from app.tasks.inference import run_inference_task

        mock_job = create_mock_job()
        mock_model = create_mock_model(mock_job.model_id)

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [
            mock_job,
            mock_model,
        ]

        mock_onnx_service = MagicMock()
        mock_onnx_service.run_inference.side_effect = ONNXError("Model failed to load")

        with (
            patch("app.tasks.inference._get_sync_session") as mock_session,
            patch("app.tasks.inference.ONNXService") as mock_onnx_class,
            patch("app.tasks.inference.settings") as mock_settings,
        ):
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_onnx_class.return_value = mock_onnx_service
            mock_settings.model_storage_path = "/models"
            mock_settings.job_max_retries = 3

            from app.celery import celery_app

            celery_app.conf.task_always_eager = True

            run_inference_task.apply(args=[mock_job.id])

        assert mock_job.error_traceback is not None
        assert "ONNXError" in mock_job.error_traceback


class TestTaskConfiguration:
    """Tests for task configuration and routing."""

    def test_task_registered_with_celery(self):
        """Test task is registered with Celery app."""
        from app.celery import celery_app

        assert "app.tasks.inference.run_inference_task" in celery_app.tasks

    def test_task_has_retry_configuration(self):
        """Test task has retry configuration."""
        from app.tasks.inference import run_inference_task

        # Check retry settings are configured
        assert hasattr(run_inference_task, "max_retries")
        assert hasattr(run_inference_task, "default_retry_delay")

    def test_task_routes_to_inference_queue(self):
        """Test task is routed to inference queue."""
        from app.celery import celery_app

        routes = celery_app.conf.task_routes
        assert "app.tasks.inference.*" in routes
        assert routes["app.tasks.inference.*"]["queue"] == "inference"


class TestTaskExport:
    """Tests for task exports."""

    def test_task_exported_from_tasks_package(self):
        """Test run_inference_task is exported from tasks package."""
        from app.tasks import run_inference_task

        assert run_inference_task is not None
        assert callable(run_inference_task.delay)

    def test_task_has_correct_name(self):
        """Test task has expected name."""
        from app.tasks.inference import run_inference_task

        assert run_inference_task.name == "app.tasks.inference.run_inference_task"
