"""Celery task for async model inference.

This module contains the Celery task that runs ONNX model inference
asynchronously. The task:
1. Loads the model from storage
2. Runs inference using ONNXService
3. Updates the job record with results or error info
"""

import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.celery import celery_app
from app.config import settings
from app.database import sync_engine
from app.models.job import Job, JobStatus
from app.models.ml_model import MLModel, ModelStatus
from app.services.onnx import ONNXService, ONNXError

logger = logging.getLogger(__name__)


def _get_sync_session() -> Session:
    """Create a synchronous database session for Celery tasks.

    Celery tasks run in a separate process and can't use async sessions.
    """
    return Session(sync_engine)


@celery_app.task(
    bind=True,
    name="app.tasks.inference.run_inference_task",
    max_retries=settings.job_max_retries,
    default_retry_delay=60,  # 1 minute between retries
    autoretry_for=(Exception,),
    dont_autoretry_for=(ONNXError,),  # ONNX errors are permanent, don't retry
    retry_backoff=True,  # Exponential backoff
    retry_backoff_max=600,  # Max 10 minutes between retries
    retry_jitter=True,  # Add randomness to prevent thundering herd
)
def run_inference_task(self, job_id: str) -> dict[str, Any]:
    """Run inference for a job asynchronously.

    This task:
    1. Updates job status to RUNNING
    2. Loads the model and runs inference
    3. Updates job with results (COMPLETED) or error (FAILED)

    Args:
        job_id: UUID of the job to process

    Returns:
        Dict with job_id, status, and output_data or error_message
    """
    logger.info(f"Starting inference task for job {job_id}")
    task_start_time = time.perf_counter()

    with _get_sync_session() as db:
        # Fetch job
        job = db.execute(select(Job).where(Job.id == job_id)).scalar_one_or_none()
        if not job:
            logger.error(f"Job {job_id} not found")
            return {"job_id": job_id, "status": "error", "error": "Job not found"}

        # Calculate queue time (time from job creation to task start)
        queue_time_ms = (datetime.now(timezone.utc) - job.created_at).total_seconds() * 1000

        # Update job status to RUNNING
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        job.queue_time_ms = queue_time_ms
        job.celery_task_id = self.request.id
        job.worker_id = self.request.hostname
        job.retries = self.request.retries
        db.commit()

        try:
            # Fetch model
            model = db.execute(
                select(MLModel).where(MLModel.id == job.model_id)
            ).scalar_one_or_none()

            if not model:
                raise ValueError(f"Model {job.model_id} not found")

            if model.status != ModelStatus.READY:
                raise ValueError(f"Model {job.model_id} is not ready (status: {model.status})")

            if not model.file_path:
                raise ValueError(f"Model {job.model_id} has no file uploaded")

            # Run inference
            onnx_service = ONNXService()
            model_path = Path(settings.model_storage_path) / model.file_path

            logger.info(f"Running inference for job {job_id} using model {model.name}")
            result = onnx_service.run_inference(model_path, job.input_data)

            # Update job with success
            job.status = JobStatus.COMPLETED
            job.output_data = result.outputs
            job.inference_time_ms = result.inference_time_ms
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

            total_time_ms = (time.perf_counter() - task_start_time) * 1000
            logger.info(
                f"Job {job_id} completed successfully in {total_time_ms:.2f}ms "
                f"(queue: {queue_time_ms:.2f}ms, inference: {result.inference_time_ms:.2f}ms)"
            )

            return {
                "job_id": job_id,
                "status": "completed",
                "output_data": result.outputs,
                "inference_time_ms": result.inference_time_ms,
                "queue_time_ms": queue_time_ms,
            }

        except ONNXError as e:
            # ONNX-specific errors (model load, inference, input validation)
            # These are permanent failures - don't retry (excluded via dont_autoretry_for)
            error_msg = str(e)
            logger.error(f"ONNX error for job {job_id}: {error_msg}")

            job.status = JobStatus.FAILED
            job.error_message = error_msg
            job.error_traceback = traceback.format_exc()
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

            return {
                "job_id": job_id,
                "status": "failed",
                "error_message": error_msg,
            }

        except Exception as e:
            # Unexpected errors - Celery will auto-retry via autoretry_for
            error_msg = str(e)
            logger.error(f"Unexpected error for job {job_id}: {error_msg}")

            # Update retry count for tracking
            job.retries = self.request.retries + 1
            db.commit()

            # Re-raise to let Celery handle retry with exponential backoff
            raise

        finally:
            # Safety cleanup: if job is still RUNNING after task exits unexpectedly,
            # mark it as FAILED to prevent it from being stuck forever
            if job.status == JobStatus.RUNNING:
                logger.warning(f"Job {job_id} still in RUNNING state at task exit, marking as FAILED")
                job.status = JobStatus.FAILED
                job.error_message = "Task exited unexpectedly while job was running"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
