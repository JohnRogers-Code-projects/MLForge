"""API routes for async job management."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.api.deps import DBSession
from app.celery import celery_app
from app.crud import job_crud, model_crud
from app.models.job import JobStatus
from app.models.ml_model import ModelStatus
from app.schemas.job import JobCreate, JobListResponse, JobResponse, JobResultResponse
from app.tasks.inference import run_inference_task

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_job(
    job_in: JobCreate,
    db: DBSession,
) -> JobResponse:
    """
    Create a new async inference job.

    Creates a job record and queues it for async processing via Celery.
    The job will transition through states: PENDING -> QUEUED -> RUNNING -> COMPLETED/FAILED
    """
    # Validate model exists and is ready
    model = await model_crud.get(db, job_in.model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {job_in.model_id} not found",
        )

    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model {job_in.model_id} is not ready for inference "
                f"(current status: {model.status}). "
                "Please ensure the model file is uploaded and validated successfully."
            ),
        )

    # Create job in PENDING state
    job = await job_crud.create(db, obj_in=job_in)

    # Queue the Celery task
    try:
        task_result = run_inference_task.delay(job.id)
        logger.info(f"Queued inference task {task_result.id} for job {job.id}")

        # Update job with task ID and status
        job.celery_task_id = task_result.id
        job.status = JobStatus.QUEUED
        await db.flush()
        await db.refresh(job)
    except Exception as e:
        # If queuing fails, leave job in PENDING state
        # (could be picked up by a retry mechanism later)
        logger.warning(
            f"Failed to queue task for job {job.id}: {e}",
            exc_info=True,
        )

    return JobResponse.model_validate(job)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    db: DBSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: JobStatus | None = Query(None, alias="status"),
) -> JobListResponse:
    """List all jobs with optional status filter."""
    offset = (page - 1) * page_size

    if status_filter:
        jobs = await job_crud.get_by_status(
            db,
            status=status_filter,
            offset=offset,
            limit=page_size,
        )
        total = await job_crud.count_by_status(db, status=status_filter)
    else:
        jobs = await job_crud.get_multi(db, offset=offset, limit=page_size)
        total = await job_crud.count(db)

    total_pages = (total + page_size - 1) // page_size

    return JobListResponse(
        items=[JobResponse.model_validate(j) for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: DBSession,
) -> JobResponse:
    """Get a specific job by ID."""
    job = await job_crud.get(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found",
        )
    return JobResponse.model_validate(job)


# Terminal states where the job is considered "done"
_TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
# Processing states where the job is still in-flight
_PROCESSING_STATUSES = {JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING}

# Maximum wait time in seconds (prevent long-running HTTP requests)
_MAX_WAIT_SECONDS = 30
# Polling interval when waiting
_POLL_INTERVAL_SECONDS = 0.5


@router.get(
    "/{job_id}/result",
    response_model=None,  # Disable auto-generation since we return different types
    responses={
        200: {
            "model": JobResultResponse,
            "description": "Job completed (success or failure)",
        },
        202: {"description": "Job still processing"},
        404: {"description": "Job not found"},
    },
)
async def get_job_result(
    job_id: str,
    db: DBSession,
    wait: float = Query(
        default=0,
        ge=0,
        le=_MAX_WAIT_SECONDS,
        description=(
            f"Seconds to wait for job completion (0-{_MAX_WAIT_SECONDS}). "
            "If 0, returns immediately. If > 0, polls until job completes or timeout."
        ),
    ),
):
    """
    Get the result of a job.

    - Returns 200 with result if job completed successfully
    - Returns 200 with error details if job failed
    - Returns 202 if job is still processing
    - Returns 404 if job not found

    Use the `wait` parameter to poll for completion. The server will hold the
    connection open and check the job status periodically until it completes
    or the timeout is reached.
    """
    job = await job_crud.get(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found",
        )

    # If wait > 0, poll until job reaches terminal state or timeout
    if wait > 0 and job.status in _PROCESSING_STATUSES:
        elapsed = 0.0
        while elapsed < wait:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            elapsed += _POLL_INTERVAL_SECONDS

            # Refresh from database to get latest status
            await db.refresh(job)

            if job.status in _TERMINAL_STATUSES:
                break

    # Build response based on final status
    if job.status in _PROCESSING_STATUSES:
        # Still processing - return 202 Accepted
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job.id,
                "status": job.status.value,
                "message": "Job is still processing",
            },
        )

    # Job reached terminal state - return full result
    return JobResultResponse(
        job_id=job.id,
        status=job.status,
        result=job.output_data if job.status == JobStatus.COMPLETED else None,
        error_message=job.error_message if job.status == JobStatus.FAILED else None,
        error_traceback=job.error_traceback if job.status == JobStatus.FAILED else None,
        inference_time_ms=job.inference_time_ms,
        completed_at=job.completed_at,
    )


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    db: DBSession,
) -> JobResponse:
    """Cancel a pending, queued, or running job.

    For queued/running jobs, this will also revoke the Celery task to stop
    execution. The task will be terminated if currently running.
    """
    job = await job_crud.get(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found",
        )

    # Allow cancellation of PENDING, QUEUED, or RUNNING jobs
    if job.status not in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in {job.status} status",
        )

    # Revoke the Celery task if it exists
    if job.celery_task_id:
        try:
            # terminate=True sends SIGTERM to running tasks
            celery_app.control.revoke(job.celery_task_id, terminate=True)
            logger.info(f"Revoked Celery task {job.celery_task_id} for job {job_id}")
        except Exception as e:
            # Log but don't fail - the job cancellation should still proceed
            logger.warning(f"Failed to revoke Celery task {job.celery_task_id}: {e}")

    updated = await job_crud.update_status(
        db,
        job_id=job_id,
        status=JobStatus.CANCELLED,
    )
    return JobResponse.model_validate(updated)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: str,
    db: DBSession,
) -> None:
    """Delete a job and its associated data.

    Only completed, failed, or cancelled jobs can be deleted.
    Running or queued jobs must be cancelled first.
    """
    job = await job_crud.get(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found",
        )

    # Only allow deletion of terminal state jobs
    if job.status not in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete job in {job.status} status. Cancel it first.",
        )

    await job_crud.delete(db, id=job_id)
    logger.info(f"Deleted job {job_id}")
