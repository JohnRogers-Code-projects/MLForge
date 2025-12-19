"""API routes for async job management."""

import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import DBSession
from app.crud import job_crud, model_crud
from app.models.job import JobStatus
from app.models.ml_model import ModelStatus
from app.schemas.job import JobCreate, JobListResponse, JobResponse
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
            detail=f"Model {job_in.model_id} is not ready for inference (status: {model.status})",
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
        logger.error(f"Failed to queue task for job {job.id}: {e}")

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


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    db: DBSession,
) -> JobResponse:
    """Cancel a pending or queued job."""
    job = await job_crud.get(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found",
        )

    if job.status not in (JobStatus.PENDING, JobStatus.QUEUED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in {job.status} status",
        )

    updated = await job_crud.update_status(
        db,
        job_id=job_id,
        status=JobStatus.CANCELLED,
    )
    return JobResponse.model_validate(updated)
