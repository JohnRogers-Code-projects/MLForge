"""Celery task for cleaning up old jobs.

This module contains a periodic task that removes old completed/failed/cancelled
jobs based on the configured retention period.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, func

from app.celery import celery_app
from app.config import settings
from app.database import sync_engine
from app.models.job import Job, JobStatus
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Terminal statuses eligible for cleanup
_CLEANUP_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


def _get_sync_session() -> Session:
    """Create a synchronous database session for Celery tasks."""
    return Session(sync_engine)


@celery_app.task(name="app.tasks.cleanup.cleanup_old_jobs")
def cleanup_old_jobs() -> dict:
    """Delete jobs older than the configured retention period.

    Only deletes jobs in terminal states (COMPLETED, FAILED, CANCELLED).
    Jobs in PENDING, QUEUED, or RUNNING states are never deleted.

    Returns:
        Dict with count of deleted jobs and any errors
    """
    retention_days = settings.job_retention_days
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

    logger.info(
        f"Starting job cleanup: removing jobs completed before {cutoff_date} "
        f"(retention: {retention_days} days)"
    )

    with _get_sync_session() as db:
        try:
            # Count jobs to be deleted (for logging)
            count_query = (
                select(func.count())
                .select_from(Job)
                .where(
                    Job.status.in_(_CLEANUP_STATUSES),
                    Job.completed_at < cutoff_date,
                )
            )
            count = db.execute(count_query).scalar() or 0

            if count == 0:
                logger.info("No old jobs to clean up")
                return {"deleted_count": 0, "error": None}

            # Delete old jobs
            delete_query = (
                delete(Job)
                .where(
                    Job.status.in_(_CLEANUP_STATUSES),
                    Job.completed_at < cutoff_date,
                )
            )
            db.execute(delete_query)
            db.commit()

            logger.info(f"Cleaned up {count} old jobs")
            return {"deleted_count": count, "error": None}

        except Exception as e:
            db.rollback()
            error_msg = f"Failed to cleanup old jobs: {e}"
            logger.exception(error_msg)
            return {"deleted_count": 0, "error": error_msg}


# Configure the periodic task schedule
# This runs daily at midnight UTC
celery_app.conf.beat_schedule = celery_app.conf.beat_schedule or {}
celery_app.conf.beat_schedule["cleanup-old-jobs-daily"] = {
    "task": "app.tasks.cleanup.cleanup_old_jobs",
    "schedule": 86400.0,  # 24 hours in seconds
    "options": {"queue": "celery"},
}
