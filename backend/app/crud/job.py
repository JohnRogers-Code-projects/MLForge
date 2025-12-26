"""CRUD operations for jobs."""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.job import Job, JobStatus
from app.schemas.job import JobCreate, JobStatusUpdate


class CRUDJob(CRUDBase[Job, JobCreate, JobStatusUpdate]):
    """CRUD operations for Job."""

    async def get_by_model(
        self,
        db: AsyncSession,
        *,
        model_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """Get jobs for a specific model."""
        result = await db.execute(
            select(Job)
            .where(Job.model_id == model_id)
            .offset(offset)
            .limit(limit)
            .order_by(Job.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_status(
        self,
        db: AsyncSession,
        *,
        status: JobStatus,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """Get jobs by status."""
        result = await db.execute(
            select(Job)
            .where(Job.status == status)
            .offset(offset)
            .limit(limit)
            .order_by(Job.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_pending_jobs(
        self,
        db: AsyncSession,
        *,
        limit: int = 10,
    ) -> list[Job]:
        """Get pending jobs ordered by priority and creation time."""
        result = await db.execute(
            select(Job)
            .where(Job.status == JobStatus.PENDING)
            .order_by(Job.priority.desc(), Job.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_status(
        self,
        db: AsyncSession,
        *,
        status: JobStatus,
    ) -> int:
        """Count jobs by status."""
        result = await db.execute(
            select(func.count()).select_from(Job).where(Job.status == status)
        )
        return result.scalar() or 0

    async def update_status(
        self,
        db: AsyncSession,
        *,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
    ) -> Job | None:
        """Update job status with timestamps."""
        job = await self.get(db, job_id)
        if not job:
            return None

        job.status = status
        if error_message:
            job.error_message = error_message

        if status == JobStatus.RUNNING:
            job.started_at = datetime.now(UTC)
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            job.completed_at = datetime.now(UTC)

        await db.flush()
        await db.refresh(job)
        return job


job_crud = CRUDJob(Job)
