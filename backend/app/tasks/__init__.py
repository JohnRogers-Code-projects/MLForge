"""Celery tasks package.

Tasks are auto-discovered by the Celery app.
Import tasks here to ensure they're registered.
"""

from app.tasks.inference import run_inference_task
from app.tasks.cleanup import cleanup_old_jobs

__all__ = ["run_inference_task", "cleanup_old_jobs"]
