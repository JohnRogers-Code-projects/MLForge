"""Celery tasks package.

Tasks are auto-discovered by the Celery app.
Import tasks here to ensure they're registered.
"""

from app.tasks.inference import run_inference_task

__all__ = ["run_inference_task"]
