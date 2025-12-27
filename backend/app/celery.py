"""Celery application configuration.

This module configures the Celery distributed task queue for async job processing.
Uses Redis as both the message broker and result backend.
"""

from celery import Celery

from app.config import settings


def create_celery_app() -> Celery:
    """Create and configure Celery application.

    Configuration follows Celery best practices:
    - JSON serialization for security (no pickle)
    - UTC timezone for consistency
    - Task acknowledgement after completion (late ack)
    - Result expiration to prevent Redis bloat
    """
    app = Celery(
        "modelforge",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )

    # Task serialization - JSON only for security (pickle can execute arbitrary code)
    app.conf.task_serializer = "json"
    app.conf.result_serializer = "json"
    app.conf.accept_content = ["json"]

    # Timezone
    app.conf.timezone = "UTC"
    app.conf.enable_utc = True

    # Task execution settings
    app.conf.task_acks_late = True  # Ack after task completes (not when received)
    app.conf.task_reject_on_worker_lost = True  # Requeue if worker dies mid-task
    app.conf.worker_prefetch_multiplier = (
        1  # One task at a time per worker (for long tasks)
    )

    # Time limits (can be overridden per-task)
    app.conf.task_soft_time_limit = settings.celery_task_soft_time_limit
    app.conf.task_time_limit = settings.celery_task_time_limit

    # Result backend settings
    app.conf.result_expires = settings.celery_result_expires  # Clean up old results

    # Task routing - all inference tasks go to the inference queue
    app.conf.task_routes = {
        "app.tasks.inference.*": {"queue": "inference"},
    }

    # Default queue for tasks without explicit routing
    app.conf.task_default_queue = "default"

    # Auto-discover tasks in the tasks module
    app.autodiscover_tasks(["app.tasks"])

    return app


# Global Celery app instance
celery_app = create_celery_app()
