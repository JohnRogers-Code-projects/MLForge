"""Celery worker entry point.

Start the worker with:
    celery -A app.worker worker --loglevel=info

For development with auto-reload:
    celery -A app.worker worker --loglevel=debug

For production with concurrency:
    celery -A app.worker worker --loglevel=info --concurrency=4

With specific queues:
    celery -A app.worker worker --loglevel=info -Q inference,default
"""

from app.celery import celery_app

# Re-export celery_app for the celery CLI
# The -A flag expects a module with a Celery instance
app = celery_app

if __name__ == "__main__":
    # Allow running directly with: python -m app.worker
    celery_app.start()
