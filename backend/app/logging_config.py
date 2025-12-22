"""Structured logging configuration for ModelForge.

Provides JSON-formatted logs in production and human-readable logs in development.
Includes request ID tracking for log correlation.
"""

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any
from contextvars import ContextVar

from app.config import settings

# Context variable for request ID (for log correlation)
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_ctx.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, default=str)


class DevelopmentFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        # Add request ID prefix if available
        request_id = request_id_ctx.get()
        if request_id:
            record.msg = f"[{request_id[:8]}] {record.msg}"
        return super().format(record)


def setup_logging() -> None:
    """Configure application logging based on environment."""
    # Determine log level
    log_level = logging.DEBUG if settings.debug else logging.INFO

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Use JSON formatter in production, human-readable in development
    if settings.is_production:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(DevelopmentFormatter())

    root_logger.addHandler(handler)

    # Configure third-party loggers to be less verbose
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.database_echo else logging.WARNING
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
