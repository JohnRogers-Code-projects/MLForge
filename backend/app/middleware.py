"""FastAPI middleware for request logging and monitoring."""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.logging_config import get_logger, request_id_ctx

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging with timing and request ID tracking."""

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Set request ID in context for log correlation
        token = request_id_ctx.set(request_id)

        # Record start time
        start_time = time.perf_counter()

        # Get client IP (check X-Forwarded-For for proxied requests)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else None

        # Log incoming request (note: query params may contain sensitive data)
        logger.info(
            "Request started",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": client_ip,
                }
            },
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": round(duration_ms, 2),
                    }
                },
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Calculate duration even on error
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.exception(
                "Request failed",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": round(duration_ms, 2),
                        "error": str(e),
                    }
                },
            )
            raise

        finally:
            # Reset request ID context
            request_id_ctx.reset(token)
