"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.config import settings
from app.database import init_db
from app.logging_config import get_logger, setup_logging
from app.middleware import RequestLoggingMiddleware
from app.services.cache import close_cache_service, get_cache_service

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Optional Sentry integration for error alerting
if settings.sentry_dsn:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.celery import CeleryIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            release=settings.app_version,
            traces_sample_rate=0.1,  # Sample 10% of requests for performance monitoring
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                CeleryIntegration(),
            ],
        )
        logger.info("Sentry error tracking initialized")
    except ImportError:
        logger.error(
            "SENTRY_DSN is set but sentry-sdk is not installed; error alerting disabled"
        )


# OpenAPI tag metadata for better API documentation
OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Health check and monitoring endpoints. Use these for Kubernetes probes and service health monitoring.",
    },
    {
        "name": "models",
        "description": "ONNX model management. Upload, validate, and manage ML models with versioning support.",
    },
    {
        "name": "predictions",
        "description": "Synchronous inference endpoints. Run predictions on uploaded and validated models.",
    },
    {
        "name": "jobs",
        "description": "Async job queue management. Create, monitor, and manage long-running inference jobs.",
    },
    {
        "name": "cache",
        "description": "Cache management and metrics. Monitor cache performance and clear cached data.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    # Startup
    logger.info(
        "Starting ModelForge API",
        extra={
            "extra_fields": {
                "version": settings.app_version,
                "environment": settings.environment,
            }
        },
    )

    if settings.environment == "development":
        await init_db()

    # Initialize Redis cache (graceful - won't fail if Redis unavailable)
    await get_cache_service()
    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down ModelForge API")
    await close_cache_service()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
# ModelForge API

ML Model Serving Platform for deploying and serving ONNX models at scale.

## Features

- **Model Management**: Upload, validate, and version ONNX models
- **Synchronous Inference**: Low-latency predictions with Redis caching
- **Async Job Queue**: Long-running inference via Celery workers
- **Health Monitoring**: Comprehensive health checks for all services

## Authentication

Currently, the API does not require authentication. This is suitable for
internal deployments behind a VPN or API gateway.

## Rate Limiting

No rate limiting is currently enforced. Consider adding rate limiting
at the API gateway level for production deployments.
    """,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    contact={
        "name": "ModelForge Team",
        "url": "https://github.com/JohnRogers-Code-projects/MLForge",
    },
)

# Request logging middleware (adds request ID and timing)
# Note: Middleware is applied in reverse order, so this runs after CORS
app.add_middleware(RequestLoggingMiddleware)

# CORS middleware (outermost, runs first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": f"{settings.api_prefix}/docs",
    }
