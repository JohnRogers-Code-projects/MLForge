"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.config import settings
from app.database import init_db
from app.services.cache import get_cache_service, close_cache_service


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
    if settings.environment == "development":
        await init_db()

    # Initialize Redis cache (graceful - won't fail if Redis unavailable)
    await get_cache_service()

    yield

    # Shutdown
    await close_cache_service()


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

# CORS middleware
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
