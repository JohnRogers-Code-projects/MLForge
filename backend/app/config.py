"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow model_ prefix for fields like model_storage_path
        protected_namespaces=(),
    )

    # Application
    app_name: str = "ModelForge"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # API
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/modelforge"
    database_echo: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 10
    redis_socket_timeout: float = 5.0  # seconds
    redis_socket_connect_timeout: float = 5.0  # seconds
    redis_retry_on_timeout: bool = True
    redis_health_check_interval: int = 30  # seconds
    redis_enabled: bool = True  # Set to False to disable Redis entirely

    # Cache settings
    cache_ttl: int = 3600  # Default TTL: 1 hour
    cache_key_prefix: str = "modelforge:"  # Namespace for all cache keys

    # Model-specific cache settings
    cache_model_ttl: int = 300  # Model metadata TTL: 5 minutes
    cache_model_list_ttl: int = 60  # Model list TTL: 1 minute (shorter for fresher lists)

    # Prediction cache settings
    cache_prediction_ttl: int = 60  # Prediction TTL: 1 minute (short, model outputs may change)
    cache_prediction_enabled: bool = True  # Enable prediction caching

    # Security
    secret_key: str = "change-me-in-production"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Model storage
    model_storage_path: str = "./models"
    max_model_size_mb: int = 500

    # Celery settings
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    celery_task_soft_time_limit: int = 300  # 5 minutes soft limit (raises SoftTimeLimitExceeded)
    celery_task_time_limit: int = 600  # 10 minutes hard limit (kills task)
    celery_result_expires: int = 86400  # Results expire after 24 hours
    celery_worker_concurrency: int = 2  # Number of worker processes

    # Job settings
    job_retention_days: int = 30  # Keep completed/failed jobs for 30 days
    job_max_retries: int = 3  # Max retry attempts for failed tasks

    @property
    def max_model_size_bytes(self) -> int:
        """Max model size in bytes."""
        return self.max_model_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
