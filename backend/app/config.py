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
    cache_ttl: int = 3600  # 1 hour default

    # Security
    secret_key: str = "change-me-in-production"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Model storage
    model_storage_path: str = "./models"
    max_model_size_mb: int = 500

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
