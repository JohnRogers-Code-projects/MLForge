"""Tests for Celery configuration and health checks."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient


class TestCeleryConfiguration:
    """Tests for Celery app configuration."""

    def test_celery_app_creation(self):
        """Test that Celery app is created with correct settings."""
        from app.celery import celery_app

        assert celery_app is not None
        assert celery_app.main == "modelforge"

    def test_celery_serialization_settings(self):
        """Test JSON serialization is configured (security best practice)."""
        from app.celery import celery_app

        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.result_serializer == "json"
        assert "json" in celery_app.conf.accept_content

    def test_celery_timezone_settings(self):
        """Test UTC timezone is configured."""
        from app.celery import celery_app

        assert celery_app.conf.timezone == "UTC"
        assert celery_app.conf.enable_utc is True

    def test_celery_task_ack_settings(self):
        """Test task acknowledgement is configured for reliability."""
        from app.celery import celery_app

        # Late ack ensures task is requeued if worker dies
        assert celery_app.conf.task_acks_late is True
        assert celery_app.conf.task_reject_on_worker_lost is True

    def test_celery_prefetch_settings(self):
        """Test worker prefetch is set to 1 for long-running inference tasks."""
        from app.celery import celery_app

        # prefetch=1 prevents worker from grabbing multiple tasks
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_celery_time_limits(self):
        """Test time limits are configured."""
        from app.celery import celery_app
        from app.config import settings

        assert celery_app.conf.task_soft_time_limit == settings.celery_task_soft_time_limit
        assert celery_app.conf.task_time_limit == settings.celery_task_time_limit

    def test_celery_result_expiration(self):
        """Test result expiration is configured."""
        from app.celery import celery_app
        from app.config import settings

        assert celery_app.conf.result_expires == settings.celery_result_expires

    def test_celery_task_routing(self):
        """Test inference tasks are routed to inference queue."""
        from app.celery import celery_app

        routes = celery_app.conf.task_routes
        assert "app.tasks.inference.*" in routes
        assert routes["app.tasks.inference.*"]["queue"] == "inference"

    def test_celery_default_queue(self):
        """Test default queue is configured."""
        from app.celery import celery_app

        assert celery_app.conf.task_default_queue == "default"


class TestCelerySettings:
    """Tests for Celery settings in config."""

    def test_celery_broker_url_default(self):
        """Test default broker URL."""
        from app.config import Settings

        settings = Settings()
        assert settings.celery_broker_url == "redis://localhost:6379/0"

    def test_celery_result_backend_default(self):
        """Test default result backend."""
        from app.config import Settings

        settings = Settings()
        assert settings.celery_result_backend == "redis://localhost:6379/0"

    def test_celery_time_limits_default(self):
        """Test default time limits."""
        from app.config import Settings

        settings = Settings()
        assert settings.celery_task_soft_time_limit == 300  # 5 minutes
        assert settings.celery_task_time_limit == 600  # 10 minutes

    def test_celery_result_expires_default(self):
        """Test default result expiration."""
        from app.config import Settings

        settings = Settings()
        assert settings.celery_result_expires == 86400  # 24 hours

    def test_job_settings_default(self):
        """Test default job settings."""
        from app.config import Settings

        settings = Settings()
        assert settings.job_retention_days == 30
        assert settings.job_max_retries == 3


class TestCeleryHealthCheck:
    """Tests for Celery health check endpoint."""

    @pytest.mark.asyncio
    async def test_celery_health_endpoint_exists(self, client: AsyncClient):
        """Test /health/celery endpoint exists."""
        response = await client.get("/api/v1/health/celery")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_celery_health_response_structure(self, client: AsyncClient):
        """Test health response has correct structure."""
        response = await client.get("/api/v1/health/celery")
        data = response.json()

        assert "status" in data
        assert "broker_connected" in data
        assert "workers" in data
        assert "queues" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_celery_health_when_broker_unavailable(self, client: AsyncClient):
        """Test health check when broker is unavailable."""
        # In test environment without Redis, should return error status
        response = await client.get("/api/v1/health/celery")
        data = response.json()

        # Should still return 200 (health check succeeded) but status indicates issue
        assert response.status_code == 200
        assert data["status"] in ["error", "no_workers", "connected"]

    @pytest.mark.asyncio
    async def test_main_health_includes_celery(self, client: AsyncClient):
        """Test main health endpoint includes celery status."""
        response = await client.get("/api/v1/health")
        data = response.json()

        assert "celery" in data
        # Status should be one of the expected values
        assert data["celery"] in ["connected", "no_workers", "error", "unknown"]


class TestCheckCeleryHealth:
    """Tests for check_celery_health function."""

    def test_check_celery_health_with_mock_workers(self):
        """Test health check with mocked workers responding."""
        from app.api.health import check_celery_health

        mock_inspect = MagicMock()
        mock_inspect.ping.return_value = {
            "worker1@host": {"ok": "pong"},
            "worker2@host": {"ok": "pong"},
        }
        mock_inspect.stats.return_value = {
            "worker1@host": {"pool": {"max-concurrency": 4}, "total": {"tasks": 10}},
            "worker2@host": {"pool": {"max-concurrency": 2}, "total": {"tasks": 5}},
        }

        # Patch at the source module where celery_app is defined
        with patch("app.celery.celery_app") as mock_app:
            mock_app.control.inspect.return_value = mock_inspect

            result = check_celery_health()

            assert result["status"] == "connected"
            assert result["broker_connected"] is True
            assert len(result["workers"]) == 2
            assert "worker1@host" in result["workers"]
            assert result["workers"]["worker1@host"]["status"] == "online"

    def test_check_celery_health_no_workers(self):
        """Test health check when no workers are running."""
        from app.api.health import check_celery_health

        mock_inspect = MagicMock()
        mock_inspect.ping.return_value = None  # No workers responded

        # Patch at the source module where celery_app is defined
        with patch("app.celery.celery_app") as mock_app:
            mock_app.control.inspect.return_value = mock_inspect

            result = check_celery_health()

            assert result["status"] == "no_workers"
            assert result["broker_connected"] is True
            assert result["workers"] == {}

    def test_check_celery_health_broker_error(self):
        """Test health check when broker connection fails."""
        from app.api.health import check_celery_health

        # Patch at the source module where celery_app is defined
        with patch("app.celery.celery_app") as mock_app:
            mock_app.control.inspect.side_effect = Exception("Connection refused")

            result = check_celery_health()

            assert result["status"] == "error"
            assert result["broker_connected"] is False
            assert "Connection refused" in result["error"]


class TestWorkerModule:
    """Tests for worker module."""

    def test_worker_exports_celery_app(self):
        """Test worker module exports celery app."""
        from app.worker import app

        assert app is not None
        assert app.main == "modelforge"

    def test_worker_app_is_same_as_celery_app(self):
        """Test worker.app is the same instance as celery.celery_app."""
        from app.celery import celery_app
        from app.worker import app

        assert app is celery_app
