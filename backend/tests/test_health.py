"""Tests for health check endpoints."""

from unittest.mock import patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test root endpoint returns app info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "ModelForge"
    assert "version" in data
    assert "docs" in data


@pytest.mark.asyncio
async def test_liveness_probe(client: AsyncClient):
    """Test liveness probe returns alive status."""
    response = await client.get("/api/v1/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


@pytest.mark.asyncio
async def test_readiness_probe(client: AsyncClient):
    """Test readiness probe."""
    response = await client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


class TestHealthEndpoint:
    """Tests for the main /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self, client: AsyncClient):
        """Test health check returns healthy status when DB is connected."""
        # Mock Celery health check since Celery isn't running in tests
        with patch("app.api.health.check_celery_health") as mock_celery:
            mock_celery.return_value = {
                "status": "no_workers",
                "broker_connected": False,
                "workers": {},
                "queues": [],
            }

            response = await client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data
        assert data["database"] == "connected"
        assert "redis" in data
        assert "celery" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_health_check_with_celery_connected(self, client: AsyncClient):
        """Test health check when Celery workers are connected."""
        with patch("app.api.health.check_celery_health") as mock_celery:
            mock_celery.return_value = {
                "status": "connected",
                "broker_connected": True,
                "workers": {"worker@host": {"status": "online"}},
                "queues": ["inference", "default"],
            }

            response = await client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["celery"] == "connected"

    @pytest.mark.asyncio
    async def test_health_check_with_celery_error(self, client: AsyncClient):
        """Test health check when Celery has an error."""
        with patch("app.api.health.check_celery_health") as mock_celery:
            mock_celery.return_value = {
                "status": "error",
                "broker_connected": False,
                "workers": {},
                "queues": [],
                "error": "Connection refused",
            }

            response = await client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        # DB is still connected, so overall status is healthy
        assert data["status"] == "healthy"
        assert data["celery"] == "error"


class TestCeleryHealthEndpoint:
    """Tests for the /health/celery endpoint."""

    @pytest.mark.asyncio
    async def test_celery_health_no_workers(self, client: AsyncClient):
        """Test Celery health when no workers are available."""
        with patch("app.api.health.check_celery_health") as mock_celery:
            mock_celery.return_value = {
                "status": "no_workers",
                "broker_connected": True,
                "workers": {},
                "queues": ["inference", "default"],
            }

            response = await client.get("/api/v1/health/celery")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_workers"
        assert data["broker_connected"] is True
        assert data["workers"] == {}
        assert data["queues"] == ["inference", "default"]
        assert data["error"] is None
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_celery_health_connected_with_workers(self, client: AsyncClient):
        """Test Celery health when workers are connected."""
        with patch("app.api.health.check_celery_health") as mock_celery:
            mock_celery.return_value = {
                "status": "connected",
                "broker_connected": True,
                "workers": {
                    "celery@worker1": {
                        "status": "online",
                        "concurrency": 4,
                        "processed": {"task1": 100},
                    }
                },
                "queues": ["inference", "default"],
            }

            response = await client.get("/api/v1/health/celery")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "connected"
        assert data["broker_connected"] is True
        assert "celery@worker1" in data["workers"]
        assert data["workers"]["celery@worker1"]["status"] == "online"

    @pytest.mark.asyncio
    async def test_celery_health_error(self, client: AsyncClient):
        """Test Celery health when there's a connection error."""
        with patch("app.api.health.check_celery_health") as mock_celery:
            mock_celery.return_value = {
                "status": "error",
                "broker_connected": False,
                "workers": {},
                "queues": [],
                "error": "Connection to broker refused",
            }

            response = await client.get("/api/v1/health/celery")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["broker_connected"] is False
        assert data["error"] == "Connection to broker refused"


class TestReadinessProbe:
    """Tests for the /ready endpoint edge cases."""

    @pytest.mark.asyncio
    async def test_readiness_returns_ready_status(self, client: AsyncClient):
        """Test readiness probe returns ready when DB is connected."""
        response = await client.get("/api/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_readiness_returns_not_ready_on_db_error(self, client: AsyncClient):
        """Test readiness probe returns not_ready when DB fails."""
        from app.main import app
        from app.database import get_db
        from httpx import ASGITransport, AsyncClient as AC

        # Define a failing database dependency
        async def failing_db():
            class FailingSession:
                async def execute(self, query):
                    raise Exception("Database connection failed")
            yield FailingSession()

        # Override the dependency with our failing version
        app.dependency_overrides[get_db] = failing_db

        try:
            transport = ASGITransport(app=app)
            async with AC(transport=transport, base_url="http://test") as test_client:
                response = await test_client.get("/api/v1/ready")
        finally:
            # Clean up override even if test fails
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_ready"
        assert "error" in data


class TestCheckCeleryHealth:
    """Unit tests for the check_celery_health function."""

    def test_celery_health_with_workers(self):
        """Test check_celery_health when workers respond."""
        from app.api.health import check_celery_health

        # Patch at the source module where celery_app is defined
        with patch("app.celery.celery_app") as mock_celery:
            mock_inspect = mock_celery.control.inspect.return_value
            mock_inspect.ping.return_value = {
                "celery@worker1": {"ok": "pong"},
                "celery@worker2": {"ok": "pong"},
            }
            mock_inspect.stats.return_value = {
                "celery@worker1": {
                    "pool": {"max-concurrency": 4},
                    "total": {"task.name": 50},
                },
                "celery@worker2": {
                    "pool": {"max-concurrency": 8},
                    "total": {"task.name": 100},
                },
            }

            result = check_celery_health()

        assert result["status"] == "connected"
        assert result["broker_connected"] is True
        assert len(result["workers"]) == 2
        assert result["workers"]["celery@worker1"]["status"] == "online"
        assert result["workers"]["celery@worker1"]["concurrency"] == 4

    def test_celery_health_no_workers_respond(self):
        """Test check_celery_health when no workers respond."""
        from app.api.health import check_celery_health

        with patch("app.celery.celery_app") as mock_celery:
            mock_inspect = mock_celery.control.inspect.return_value
            mock_inspect.ping.return_value = None

            result = check_celery_health()

        assert result["status"] == "no_workers"
        assert result["broker_connected"] is True
        assert result["workers"] == {}

    def test_celery_health_connection_error(self):
        """Test check_celery_health when connection fails."""
        from app.api.health import check_celery_health

        with patch("app.celery.celery_app") as mock_celery:
            mock_celery.control.inspect.side_effect = Exception("Connection refused")

            result = check_celery_health()

        assert result["status"] == "error"
        assert result["broker_connected"] is False
        assert "Connection refused" in result["error"]
