"""Tests for health check endpoints."""

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
