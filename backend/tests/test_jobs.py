"""Tests for job endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_job(client: AsyncClient):
    """Test creating a new job."""
    # First create a model
    model_response = await client.post(
        "/api/v1/models",
        json={"name": "job-test-model", "version": "1.0.0"},
    )
    model_id = model_response.json()["id"]

    # Create a job
    job_data = {
        "model_id": model_id,
        "input_data": {"feature1": 1.0, "feature2": 2.0},
        "priority": "normal",
    }
    response = await client.post("/api/v1/jobs", json=job_data)
    assert response.status_code == 201
    data = response.json()
    assert data["model_id"] == model_id
    assert data["status"] == "pending"
    assert data["priority"] == "normal"
    assert "id" in data


@pytest.mark.asyncio
async def test_create_job_nonexistent_model(client: AsyncClient):
    """Test creating a job for a nonexistent model."""
    job_data = {
        "model_id": "nonexistent-model-id",
        "input_data": {"feature1": 1.0},
    }
    response = await client.post("/api/v1/jobs", json=job_data)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_jobs(client: AsyncClient):
    """Test listing jobs."""
    # Create a model and some jobs
    model_response = await client.post(
        "/api/v1/models",
        json={"name": "list-jobs-model", "version": "1.0.0"},
    )
    model_id = model_response.json()["id"]

    for i in range(3):
        await client.post(
            "/api/v1/jobs",
            json={"model_id": model_id, "input_data": {"value": i}},
        )

    response = await client.get("/api/v1/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) >= 3


@pytest.mark.asyncio
async def test_list_jobs_by_status(client: AsyncClient):
    """Test filtering jobs by status."""
    response = await client.get("/api/v1/jobs?status=pending")
    assert response.status_code == 200
    data = response.json()
    for job in data["items"]:
        assert job["status"] == "pending"


@pytest.mark.asyncio
async def test_get_job(client: AsyncClient):
    """Test getting a specific job."""
    # Create a model and job
    model_response = await client.post(
        "/api/v1/models",
        json={"name": "get-job-model", "version": "1.0.0"},
    )
    model_id = model_response.json()["id"]

    job_response = await client.post(
        "/api/v1/jobs",
        json={"model_id": model_id, "input_data": {"test": True}},
    )
    job_id = job_response.json()["id"]

    # Get the job
    response = await client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id


@pytest.mark.asyncio
async def test_cancel_job(client: AsyncClient):
    """Test cancelling a pending job."""
    # Create a model and job
    model_response = await client.post(
        "/api/v1/models",
        json={"name": "cancel-job-model", "version": "1.0.0"},
    )
    model_id = model_response.json()["id"]

    job_response = await client.post(
        "/api/v1/jobs",
        json={"model_id": model_id, "input_data": {"test": True}},
    )
    job_id = job_response.json()["id"]

    # Cancel the job
    response = await client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"
