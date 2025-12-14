"""Tests for ML model endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_model(client: AsyncClient):
    """Test creating a new model."""
    model_data = {
        "name": "test-model",
        "description": "A test model",
        "version": "1.0.0",
    }
    response = await client.post("/api/v1/models", json=model_data)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == model_data["name"]
    assert data["description"] == model_data["description"]
    assert data["version"] == model_data["version"]
    assert data["status"] == "pending"
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_duplicate_model(client: AsyncClient):
    """Test creating a duplicate model returns conflict."""
    model_data = {
        "name": "duplicate-model",
        "version": "1.0.0",
    }
    # Create first model
    response = await client.post("/api/v1/models", json=model_data)
    assert response.status_code == 201

    # Try to create duplicate
    response = await client.post("/api/v1/models", json=model_data)
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_list_models(client: AsyncClient):
    """Test listing models with pagination."""
    # Create a few models
    for i in range(3):
        await client.post(
            "/api/v1/models",
            json={"name": f"list-test-model-{i}", "version": "1.0.0"},
        )

    response = await client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert len(data["items"]) >= 3


@pytest.mark.asyncio
async def test_get_model(client: AsyncClient):
    """Test getting a specific model."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "get-test-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Get the model
    response = await client.get(f"/api/v1/models/{model_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == model_id
    assert data["name"] == "get-test-model"


@pytest.mark.asyncio
async def test_get_nonexistent_model(client: AsyncClient):
    """Test getting a model that doesn't exist."""
    response = await client.get("/api/v1/models/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_model(client: AsyncClient):
    """Test updating a model."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "update-test-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Update the model
    update_data = {"description": "Updated description"}
    response = await client.patch(f"/api/v1/models/{model_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "Updated description"


@pytest.mark.asyncio
async def test_delete_model(client: AsyncClient):
    """Test deleting a model."""
    # Create a model
    create_response = await client.post(
        "/api/v1/models",
        json={"name": "delete-test-model", "version": "1.0.0"},
    )
    model_id = create_response.json()["id"]

    # Delete the model
    response = await client.delete(f"/api/v1/models/{model_id}")
    assert response.status_code == 204

    # Verify it's deleted
    get_response = await client.get(f"/api/v1/models/{model_id}")
    assert get_response.status_code == 404
