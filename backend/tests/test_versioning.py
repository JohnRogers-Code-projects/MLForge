"""Tests for model versioning functionality.

These tests verify:
- Unique constraint on (name, version)
- Semantic version comparison logic
- Listing versions by model name
- Getting latest version with optional ready_only filter
"""

import pytest
from httpx import AsyncClient

from app.crud.ml_model import compare_versions, parse_semver


class TestSemverParsing:
    """Tests for semantic version parsing."""

    def test_parse_simple_version(self):
        """Parse standard semver format."""
        assert parse_semver("1.0.0") == (1, 0, 0, "")
        assert parse_semver("2.1.3") == (2, 1, 3, "")
        assert parse_semver("10.20.30") == (10, 20, 30, "")

    def test_parse_version_with_prerelease(self):
        """Parse versions with prerelease tags."""
        assert parse_semver("1.0.0-alpha") == (1, 0, 0, "alpha")
        assert parse_semver("2.0.0-beta.1") == (2, 0, 0, "beta.1")
        assert parse_semver("1.0.0-rc.1") == (1, 0, 0, "rc.1")

    def test_parse_non_semver(self):
        """Non-semver strings get special handling."""
        assert parse_semver("latest") == (0, 0, 0, "latest")
        assert parse_semver("v1") == (0, 0, 0, "v1")
        assert parse_semver("1.0") == (0, 0, 0, "1.0")


class TestVersionComparison:
    """Tests for version comparison logic."""

    def test_compare_equal_versions(self):
        """Equal versions return 0."""
        assert compare_versions("1.0.0", "1.0.0") == 0
        assert compare_versions("2.3.4", "2.3.4") == 0

    def test_compare_major_versions(self):
        """Major version differences."""
        assert compare_versions("2.0.0", "1.0.0") == 1
        assert compare_versions("1.0.0", "2.0.0") == -1

    def test_compare_minor_versions(self):
        """Minor version differences."""
        assert compare_versions("1.2.0", "1.1.0") == 1
        assert compare_versions("1.1.0", "1.2.0") == -1

    def test_compare_patch_versions(self):
        """Patch version differences."""
        assert compare_versions("1.0.2", "1.0.1") == 1
        assert compare_versions("1.0.1", "1.0.2") == -1

    def test_compare_prerelease(self):
        """Prerelease versions are less than stable."""
        assert compare_versions("1.0.0", "1.0.0-alpha") == 1
        assert compare_versions("1.0.0-alpha", "1.0.0") == -1

    def test_compare_prerelease_alphabetically(self):
        """Prerelease tags compare alphabetically."""
        assert compare_versions("1.0.0-beta", "1.0.0-alpha") == 1
        assert compare_versions("1.0.0-alpha", "1.0.0-beta") == -1

    def test_double_digit_versions(self):
        """Handles double-digit version numbers correctly (not string comparison)."""
        # String comparison would say "9" > "10", but semver says 10 > 9
        assert compare_versions("1.10.0", "1.9.0") == 1
        assert compare_versions("1.9.0", "1.10.0") == -1


class TestUniqueConstraint:
    """Tests for the unique (name, version) constraint."""

    @pytest.mark.asyncio
    async def test_create_duplicate_name_version_fails(self, client: AsyncClient):
        """Cannot create two models with same name and version."""
        # Create first model
        response1 = await client.post(
            "/api/v1/models",
            json={"name": "unique-test-model", "version": "1.0.0"},
        )
        assert response1.status_code == 201

        # Try to create duplicate
        response2 = await client.post(
            "/api/v1/models",
            json={"name": "unique-test-model", "version": "1.0.0"},
        )
        assert response2.status_code == 409
        assert "already exists" in response2.json()["detail"]

    @pytest.mark.asyncio
    async def test_same_name_different_versions_allowed(self, client: AsyncClient):
        """Can create multiple versions of the same model name."""
        # Create v1
        response1 = await client.post(
            "/api/v1/models",
            json={"name": "multi-version-model", "version": "1.0.0"},
        )
        assert response1.status_code == 201

        # Create v2
        response2 = await client.post(
            "/api/v1/models",
            json={"name": "multi-version-model", "version": "2.0.0"},
        )
        assert response2.status_code == 201

        # Different IDs
        assert response1.json()["id"] != response2.json()["id"]

    @pytest.mark.asyncio
    async def test_same_version_different_names_allowed(self, client: AsyncClient):
        """Different models can have the same version number."""
        response1 = await client.post(
            "/api/v1/models",
            json={"name": "model-a", "version": "1.0.0"},
        )
        assert response1.status_code == 201

        response2 = await client.post(
            "/api/v1/models",
            json={"name": "model-b", "version": "1.0.0"},
        )
        assert response2.status_code == 201


class TestListVersions:
    """Tests for listing model versions."""

    @pytest.mark.asyncio
    async def test_list_versions_returns_all(self, client: AsyncClient):
        """List versions returns all versions of a model."""
        # Create multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            await client.post(
                "/api/v1/models",
                json={"name": "version-list-model", "version": version},
            )

        response = await client.get("/api/v1/models/by-name/version-list-model/versions")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "version-list-model"
        assert data["total"] == 3
        assert len(data["versions"]) == 3

    @pytest.mark.asyncio
    async def test_list_versions_sorted_by_semver(self, client: AsyncClient):
        """Versions are sorted by semantic version (highest first)."""
        # Create versions out of order
        for version in ["1.0.0", "2.0.0", "1.10.0", "1.9.0"]:
            await client.post(
                "/api/v1/models",
                json={"name": "sorted-versions-model", "version": version},
            )

        response = await client.get("/api/v1/models/by-name/sorted-versions-model/versions")

        assert response.status_code == 200
        versions = [v["version"] for v in response.json()["versions"]]
        # Should be sorted: 2.0.0 > 1.10.0 > 1.9.0 > 1.0.0
        assert versions == ["2.0.0", "1.10.0", "1.9.0", "1.0.0"]

    @pytest.mark.asyncio
    async def test_list_versions_includes_latest(self, client: AsyncClient):
        """Response includes the latest version."""
        for version in ["1.0.0", "3.0.0", "2.0.0"]:
            await client.post(
                "/api/v1/models",
                json={"name": "latest-version-model", "version": version},
            )

        response = await client.get("/api/v1/models/by-name/latest-version-model/versions")

        assert response.status_code == 200
        assert response.json()["latest_version"] == "3.0.0"

    @pytest.mark.asyncio
    async def test_list_versions_nonexistent_model(self, client: AsyncClient):
        """404 for model name that doesn't exist."""
        response = await client.get("/api/v1/models/by-name/nonexistent-model/versions")
        assert response.status_code == 404


class TestGetLatestVersion:
    """Tests for getting the latest model version."""

    @pytest.mark.asyncio
    async def test_get_latest_returns_highest_version(self, client: AsyncClient):
        """Get latest returns the highest semantic version."""
        versions = ["1.0.0", "1.5.0", "2.0.0", "1.10.0"]
        for version in versions:
            response = await client.post(
                "/api/v1/models",
                json={"name": "get-latest-model", "version": version},
            )
            assert response.status_code == 201

        response = await client.get("/api/v1/models/by-name/get-latest-model/latest")

        assert response.status_code == 200
        assert response.json()["version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_get_latest_nonexistent_model(self, client: AsyncClient):
        """404 for model name that doesn't exist."""
        response = await client.get("/api/v1/models/by-name/nonexistent-model/latest")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_latest_ready_only(self, client: AsyncClient):
        """ready_only filter only returns READY models."""
        # Create v1 (will remain pending)
        await client.post(
            "/api/v1/models",
            json={"name": "ready-filter-model", "version": "1.0.0"},
        )

        # Create v2 (will remain pending)
        await client.post(
            "/api/v1/models",
            json={"name": "ready-filter-model", "version": "2.0.0"},
        )

        # Without ready_only, should return v2 (highest)
        response = await client.get("/api/v1/models/by-name/ready-filter-model/latest")
        assert response.status_code == 200
        assert response.json()["version"] == "2.0.0"

        # With ready_only=true, should return 404 (no ready models)
        response = await client.get(
            "/api/v1/models/by-name/ready-filter-model/latest?ready_only=true"
        )
        assert response.status_code == 404
        assert "ready" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_latest_includes_full_model_data(self, client: AsyncClient):
        """Latest endpoint returns full model response."""
        await client.post(
            "/api/v1/models",
            json={
                "name": "full-data-model",
                "version": "1.0.0",
                "description": "Test description",
            },
        )

        response = await client.get("/api/v1/models/by-name/full-data-model/latest")

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "full-data-model"
        assert data["version"] == "1.0.0"
        assert data["description"] == "Test description"
        assert "status" in data
        assert "created_at" in data


class TestVersionStatusTracking:
    """Tests that verify version status is tracked correctly."""

    @pytest.mark.asyncio
    async def test_versions_include_status(self, client: AsyncClient):
        """Version listing includes status for each version."""
        await client.post(
            "/api/v1/models",
            json={"name": "status-tracking-model", "version": "1.0.0"},
        )

        response = await client.get("/api/v1/models/by-name/status-tracking-model/versions")

        assert response.status_code == 200
        version = response.json()["versions"][0]
        assert "status" in version
        assert version["status"] == "pending"  # New models start as pending
