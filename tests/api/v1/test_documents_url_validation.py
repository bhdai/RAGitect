"""Integration tests for URL validation in document upload API.

Tests the POST /api/v1/workspaces/{workspace_id}/documents/upload-url endpoint
for proper URL validation, SSRF prevention, and discriminated union routing (AC5, AC6).
"""

import uuid

import pytest
from httpx import AsyncClient


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestURLUploadEndpoint:
    """Integration tests for URL upload endpoint."""

    async def test_valid_url_submission_returns_201(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """Valid URL submission should return 201 Created with document info."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "https://example.com/page",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["sourceType"] == "url"
        assert data["sourceUrl"] == "https://example.com/page"
        assert data["status"] == "backlog"
        assert "message" in data

    async def test_youtube_url_submission_returns_201(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """YouTube URL submission should return 201 with youtube source type."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "youtube",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["sourceType"] == "youtube"

    async def test_pdf_url_submission_returns_201(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """PDF URL submission should return 201 with pdf source type."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "pdf",
                "url": "https://example.com/document.pdf",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["sourceType"] == "pdf"


class TestURLUploadSchemeValidation:
    """Tests for URL scheme validation (AC2)."""

    async def test_file_scheme_returns_422(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """file:// scheme should be rejected with 400 and required message."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "file:///etc/passwd",
            },
        )

        assert response.status_code == 400
        assert "Only HTTP and HTTPS URLs are allowed" in response.json()["detail"]

    async def test_ftp_scheme_returns_400(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """ftp:// scheme should be rejected."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "ftp://server.com/file.txt",
            },
        )

        assert response.status_code == 400
        assert "Only HTTP and HTTPS URLs are allowed" in response.json()["detail"]


class TestURLUploadSSRFPrevention:
    """Tests for SSRF attack prevention (AC3)."""

    async def test_localhost_returns_400(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """localhost URL should be rejected with 400 and security error."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "http://localhost:8080/admin",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "Private and localhost URLs are not allowed" in data["detail"]

    async def test_127_0_0_1_returns_400(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """127.0.0.1 should be rejected as SSRF attempt."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "http://127.0.0.1/secret",
            },
        )

        assert response.status_code == 400
        assert "Private and localhost" in response.json()["detail"]

    async def test_private_ip_192_168_returns_400(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """192.168.x.x private IP should be rejected."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "http://192.168.1.1/admin",
            },
        )

        assert response.status_code == 400
        assert "Private and localhost" in response.json()["detail"]

    async def test_private_ip_10_x_returns_400(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """10.x.x.x private IP should be rejected."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "http://10.0.0.1/internal",
            },
        )

        assert response.status_code == 400

    async def test_cloud_metadata_endpoint_returns_400(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """Cloud metadata endpoint (169.254.169.254) should be blocked."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "http://169.254.169.254/latest/meta-data/",
            },
        )

        assert response.status_code == 400


class TestURLUploadValidation:
    """Tests for general validation and error handling."""

    async def test_duplicate_url_submission_returns_409(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """Submitting the same URL twice should return 409 Conflict."""
        url = "https://example.com/duplicate-test"

        first = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": url,
            },
        )
        assert first.status_code == 201

        second = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": url,
            },
        )
        assert second.status_code == 409

    async def test_malformed_url_returns_422(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """Malformed URL should return 422 Unprocessable Entity (Pydantic)."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "not-a-valid-url",
            },
        )

        assert response.status_code == 422

    async def test_missing_url_returns_422(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """Missing URL field should return 422."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                # url field missing
            },
        )

        assert response.status_code == 422

    async def test_invalid_source_type_returns_422(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """Invalid source type should return 422."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "invalid_type",
                "url": "https://example.com",
            },
        )

        assert response.status_code == 422

    async def test_workspace_not_found_returns_404(
        self, shared_integration_client: AsyncClient
    ) -> None:
        """Non-existent workspace should return 404."""
        fake_workspace_id = uuid.uuid4()

        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{fake_workspace_id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "https://example.com",
            },
        )

        assert response.status_code == 404
        assert "Workspace not found" in response.json()["detail"]


class TestURLUploadCamelCaseSerialization:
    """Tests for camelCase JSON serialization (project standard)."""

    async def test_response_uses_camel_case(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """Response should use camelCase field names."""
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "url",
                "url": "https://unique-test-1.example.com",
            },
        )

        assert response.status_code == 201
        data = response.json()

        # Verify camelCase serialization
        assert "sourceType" in data  # Not source_type
        assert "sourceUrl" in data  # Not source_url
        assert "source_type" not in data
        assert "source_url" not in data

    async def test_request_accepts_camel_case(
        self, shared_integration_client: AsyncClient, test_workspace
    ) -> None:
        """Request should accept camelCase field names (populate_by_name)."""
        # Using camelCase in request body
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents/upload-url",
            json={
                "sourceType": "pdf",  # camelCase
                "url": "https://unique-test-2.example.com/doc.pdf",
            },
        )

        assert response.status_code == 201
