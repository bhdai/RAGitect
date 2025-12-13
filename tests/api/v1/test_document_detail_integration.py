"""Integration tests for document detail endpoint

Tests for GET /api/v1/documents/{document_id} endpoint.

To run these tests:
    uv run --env-file .env.test pytest tests/api/v1/test_document_detail_integration.py -m integration
"""

import uuid

import pytest
from httpx import AsyncClient

from ragitect.services.database.connection import get_session
from ragitect.services.database.repositories.document_repo import DocumentRepository

# Apply asyncio and integration markers to all tests in this module
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestGetDocumentDetail:
    """Integration tests for GET /documents/{document_id} endpoint"""

    async def test_returns_document_with_processed_content(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test getting document detail returns document with processed content"""
        # Step 1: Create workspace via API
        workspace_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Detail Test Workspace"},
        )
        assert workspace_response.status_code == 201
        workspace_id = workspace_response.json()["id"]

        # Step 2: Create document via repository
        async with get_session() as session:
            doc_repo = DocumentRepository(session)
            document = await doc_repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="detailed_doc.pdf",
                processed_content="# Document Title\n\nThis is the extracted content.",
                file_type="pdf",
                summary="A test document with detailed content",
                metadata={"status": "ready"},
            )
            await session.commit()
            document_id = str(document.id)

        # Step 2: Get document detail via API
        response = await shared_integration_client.get(
            f"/api/v1/documents/{document_id}"
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all expected fields are present (camelCase)
        assert data["id"] == document_id
        assert data["fileName"] == "detailed_doc.pdf"
        assert data["fileType"] == "pdf"
        assert data["status"] == "ready"
        assert (
            data["processedContent"]
            == "# Document Title\n\nThis is the extracted content."
        )
        assert data["summary"] == "A test document with detailed content"
        assert "createdAt" in data

    async def test_document_not_found_returns_404(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test getting non-existent document returns 404"""
        fake_document_id = str(uuid.uuid4())

        response = await shared_integration_client.get(
            f"/api/v1/documents/{fake_document_id}"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_returns_null_for_unprocessed_document(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test getting unprocessed document returns null for processed_content"""
        import hashlib
        import time

        # Create workspace via API with unique name
        unique_suffix = str(time.time())
        workspace_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": f"Unprocessed Test Workspace {unique_suffix}"},
        )
        assert workspace_response.status_code == 201
        workspace_id = workspace_response.json()["id"]

        # Create document without processed content (just uploaded)
        # Use unique content_hash to avoid duplicate detection
        unique_content_hash = hashlib.sha256(unique_suffix.encode()).hexdigest()
        async with get_session() as session:
            doc_repo = DocumentRepository(session)
            document = await doc_repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name=f"unprocessed_{unique_suffix}.pdf",
                processed_content=None,  # Not processed yet
                file_type="pdf",
                content_hash=unique_content_hash,
                metadata={"status": "uploaded"},
            )
            await session.commit()
            document_id = str(document.id)

        response = await shared_integration_client.get(
            f"/api/v1/documents/{document_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["processedContent"] is None
        assert data["status"] == "uploaded"
