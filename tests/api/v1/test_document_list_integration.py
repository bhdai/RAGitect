"""Integration tests for document listing endpoint

Tests for GET /api/v1/workspaces/{workspace_id}/documents endpoint.

To run these tests:
    uv run --env-file .env.test pytest tests/api/v1/test_document_list_integration.py -m integration
"""

import uuid

import pytest
from httpx import AsyncClient

from ragitect.services.database.connection import get_session
from ragitect.services.database.repositories.document_repo import DocumentRepository

# Apply asyncio and integration markers to all tests in this module
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestListDocuments:
    """Integration tests for GET /workspaces/{workspace_id}/documents endpoint"""

    async def test_returns_documents_for_workspace(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test listing documents in a workspace with documents"""
        # Step 1: Create a workspace
        workspace_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "List Documents Test"},
        )
        assert workspace_response.status_code == 201
        workspace_id = workspace_response.json()["id"]

        # Step 2: Add documents via repository
        async with get_session() as session:
            repo = DocumentRepository(session)
            await repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="document1.pdf",
                processed_content="Content of document 1",
                file_type="pdf",
            )
            await repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="document2.txt",
                processed_content="Content of document 2",
                file_type="txt",
            )
            await session.commit()

        # Step 3: List documents via API
        response = await shared_integration_client.get(
            f"/api/v1/workspaces/{workspace_id}/documents"
        )

        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["documents"]) == 2

        file_names = {doc["fileName"] for doc in data["documents"]}
        assert file_names == {"document1.pdf", "document2.txt"}

    async def test_empty_workspace_returns_empty_list(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test empty workspace returns empty list"""
        # Create workspace without documents
        workspace_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Empty Workspace"},
        )
        assert workspace_response.status_code == 201
        workspace_id = workspace_response.json()["id"]

        # List documents
        response = await shared_integration_client.get(
            f"/api/v1/workspaces/{workspace_id}/documents"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0

    async def test_workspace_not_found_returns_404(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test listing documents for non-existent workspace returns 404"""
        fake_workspace_id = str(uuid.uuid4())

        response = await shared_integration_client.get(
            f"/api/v1/workspaces/{fake_workspace_id}/documents"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_pagination_with_skip_and_limit(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test document listing with pagination parameters"""
        # Create workspace
        workspace_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Pagination Test Workspace"},
        )
        workspace_id = workspace_response.json()["id"]

        # Add 5 documents
        async with get_session() as session:
            repo = DocumentRepository(session)
            for i in range(5):
                await repo.create(
                    workspace_id=uuid.UUID(workspace_id),
                    file_name=f"doc_{i}.pdf",
                    processed_content=f"Content {i}",
                    file_type="pdf",
                )
            await session.commit()

        # Get first page (limit=2)
        response = await shared_integration_client.get(
            f"/api/v1/workspaces/{workspace_id}/documents?skip=0&limit=2"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 2
        assert data["total"] == 2  # Total reflects returned count

        # Get second page
        response = await shared_integration_client.get(
            f"/api/v1/workspaces/{workspace_id}/documents?skip=2&limit=2"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 2

    async def test_documents_ordered_by_created_at_desc(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test documents are returned in descending order by creation date"""
        import asyncio

        # Create workspace
        workspace_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Order Test Workspace"},
        )
        workspace_id = workspace_response.json()["id"]

        # Add documents sequentially with delay to ensure different timestamps
        async with get_session() as session:
            repo = DocumentRepository(session)
            await repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="first.pdf",
                processed_content="First doc",
                file_type="pdf",
            )
            await session.commit()

        # Small delay to ensure different timestamp
        await asyncio.sleep(0.1)

        async with get_session() as session:
            repo = DocumentRepository(session)
            await repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="second.pdf",
                processed_content="Second doc",
                file_type="pdf",
            )
            await session.commit()

        response = await shared_integration_client.get(
            f"/api/v1/workspaces/{workspace_id}/documents"
        )
        data = response.json()

        # Most recently created should be first (desc order)
        assert data["documents"][0]["fileName"] == "second.pdf"
        assert data["documents"][1]["fileName"] == "first.pdf"
