"""Integration tests for workspace API endpoints

These tests run against a real database to verify end-to-end functionality.
They use shared fixtures from conftest.py which handle database setup,
cleanup, and test client creation.

To run these tests:
    uv run --env-file .env.test pytest tests/api/v1/test_workspaces_integration.py -m integration
"""

import uuid

import pytest
from httpx import AsyncClient

from ragitect.services.database.connection import get_session

# Apply asyncio and integration markers to all tests in this module
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestWorkspaceIntegration:
    """Integration tests for workspace API endpoints with real database"""

    async def test_create_workspace_integration(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test creating a workspace end-to-end with real database"""
        response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={
                "name": "Integration Test Workspace",
                "description": "Test description",
            },
        )

        assert response.status_code == 201
        data = response.json()

        # Verify response structure (camelCase)
        assert "id" in data
        assert "name" in data
        assert "createdAt" in data
        assert "updatedAt" in data
        assert data["name"] == "Integration Test Workspace"
        assert data["description"] == "Test description"

        # Verify it's a valid UUID
        uuid.UUID(data["id"])

    async def test_list_workspaces_integration(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test listing workspaces end-to-end with real database"""
        # Create some workspaces first
        await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Workspace 1"},
        )
        await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Workspace 2"},
        )

        # List all workspaces
        response = await shared_integration_client.get("/api/v1/workspaces")

        assert response.status_code == 200
        data = response.json()

        assert "workspaces" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["workspaces"]) == 2

        names = {ws["name"] for ws in data["workspaces"]}
        assert names == {"Workspace 1", "Workspace 2"}

    async def test_get_single_workspace_integration(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test getting a single workspace by ID with real database"""
        # Create a workspace
        create_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Get Me", "description": "Test getting single workspace"},
        )
        assert create_response.status_code == 201
        workspace_id = create_response.json()["id"]

        # Get the workspace
        response = await shared_integration_client.get(
            f"/api/v1/workspaces/{workspace_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == workspace_id
        assert data["name"] == "Get Me"
        assert data["description"] == "Test getting single workspace"

    async def test_get_nonexistent_workspace_returns_404(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test getting a non-existent workspace returns 404"""
        fake_id = str(uuid.uuid4())
        response = await shared_integration_client.get(f"/api/v1/workspaces/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_duplicate_workspace_name_returns_409(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test creating duplicate workspace name returns 409"""
        # Create first workspace
        await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Unique Name"},
        )

        # Try to create duplicate
        response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Unique Name"},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()

    async def test_empty_workspace_name_returns_422(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test creating workspace with empty name returns validation error"""
        response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": ""},
        )

        assert response.status_code == 422

    async def test_workspace_pagination(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test workspace listing pagination"""
        # Create 5 workspaces
        for i in range(5):
            await shared_integration_client.post(
                "/api/v1/workspaces",
                json={"name": f"Paginated {i}"},
            )

        # Get first page
        response = await shared_integration_client.get(
            "/api/v1/workspaces?skip=0&limit=2"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["workspaces"]) == 2

        # Get second page
        response = await shared_integration_client.get(
            "/api/v1/workspaces?skip=2&limit=2"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["workspaces"]) == 2

    async def test_delete_workspace_cascade_integration(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test workspace deletion cascades to documents and chunks"""
        from sqlalchemy import text

        from ragitect.services.database.repositories.document_repo import (
            DocumentRepository,
        )

        # Step 1: Create a workspace
        create_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Delete Me", "description": "Will be deleted"},
        )
        assert create_response.status_code == 201
        workspace_id = create_response.json()["id"]

        # Step 2: Add documents and chunks to workspace via repository
        async with get_session() as session:
            doc_repo = DocumentRepository(session)

            # Create document with chunks
            document = await doc_repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="test.pdf",
                processed_content="Test content for cascade deletion",
                file_type="pdf",
                embedding=[0.1] * 768,
            )

            # Add chunks to document
            await doc_repo.add_chunks(
                document_id=document.id,
                chunks=[
                    ("Chunk 1 content", [0.2] * 768, None),
                    ("Chunk 2 content", [0.3] * 768, None),
                ],
            )
            await session.commit()

            document_id = document.id

        # Step 3: Verify documents and chunks exist
        async with get_session() as session:
            # Check document exists
            result = await session.execute(
                text("SELECT COUNT(*) FROM documents WHERE workspace_id = :ws_id"),
                {"ws_id": workspace_id},
            )
            doc_count = result.scalar()
            assert doc_count == 1

            # Check chunks exist
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM document_chunks WHERE document_id = :doc_id"
                ),
                {"doc_id": str(document_id)},
            )
            chunk_count = result.scalar()
            assert chunk_count == 2

        # Step 4: Delete workspace via API
        delete_response = await shared_integration_client.delete(
            f"/api/v1/workspaces/{workspace_id}"
        )
        assert delete_response.status_code == 204
        assert delete_response.content == b""

        # Step 5: Verify workspace is deleted
        get_response = await shared_integration_client.get(
            f"/api/v1/workspaces/{workspace_id}"
        )
        assert get_response.status_code == 404

        # Step 6: Verify documents were cascade deleted
        async with get_session() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM documents WHERE workspace_id = :ws_id"),
                {"ws_id": workspace_id},
            )
            doc_count = result.scalar()
            assert doc_count == 0

            # Step 7: Verify chunks were cascade deleted
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM document_chunks WHERE document_id = :doc_id"
                ),
                {"doc_id": str(document_id)},
            )
            chunk_count = result.scalar()
            assert chunk_count == 0

    async def test_delete_nonexistent_workspace_returns_404(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test deleting a non-existent workspace returns 404"""
        fake_id = str(uuid.uuid4())
        response = await shared_integration_client.delete(
            f"/api/v1/workspaces/{fake_id}"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
