"""Integration tests for document delete endpoint

Tests for DELETE /api/v1/documents/{document_id} endpoint.

To run these tests:
    uv run --env-file .env.test pytest tests/api/v1/test_document_delete_integration.py -m integration
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from ragitect.services.database.connection import get_session
from ragitect.services.database.repositories.document_repo import DocumentRepository

# Apply asyncio and integration markers to all tests in this module
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestDeleteDocument:
    """Integration tests for DELETE /documents/{document_id} endpoint"""

    async def test_deletes_document_successfully(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test deleting a document removes it from the database"""
        # Step 1: Create workspace via API
        workspace_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Delete Test Workspace"},
        )
        assert workspace_response.status_code == 201
        workspace_id = workspace_response.json()["id"]

        # Step 2: Create document via repository
        async with get_session() as session:
            doc_repo = DocumentRepository(session)
            document = await doc_repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="to_delete.pdf",
                processed_content="Content to be deleted",
                file_type="pdf",
            )
            await session.commit()
            document_id = str(document.id)

        # Step 2: Delete document via API
        response = await shared_integration_client.delete(
            f"/api/v1/documents/{document_id}"
        )

        assert response.status_code == 204
        assert response.content == b""

        # Step 3: Verify document is deleted
        get_response = await shared_integration_client.get(
            f"/api/v1/documents/{document_id}"
        )
        assert get_response.status_code == 404

    async def test_cascade_deletes_chunks_and_embeddings(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test deleting document also deletes its chunks (cascade)"""
        # Step 1: Create workspace via API
        workspace_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Cascade Delete Test"},
        )
        assert workspace_response.status_code == 201
        workspace_id = workspace_response.json()["id"]

        # Step 2: Create document and chunks via repository
        async with get_session() as session:
            doc_repo = DocumentRepository(session)
            document = await doc_repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="with_chunks.pdf",
                processed_content="Content with chunks",
                file_type="pdf",
                embedding=[0.1] * 768,
            )

            # Add chunks with embeddings
            await doc_repo.add_chunks(
                document_id=document.id,
                chunks=[
                    ("Chunk 1 content", [0.2] * 768, None),
                    ("Chunk 2 content", [0.3] * 768, None),
                    ("Chunk 3 content", [0.4] * 768, None),
                ],
            )
            await session.commit()
            document_id = str(document.id)

        # Step 2: Verify chunks exist before deletion
        async with get_session() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM document_chunks WHERE document_id = :doc_id"
                ),
                {"doc_id": document_id},
            )
            chunk_count = result.scalar()
            assert chunk_count == 3

        # Step 3: Delete document via API
        response = await shared_integration_client.delete(
            f"/api/v1/documents/{document_id}"
        )
        assert response.status_code == 204

        # Step 4: Verify chunks are cascade deleted
        async with get_session() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM document_chunks WHERE document_id = :doc_id"
                ),
                {"doc_id": document_id},
            )
            chunk_count = result.scalar()
            assert chunk_count == 0

    async def test_document_not_found_returns_404(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test deleting non-existent document returns 404"""
        fake_document_id = str(uuid.uuid4())

        response = await shared_integration_client.delete(
            f"/api/v1/documents/{fake_document_id}"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_delete_does_not_affect_other_documents(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test deleting one document doesn't affect other documents"""
        # Create workspace via API
        workspace_response = await shared_integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Multi Doc Workspace"},
        )
        assert workspace_response.status_code == 201
        workspace_id = workspace_response.json()["id"]

        # Create multiple documents via repository
        async with get_session() as session:
            doc_repo = DocumentRepository(session)
            doc1 = await doc_repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="keep_me.pdf",
                processed_content="Keep this content",
                file_type="pdf",
            )
            doc2 = await doc_repo.create(
                workspace_id=uuid.UUID(workspace_id),
                file_name="delete_me.pdf",
                processed_content="Delete this content",
                file_type="pdf",
            )
            await session.commit()
            keep_doc_id = str(doc1.id)
            delete_doc_id = str(doc2.id)

        # Delete one document
        response = await shared_integration_client.delete(
            f"/api/v1/documents/{delete_doc_id}"
        )
        assert response.status_code == 204

        # Verify other document still exists
        get_response = await shared_integration_client.get(
            f"/api/v1/documents/{keep_doc_id}"
        )
        assert get_response.status_code == 200
        assert get_response.json()["fileName"] == "keep_me.pdf"
