"""Integration tests for WorkspaceRepository.

Requirements:
    - Running PostgreSQL database (ragitect_test)
    - DATABASE_URL environment variable set
"""

import pytest
from ragitect.services.database import get_session
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.exceptions import DuplicateError

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestWorkspaceRepositoryIntegration:
    """Integration tests for WorkspaceRepository"""

    async def test_create_and_get_workspace(self, clean_database):
        async with get_session() as session:
            repo = WorkspaceRepository(session)
            workspace = await repo.create("Integration Test", "Description")

            assert workspace.id is not None
            assert workspace.name == "Integration Test"

            # Test get_by_name
            fetched = await repo.get_by_name("Integration Test")
            assert fetched is not None
            assert fetched.id == workspace.id

    async def test_duplicate_error(self, clean_database):
        async with get_session() as session:
            repo = WorkspaceRepository(session)
            await repo.create("Unique Name")

            with pytest.raises(DuplicateError):
                await repo.create("Unique Name")

    async def test_update_workspace(self, clean_database):
        async with get_session() as session:
            repo = WorkspaceRepository(session)
            workspace = await repo.create("To Update")

            updated = await repo.update(
                workspace.id, name="Updated Name", description="New Desc"
            )
            assert updated.name == "Updated Name"
            assert updated.description == "New Desc"

            fetched = await repo.get_by_name("Updated Name")
            assert fetched is not None

    async def test_count_document_integration(self, clean_database):
        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)

            workspace = await ws_repo.create("Doc Counter")

            # Create documents
            await doc_repo.create(
                workspace.id, "doc1.pdf", "content", embedding=[0.1] * 768
            )
            await doc_repo.create(
                workspace.id, "doc2.pdf", "content2", embedding=[0.1] * 768
            )

            count = await ws_repo.count_document(workspace.id)
            assert count == 2

    async def test_get_with_documents_integration(self, clean_database):
        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)

            workspace = await ws_repo.create("Eager Load")
            await doc_repo.create(
                workspace.id, "doc1.pdf", "content", embedding=[0.1] * 768
            )

        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            loaded_ws = await ws_repo.get_with_documents(workspace.id)
            assert len(loaded_ws.documents) == 1
            assert loaded_ws.documents[0].file_name == "doc1.pdf"
