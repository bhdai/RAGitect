import pytest
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.database.models import Workspace
from ragitect.services.database.exceptions import NotFoundError, DuplicateError
from sqlalchemy.exc import IntegrityError


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    session.delete = AsyncMock()
    session.get = AsyncMock()
    return session


class TestWorkspaceRepository:
    @pytest.fixture
    def repo(self, mock_session):
        return WorkspaceRepository(mock_session)

    @pytest.mark.asyncio
    async def test_create_workspace(self, repo, mock_session):
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.refresh = AsyncMock()

        name = "Test Workspace"
        description = "Test Description"

        workspace = await repo.create(name, description)

        assert workspace.name == name
        assert workspace.description == description
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_duplicate_workspace(self, repo, mock_session):
        mock_session.flush.side_effect = IntegrityError(
            None, None, Exception("Duplicate")
        )

        with pytest.raises(DuplicateError):
            await repo.create("Duplicate Name")

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_name(self, repo, mock_session):
        expected_workspace = Workspace(id=uuid4(), name="Found Me")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_workspace
        mock_session.execute.return_value = mock_result

        workspace = await repo.get_by_name("Found Me")

        assert workspace == expected_workspace
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_name_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        workspace = await repo.get_by_name("Not Found")

        assert workspace is None

    @pytest.mark.asyncio
    async def test_get_by_name_or_raise(self, repo, mock_session):
        expected_workspace = Workspace(id=uuid4(), name="Found Me")

        # Mock get_by_name behavior
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_workspace
        mock_session.execute.return_value = mock_result

        workspace = await repo.get_by_name_or_raise("Found Me")

        assert workspace == expected_workspace

    @pytest.mark.asyncio
    async def test_get_by_name_or_raise_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(NotFoundError):
            await repo.get_by_name_or_raise("Not Found")

    @pytest.mark.asyncio
    async def test_get_all(self, repo, mock_session):
        workspaces = [Workspace(name="W1"), Workspace(name="W2")]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = workspaces
        mock_session.execute.return_value = mock_result

        result = await repo.get_all(skip=0, limit=10)

        assert len(result) == 2
        assert result == workspaces

    @pytest.mark.asyncio
    async def test_update_workspace(self, repo, mock_session):
        workspace_id = uuid4()
        workspace = Workspace(id=workspace_id, name="Old Name")
        mock_session.get.return_value = workspace

        updated = await repo.update(workspace_id, name="New Name")

        assert updated.name == "New Name"
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once_with(workspace)

    @pytest.mark.asyncio
    async def test_update_workspace_not_found(self, repo, mock_session):
        mock_session.get.return_value = None

        with pytest.raises(NotFoundError):
            await repo.update(uuid4(), name="New Name")

    @pytest.mark.asyncio
    async def test_update_workspace_duplicate_name(self, repo, mock_session):
        workspace_id = uuid4()
        workspace = Workspace(id=workspace_id, name="Old Name")
        mock_session.get.return_value = workspace

        mock_session.flush.side_effect = IntegrityError(
            None, None, Exception("Duplicate")
        )

        with pytest.raises(DuplicateError):
            await repo.update(workspace_id, name="Duplicate Name")

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_exists_by_name(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = Workspace(name="Exists")
        mock_session.execute.return_value = mock_result

        exists = await repo.exists_by_name("Exists")
        assert exists is True

        mock_result.scalar_one_or_none.return_value = None
        exists = await repo.exists_by_name("Not Exists")
        assert exists is False

    @pytest.mark.asyncio
    async def test_count_document(self, repo, mock_session):
        workspace_id = uuid4()
        mock_session.get.return_value = Workspace(id=workspace_id)

        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        count = await repo.count_document(workspace_id)

        assert count == 5

    @pytest.mark.asyncio
    async def test_count_document_workspace_not_found(self, repo, mock_session):
        mock_session.get.return_value = None

        with pytest.raises(NotFoundError):
            await repo.count_document(uuid4())

    @pytest.mark.asyncio
    async def test_get_with_documents(self, repo, mock_session):
        workspace_id = uuid4()
        workspace = Workspace(id=workspace_id)
        workspace.documents = []  # Mock documents relationship

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = workspace
        mock_session.execute.return_value = mock_result

        result = await repo.get_with_documents(workspace_id)

        assert result == workspace

    @pytest.mark.asyncio
    async def test_get_with_documents_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(NotFoundError):
            await repo.get_with_documents(uuid4())


class TestWorkspaceRepositoryIntegration:
    """Integration tests for WorkspaceRepository"""

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def db_context(self, clean_db_manager):
        """Setup database for integration tests"""
        import os
        from sqlalchemy import text
        from ragitect.services.database import get_session
        from ragitect.services.database.connection import create_table, drop_table

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        await create_table()
        try:
            yield
        finally:
            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_and_get_workspace(self, clean_db_manager):
        from ragitect.services.database import get_session

        async with self.db_context(clean_db_manager):
            async with get_session() as session:
                repo = WorkspaceRepository(session)
                workspace = await repo.create("Integration Test", "Description")

                assert workspace.id is not None
                assert workspace.name == "Integration Test"

                # Test get_by_name
                fetched = await repo.get_by_name("Integration Test")
                assert fetched is not None
                assert fetched.id == workspace.id

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_duplicate_error(self, clean_db_manager):
        from ragitect.services.database import get_session

        async with self.db_context(clean_db_manager):
            async with get_session() as session:
                repo = WorkspaceRepository(session)
                await repo.create("Unique Name")

                with pytest.raises(DuplicateError):
                    await repo.create("Unique Name")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_workspace(self, clean_db_manager):
        from ragitect.services.database import get_session

        async with self.db_context(clean_db_manager):
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

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_count_document_integration(self, clean_db_manager):
        from ragitect.services.database import get_session
        from ragitect.services.database.repositories.document_repo import (
            DocumentRepository,
        )

        async with self.db_context(clean_db_manager):
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

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_with_documents_integration(self, clean_db_manager):
        from ragitect.services.database import get_session
        from ragitect.services.database.repositories.document_repo import (
            DocumentRepository,
        )

        async with self.db_context(clean_db_manager):
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
