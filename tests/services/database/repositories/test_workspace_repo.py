"""Unit tests for WorkspaceRepository."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.database.models import Workspace
from ragitect.services.database.exceptions import NotFoundError, DuplicateError
from sqlalchemy.exc import IntegrityError

pytestmark = [pytest.mark.asyncio]


class TestWorkspaceRepository:
    @pytest.fixture
    def repo(self, mock_session):
        return WorkspaceRepository(mock_session)

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

    async def test_create_duplicate_workspace(self, repo, mock_session):
        mock_session.flush.side_effect = IntegrityError(
            None, None, Exception("Duplicate")
        )

        with pytest.raises(DuplicateError):
            await repo.create("Duplicate Name")

        mock_session.rollback.assert_called_once()

    async def test_get_by_name(self, repo, mock_session):
        expected_workspace = Workspace(id=uuid4(), name="Found Me")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_workspace
        mock_session.execute.return_value = mock_result

        workspace = await repo.get_by_name("Found Me")

        assert workspace == expected_workspace
        mock_session.execute.assert_called_once()

    async def test_get_by_name_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        workspace = await repo.get_by_name("Not Found")

        assert workspace is None

    async def test_get_by_name_or_raise(self, repo, mock_session):
        expected_workspace = Workspace(id=uuid4(), name="Found Me")

        # Mock get_by_name behavior
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_workspace
        mock_session.execute.return_value = mock_result

        workspace = await repo.get_by_name_or_raise("Found Me")

        assert workspace == expected_workspace

    async def test_get_by_name_or_raise_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(NotFoundError):
            await repo.get_by_name_or_raise("Not Found")

    async def test_get_all(self, repo, mock_session):
        workspaces = [Workspace(name="W1"), Workspace(name="W2")]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = workspaces
        mock_session.execute.return_value = mock_result

        result = await repo.get_all(skip=0, limit=10)

        assert len(result) == 2
        assert result == workspaces

    async def test_update_workspace(self, repo, mock_session):
        workspace_id = uuid4()
        workspace = Workspace(id=workspace_id, name="Old Name")
        mock_session.get.return_value = workspace

        updated = await repo.update(workspace_id, name="New Name")

        assert updated.name == "New Name"
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once_with(workspace)

    async def test_update_workspace_not_found(self, repo, mock_session):
        mock_session.get.return_value = None

        with pytest.raises(NotFoundError):
            await repo.update(uuid4(), name="New Name")

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

    async def test_exists_by_name(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = Workspace(name="Exists")
        mock_session.execute.return_value = mock_result

        exists = await repo.exists_by_name("Exists")
        assert exists is True

        mock_result.scalar_one_or_none.return_value = None
        exists = await repo.exists_by_name("Not Exists")
        assert exists is False

    async def test_count_document(self, repo, mock_session):
        workspace_id = uuid4()
        mock_session.get.return_value = Workspace(id=workspace_id)

        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        count = await repo.count_document(workspace_id)

        assert count == 5

    async def test_count_document_workspace_not_found(self, repo, mock_session):
        mock_session.get.return_value = None

        with pytest.raises(NotFoundError):
            await repo.count_document(uuid4())

    async def test_get_with_documents(self, repo, mock_session):
        workspace_id = uuid4()
        workspace = Workspace(id=workspace_id)
        workspace.documents = []  # Mock documents relationship

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = workspace
        mock_session.execute.return_value = mock_result

        result = await repo.get_with_documents(workspace_id)

        assert result == workspace

    async def test_get_with_documents_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(NotFoundError):
            await repo.get_with_documents(uuid4())
