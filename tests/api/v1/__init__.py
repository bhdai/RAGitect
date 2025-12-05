"""Tests for workspace API endpoints

Tests verify:
- POST /api/v1/workspaces creates workspace (201)
- GET /api/v1/workspaces lists all workspaces (200)
- GET /api/v1/workspaces/{id} gets single workspace (200)
- Proper error handling (400, 404, 409)
- Response bodies use camelCase keys
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def async_client():
    """Create async test client for FastAPI app"""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


class TestCreateWorkspace:
    """Tests for POST /api/v1/workspaces"""

    @pytest.mark.asyncio
    async def test_create_workspace_success(self, async_client, mocker):
        """Test successful workspace creation returns 201 and camelCase response"""
        # Mock the session and repository
        from ragitect.services.database.models import Workspace
        from datetime import datetime, timezone

        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        mock_workspace = Workspace(
            id=workspace_id,
            name="Test Workspace",
            description="Test description",
        )
        mock_workspace.created_at = now
        mock_workspace.updated_at = now

        # Mock get_session to return a mock session
        mock_session = mocker.AsyncMock()
        mock_repo = mocker.AsyncMock()
        mock_repo.create.return_value = mock_workspace

        mocker.patch(
            "ragitect.api.v1.workspaces.get_session",
            return_value=mocker.AsyncMock(
                __aenter__=mocker.AsyncMock(return_value=mock_session),
                __aexit__=mocker.AsyncMock(return_value=None),
            ),
        )
        mocker.patch(
            "ragitect.api.v1.workspaces.WorkspaceRepository",
            return_value=mock_repo,
        )

        response = await async_client.post(
            "/api/v1/workspaces",
            json={"name": "Test Workspace", "description": "Test description"},
        )

        assert response.status_code == 201
        data = response.json()

        # Verify camelCase keys
        assert "id" in data
        assert "name" in data
        assert "createdAt" in data
        assert "updatedAt" in data
        assert "description" in data

        # Verify snake_case keys are NOT present
        assert "created_at" not in data
        assert "updated_at" not in data

        assert data["name"] == "Test Workspace"
        assert data["description"] == "Test description"

    @pytest.mark.asyncio
    async def test_create_workspace_name_only(self, async_client, mocker):
        """Test creating workspace with only name (no description)"""
        from ragitect.services.database.models import Workspace
        from datetime import datetime, timezone

        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        mock_workspace = Workspace(id=workspace_id, name="Name Only")
        mock_workspace.created_at = now
        mock_workspace.updated_at = now

        mock_session = mocker.AsyncMock()
        mock_repo = mocker.AsyncMock()
        mock_repo.create.return_value = mock_workspace

        mocker.patch(
            "ragitect.api.v1.workspaces.get_session",
            return_value=mocker.AsyncMock(
                __aenter__=mocker.AsyncMock(return_value=mock_session),
                __aexit__=mocker.AsyncMock(return_value=None),
            ),
        )
        mocker.patch(
            "ragitect.api.v1.workspaces.WorkspaceRepository",
            return_value=mock_repo,
        )

        response = await async_client.post(
            "/api/v1/workspaces",
            json={"name": "Name Only"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Name Only"
        assert data["description"] is None

    @pytest.mark.asyncio
    async def test_create_workspace_empty_name_returns_400(self, async_client):
        """Test that empty name returns 400 validation error"""
        response = await async_client.post(
            "/api/v1/workspaces",
            json={"name": ""},
        )

        assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.asyncio
    async def test_create_workspace_duplicate_returns_409(self, async_client, mocker):
        """Test that duplicate workspace name returns 409"""
        from ragitect.services.database.exceptions import DuplicateError

        mock_session = mocker.AsyncMock()
        mock_repo = mocker.AsyncMock()
        mock_repo.create.side_effect = DuplicateError("Workspace", "name", "Duplicate")

        mocker.patch(
            "ragitect.api.v1.workspaces.get_session",
            return_value=mocker.AsyncMock(
                __aenter__=mocker.AsyncMock(return_value=mock_session),
                __aexit__=mocker.AsyncMock(return_value=None),
            ),
        )
        mocker.patch(
            "ragitect.api.v1.workspaces.WorkspaceRepository",
            return_value=mock_repo,
        )

        response = await async_client.post(
            "/api/v1/workspaces",
            json={"name": "Duplicate"},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()


class TestListWorkspaces:
    """Tests for GET /api/v1/workspaces"""

    @pytest.mark.asyncio
    async def test_list_workspaces_success(self, async_client, mocker):
        """Test listing workspaces returns 200 with camelCase response"""
        from ragitect.services.database.models import Workspace
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        mock_workspaces = [
            Workspace(id=uuid.uuid4(), name="Workspace 1"),
            Workspace(id=uuid.uuid4(), name="Workspace 2"),
        ]
        for ws in mock_workspaces:
            ws.created_at = now
            ws.updated_at = now

        mock_session = mocker.AsyncMock()
        mock_repo = mocker.AsyncMock()
        mock_repo.get_all.return_value = mock_workspaces
        mock_repo.count.return_value = 2

        mocker.patch(
            "ragitect.api.v1.workspaces.get_session",
            return_value=mocker.AsyncMock(
                __aenter__=mocker.AsyncMock(return_value=mock_session),
                __aexit__=mocker.AsyncMock(return_value=None),
            ),
        )
        mocker.patch(
            "ragitect.api.v1.workspaces.WorkspaceRepository",
            return_value=mock_repo,
        )

        response = await async_client.get("/api/v1/workspaces")

        assert response.status_code == 200
        data = response.json()

        assert "workspaces" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["workspaces"]) == 2

        # Verify nested workspaces use camelCase
        ws = data["workspaces"][0]
        assert "createdAt" in ws
        assert "updatedAt" in ws
        assert "created_at" not in ws
        assert "updated_at" not in ws

    @pytest.mark.asyncio
    async def test_list_workspaces_empty(self, async_client, mocker):
        """Test listing workspaces when none exist"""
        mock_session = mocker.AsyncMock()
        mock_repo = mocker.AsyncMock()
        mock_repo.get_all.return_value = []
        mock_repo.count.return_value = 0

        mocker.patch(
            "ragitect.api.v1.workspaces.get_session",
            return_value=mocker.AsyncMock(
                __aenter__=mocker.AsyncMock(return_value=mock_session),
                __aexit__=mocker.AsyncMock(return_value=None),
            ),
        )
        mocker.patch(
            "ragitect.api.v1.workspaces.WorkspaceRepository",
            return_value=mock_repo,
        )

        response = await async_client.get("/api/v1/workspaces")

        assert response.status_code == 200
        data = response.json()
        assert data["workspaces"] == []
        assert data["total"] == 0


class TestGetWorkspace:
    """Tests for GET /api/v1/workspaces/{id}"""

    @pytest.mark.asyncio
    async def test_get_workspace_success(self, async_client, mocker):
        """Test getting workspace by ID returns 200 with camelCase response"""
        from ragitect.services.database.models import Workspace
        from datetime import datetime, timezone

        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        mock_workspace = Workspace(
            id=workspace_id, name="Get Me", description="Found me"
        )
        mock_workspace.created_at = now
        mock_workspace.updated_at = now

        mock_session = mocker.AsyncMock()
        mock_repo = mocker.AsyncMock()
        mock_repo.get_by_id_or_raise.return_value = mock_workspace

        mocker.patch(
            "ragitect.api.v1.workspaces.get_session",
            return_value=mocker.AsyncMock(
                __aenter__=mocker.AsyncMock(return_value=mock_session),
                __aexit__=mocker.AsyncMock(return_value=None),
            ),
        )
        mocker.patch(
            "ragitect.api.v1.workspaces.WorkspaceRepository",
            return_value=mock_repo,
        )

        response = await async_client.get(f"/api/v1/workspaces/{workspace_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == str(workspace_id)
        assert data["name"] == "Get Me"
        assert data["description"] == "Found me"
        assert "createdAt" in data
        assert "updatedAt" in data

    @pytest.mark.asyncio
    async def test_get_workspace_not_found(self, async_client, mocker):
        """Test getting non-existent workspace returns 404"""
        from ragitect.services.database.exceptions import NotFoundError

        workspace_id = uuid.uuid4()

        mock_session = mocker.AsyncMock()
        mock_repo = mocker.AsyncMock()
        mock_repo.get_by_id_or_raise.side_effect = NotFoundError(
            "Workspace", workspace_id
        )

        mocker.patch(
            "ragitect.api.v1.workspaces.get_session",
            return_value=mocker.AsyncMock(
                __aenter__=mocker.AsyncMock(return_value=mock_session),
                __aexit__=mocker.AsyncMock(return_value=None),
            ),
        )
        mocker.patch(
            "ragitect.api.v1.workspaces.WorkspaceRepository",
            return_value=mock_repo,
        )

        response = await async_client.get(f"/api/v1/workspaces/{workspace_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_workspace_invalid_uuid(self, async_client):
        """Test getting workspace with invalid UUID returns 422"""
        response = await async_client.get("/api/v1/workspaces/invalid-uuid")

        assert response.status_code == 422
