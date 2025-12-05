"""Integration tests for workspace API endpoints

These tests run against a real database to verify end-to-end functionality.
They are marked with @pytest.mark.integration and will be skipped if
DATABASE_URL is not set or doesn't end with '_test'.

To run these tests:
    DATABASE_URL=postgresql+asyncpg://admin:admin@localhost:5432/ragitect_test pytest tests/api/v1/test_workspaces_integration.py -m integration
"""

import os
import uuid
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from ragitect.services.database import DatabaseManager, get_session
from ragitect.services.database.models import Base


@pytest.fixture(scope="module")
def database_url():
    """Get test database URL or skip if not configured"""
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    if not url.split("/")[-1].split("?")[0].endswith("_test"):
        pytest.skip(f"DATABASE_URL must end with '_test' for safety (got: {url})")
    return url


@pytest_asyncio.fixture(scope="module")
async def setup_test_database(database_url):
    """Initialize test database and create tables"""
    db_manager = DatabaseManager.get_instance()

    # Close any existing connection
    if db_manager._engine:
        await db_manager.close()

    # Initialize for testing
    await db_manager.initialize_for_testing(database_url)

    # Create pgvector extension and tables
    async with get_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await session.commit()

    async with db_manager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield db_manager

    # Cleanup: drop tables and close connection
    async with db_manager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await db_manager.close()


@asynccontextmanager
async def mock_lifespan_for_integration(app):
    """Mock lifespan that doesn't re-initialize the database"""
    yield


@pytest_asyncio.fixture
async def integration_client(setup_test_database, mocker):
    """Create async test client with real database but mocked lifespan"""
    # Patch lifespan to avoid re-initializing database
    mocker.patch("main.lifespan", mock_lifespan_for_integration)

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest_asyncio.fixture
async def clean_workspaces(setup_test_database):
    """Clean up workspaces table before each test"""
    async with get_session() as session:
        await session.execute(text("DELETE FROM workspaces"))
        await session.commit()
    yield


class TestWorkspaceIntegration:
    """Integration tests for workspace API endpoints with real database"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_workspace_integration(
        self, integration_client, clean_workspaces
    ):
        """Test creating a workspace end-to-end with real database"""
        response = await integration_client.post(
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

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_workspaces_integration(
        self, integration_client, clean_workspaces
    ):
        """Test listing workspaces end-to-end with real database"""
        # Create some workspaces first
        await integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Workspace 1"},
        )
        await integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Workspace 2"},
        )

        # List all workspaces
        response = await integration_client.get("/api/v1/workspaces")

        assert response.status_code == 200
        data = response.json()

        assert "workspaces" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["workspaces"]) == 2

        names = {ws["name"] for ws in data["workspaces"]}
        assert names == {"Workspace 1", "Workspace 2"}

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_single_workspace_integration(
        self, integration_client, clean_workspaces
    ):
        """Test getting a single workspace by ID with real database"""
        # Create a workspace
        create_response = await integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Get Me", "description": "Test getting single workspace"},
        )
        assert create_response.status_code == 201
        workspace_id = create_response.json()["id"]

        # Get the workspace
        response = await integration_client.get(f"/api/v1/workspaces/{workspace_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == workspace_id
        assert data["name"] == "Get Me"
        assert data["description"] == "Test getting single workspace"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_nonexistent_workspace_returns_404(
        self, integration_client, clean_workspaces
    ):
        """Test getting a non-existent workspace returns 404"""
        fake_id = str(uuid.uuid4())
        response = await integration_client.get(f"/api/v1/workspaces/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_duplicate_workspace_name_returns_409(
        self, integration_client, clean_workspaces
    ):
        """Test creating duplicate workspace name returns 409"""
        # Create first workspace
        await integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Unique Name"},
        )

        # Try to create duplicate
        response = await integration_client.post(
            "/api/v1/workspaces",
            json={"name": "Unique Name"},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_workspace_name_returns_422(
        self, integration_client, clean_workspaces
    ):
        """Test creating workspace with empty name returns validation error"""
        response = await integration_client.post(
            "/api/v1/workspaces",
            json={"name": ""},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workspace_pagination(self, integration_client, clean_workspaces):
        """Test workspace listing pagination"""
        # Create 5 workspaces
        for i in range(5):
            await integration_client.post(
                "/api/v1/workspaces",
                json={"name": f"Paginated {i}"},
            )

        # Get first page
        response = await integration_client.get("/api/v1/workspaces?skip=0&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["workspaces"]) == 2

        # Get second page
        response = await integration_client.get("/api/v1/workspaces?skip=2&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["workspaces"]) == 2
