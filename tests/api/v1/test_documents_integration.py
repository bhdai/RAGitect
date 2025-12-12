"""Integration tests for document upload API endpoints

These tests run against a real database to verify end-to-end functionality.
They are marked with @pytest.mark.integration and will be skipped if
DATABASE_URL is not set or doesn't end with '_test'.

To run these tests:
    DATABASE_URL=postgresql+asyncpg://admin:admin@localhost:5432/ragitect_test pytest tests/api/v1/test_documents_integration.py -m integration
"""

import io
import os
import uuid
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text

from main import app
from ragitect.services.database import DatabaseManager, get_session
from ragitect.services.database.models import Base
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository


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

    # Cleanup
    async with db_manager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await db_manager.close()


@pytest_asyncio.fixture
async def clean_database(setup_test_database):
    """Clean all tables before each test"""
    db_manager = setup_test_database
    async with db_manager._engine.begin() as conn:
        # Clean tables - skip if they don't exist
        try:
            await conn.execute(text("TRUNCATE TABLE documents CASCADE"))
        except Exception:
            pass
        try:
            await conn.execute(text("TRUNCATE TABLE document_chunks CASCADE"))
        except Exception:
            pass
        try:
            await conn.execute(text("TRUNCATE TABLE workspaces CASCADE"))
        except Exception:
            pass
        try:
            await conn.execute(text("TRUNCATE TABLE llm_configs CASCADE"))
        except Exception:
            pass
        await conn.commit()
    yield


@pytest_asyncio.fixture
async def test_workspace(clean_database):
    """Create a test workspace for document uploads"""
    import time

    async with get_session() as session:
        repo = WorkspaceRepository(session)
        # Use timestamp to ensure unique name
        workspace = await repo.create(f"Doc Upload Test {int(time.time() * 1000)}")
        await session.commit()
        return workspace


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upload_single_document(test_workspace):
    """Test uploading a single document to workspace"""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Create test file
        file_content = b"This is a test PDF content"
        files = {"files": ("test.pdf", io.BytesIO(file_content), "application/pdf")}

        response = await client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents",
            files=files,
        )

        assert response.status_code == 201
        data = response.json()

        assert "documents" in data
        assert "total" in data
        assert data["total"] == 1
        assert len(data["documents"]) == 1

        doc = data["documents"][0]
        assert doc["fileName"] == "test.pdf"
        assert doc["fileType"] == ".pdf"
        assert doc["status"] == "uploaded"
        assert "id" in doc
        assert "createdAt" in doc


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upload_multiple_documents(test_workspace):
    """Test uploading multiple documents at once"""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Create multiple test files
        files = [
            ("files", ("test1.pdf", io.BytesIO(b"content1"), "application/pdf")),
            ("files", ("test2.txt", io.BytesIO(b"content2"), "text/plain")),
            (
                "files",
                ("test3.md", io.BytesIO(b"# Markdown"), "text/markdown"),
            ),
        ]

        response = await client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents",
            files=files,
        )

        assert response.status_code == 201
        data = response.json()

        assert data["total"] == 3
        assert len(data["documents"]) == 3

        filenames = [doc["fileName"] for doc in data["documents"]]
        assert "test1.pdf" in filenames
        assert "test2.txt" in filenames
        assert "test3.md" in filenames


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upload_unsupported_format(test_workspace):
    """Test uploading an unsupported file format"""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        files = {
            "files": (
                "test.exe",
                io.BytesIO(b"executable content"),
                "application/x-executable",
            )
        }

        response = await client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents",
            files=files,
        )

        assert response.status_code == 400
        data = response.json()
        assert "Unsupported format" in data["detail"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upload_to_nonexistent_workspace(clean_database):
    """Test uploading to a workspace that doesn't exist"""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        fake_workspace_id = uuid.uuid4()
        files = {"files": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}

        response = await client.post(
            f"/api/v1/workspaces/{fake_workspace_id}/documents",
            files=files,
        )

        assert response.status_code == 404
        data = response.json()
        assert "Workspace not found" in data["detail"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_response_uses_camel_case(test_workspace):
    """Test that API response uses camelCase for field names"""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        files = {"files": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}

        response = await client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents",
            files=files,
        )

        assert response.status_code == 201
        data = response.json()

        # Verify camelCase keys
        doc = data["documents"][0]
        assert "fileName" in doc  # not file_name
        assert "fileType" in doc  # not file_type
        assert "createdAt" in doc  # not created_at

        # Verify snake_case keys are NOT present
        assert "file_name" not in doc
        assert "file_type" not in doc
        assert "created_at" not in doc
