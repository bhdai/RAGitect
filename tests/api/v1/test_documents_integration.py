"""Integration tests for document upload API endpoints"""

import io
import os
import pytest
from uuid import uuid4

from httpx import AsyncClient
from sqlalchemy import text

from ragitect.main import app
from ragitect.services.database import get_session
from ragitect.services.database.connection import create_table, drop_table
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository


@pytest.fixture
async def test_workspace():
    """Create a test workspace for document uploads"""
    async with get_session() as session:
        repo = WorkspaceRepository(session)
        workspace = await repo.create("Document Upload Test Workspace")
        await session.commit()
        return workspace


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upload_single_document(clean_db_manager, test_workspace):
    """Test uploading a single document to workspace"""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")

    if not clean_db_manager._engine:
        await clean_db_manager.initialize()

    async with get_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    await create_table()

    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
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

    finally:
        await drop_table()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upload_multiple_documents(clean_db_manager, test_workspace):
    """Test uploading multiple documents at once"""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")

    if not clean_db_manager._engine:
        await clean_db_manager.initialize()

    async with get_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    await create_table()

    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
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

    finally:
        await drop_table()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upload_unsupported_format(clean_db_manager, test_workspace):
    """Test uploading an unsupported file format"""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")

    if not clean_db_manager._engine:
        await clean_db_manager.initialize()

    async with get_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    await create_table()

    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
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

    finally:
        await drop_table()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upload_to_nonexistent_workspace(clean_db_manager):
    """Test uploading to a workspace that doesn't exist"""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")

    if not clean_db_manager._engine:
        await clean_db_manager.initialize()

    async with get_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    await create_table()

    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
            fake_workspace_id = uuid4()
            files = {"files": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}

            response = await client.post(
                f"/api/v1/workspaces/{fake_workspace_id}/documents",
                files=files,
            )

            assert response.status_code == 404
            data = response.json()
            assert "Workspace not found" in data["detail"]

    finally:
        await drop_table()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_response_uses_camel_case(clean_db_manager, test_workspace):
    """Test that API response uses camelCase for field names"""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")

    if not clean_db_manager._engine:
        await clean_db_manager.initialize()

    async with get_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    await create_table()

    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
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

    finally:
        await drop_table()
