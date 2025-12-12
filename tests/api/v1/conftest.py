"""Shared fixtures for API v1 integration tests

Provides database setup and common test data for integration tests.
These fixtures require actual database connection and are separate from
the mocked API test fixtures in tests/api/conftest.py.
"""

import pytest
import pytest_asyncio
from collections.abc import AsyncGenerator
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text

from main import app
from ragitect.services.database.connection import get_session
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.database.repositories.document_repo import DocumentRepository


@pytest_asyncio.fixture
async def clean_database():
    """Clean database before tests"""
    async with get_session() as session:
        conn = await session.connection()
        # Clean test data - ordering matters due to foreign keys
        await conn.execute(text("DELETE FROM document_chunks"))
        await conn.execute(text("DELETE FROM documents"))
        await conn.execute(text("DELETE FROM workspaces"))
        try:
            await conn.execute(text("DELETE FROM llm_configs"))
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


@pytest_asyncio.fixture
async def uploaded_document(test_workspace):
    """Create an uploaded document for testing status endpoint"""
    async with get_session() as session:
        repo = DocumentRepository(session)
        doc = await repo.create_from_upload(
            workspace_id=test_workspace.id,
            file_name="test_status.txt",
            file_type=".txt",
            file_bytes=b"Test content for status checks",
        )
        await session.commit()
        return doc


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for integration tests with REAL database

    This is different from tests/api/conftest.py which mocks the database.
    Integration tests need actual DB connections.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
