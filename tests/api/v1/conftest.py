"""Shared fixtures for API v1 integration tests"""

import pytest_asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from httpx import AsyncClient, ASGITransport


@asynccontextmanager
async def mock_lifespan_for_integration(app):
    """Mock lifespan that doesn't re-initialize the database"""
    yield


@pytest_asyncio.fixture
async def test_workspace(clean_database):
    """Create a test workspace for document uploads"""
    import time

    from ragitect.services.database.connection import get_session
    from ragitect.services.database.repositories.workspace_repo import (
        WorkspaceRepository,
    )

    async with get_session() as session:
        repo = WorkspaceRepository(session)
        # Use timestamp to ensure unique name
        workspace = await repo.create(f"Doc Upload Test {int(time.time() * 1000)}")
        await session.commit()
        return workspace


@pytest_asyncio.fixture
async def uploaded_document(test_workspace):
    """Create an uploaded document for testing status endpoint"""
    from ragitect.services.database.connection import get_session
    from ragitect.services.database.repositories.document_repo import DocumentRepository

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
async def shared_integration_client(
    setup_integration_database, mocker
) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for integration tests with REAL database

    This is different from tests/api/conftest.py which mocks the database.
    Integration tests need actual DB connections.
    """
    # Patch lifespan to avoid re-initializing database
    mocker.patch("main.lifespan", mock_lifespan_for_integration)

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
