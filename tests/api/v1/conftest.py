"""Shared fixtures for API v1 integration tests

Provides database setup and common test data for integration tests.
These fixtures require actual database connection and are separate from
the mocked API test fixtures in tests/api/conftest.py.

CRITICAL: These fixtures are ONLY used by integration tests.
If DATABASE_URL is not set or doesn't point to a test database,
tests using these fixtures will be skipped.
"""

import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text


# Get database URL and validate it's a test database
_DATABASE_URL = os.getenv("DATABASE_URL")
_SKIP_REASON = None
if not _DATABASE_URL:
    _SKIP_REASON = "DATABASE_URL not set - skipping integration tests"
elif not _DATABASE_URL.split("/")[-1].split("?")[0].endswith("_test"):
    _SKIP_REASON = (
        f"DATABASE_URL must end with '_test' for safety (got: {_DATABASE_URL})"
    )


@pytest.fixture(scope="module")
def database_url():
    """Get test database URL or skip if not configured"""
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)
    return _DATABASE_URL


@asynccontextmanager
async def mock_lifespan_for_integration(app):
    """Mock lifespan that doesn't re-initialize the database"""
    yield


@pytest_asyncio.fixture(scope="module")
async def setup_integration_database(database_url):
    """Initialize test database and create tables for integration tests"""
    from ragitect.services.database.connection import DatabaseManager, get_session
    from ragitect.services.database.models import Base

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


@pytest_asyncio.fixture
async def clean_database(setup_integration_database):
    """Clean database before tests"""
    db_manager = setup_integration_database

    async with db_manager._engine.begin() as conn:
        # Clean test data - ordering matters due to foreign keys
        await conn.execute(text("DELETE FROM document_chunks"))
        await conn.execute(text("DELETE FROM documents"))
        await conn.execute(text("DELETE FROM workspaces"))
        try:
            await conn.execute(text("DELETE FROM llm_provider_configs"))
        except Exception:
            pass
        # Commit is automatic with engine.begin() context manager
    yield


# Alias for backward compatibility
clean_workspaces = clean_database


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
