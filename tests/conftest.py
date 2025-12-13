"""Global shared fixtures for RAGitect tests.

Contains:
- Database integration fixtures (setup_integration_database, clean_database)
- Mock fixtures for unit tests (mock_session, mock_async_engine)
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

# --- Integration Test Configuration ---

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


@pytest_asyncio.fixture(scope="module")
async def setup_integration_database(database_url):
    """Initialize test database and create tables for integration tests.

    This fixture:
    1. Connects to the real database (must end in _test)
    2. Drops/Recreates schema
    3. Yields the DatabaseManager
    """
    from ragitect.services.database.connection import DatabaseManager, get_session
    from ragitect.services.database.models import Base

    db_manager = DatabaseManager.get_instance()

    # Close any existing connection
    if db_manager._engine:
        await db_manager.close()

    # Initialize for testing
    await db_manager.initialize_for_testing(database_url)

    # Create pgvector extension
    async with get_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await session.commit()

    # Create all tables
    async with db_manager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield db_manager

    # Cleanup: drop tables and close connection
    if db_manager._engine:
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await db_manager.close()


@pytest_asyncio.fixture
async def clean_database(setup_integration_database):
    """Clean all tables before each test case."""
    db_manager = setup_integration_database

    if not db_manager._engine:
        # Should not happen if setup_integration_database worked
        await db_manager.initialize_for_testing(os.getenv("DATABASE_URL"))

    async with db_manager._engine.begin() as conn:
        # Clean test data - ordering matters due to foreign keys
        await conn.execute(text("DELETE FROM document_chunks"))
        await conn.execute(text("DELETE FROM documents"))
        await conn.execute(text("DELETE FROM workspaces"))
        try:
            await conn.execute(text("DELETE FROM llm_provider_configs"))
        except Exception:
            pass
    yield


# Alias for backward compatibility if needed
clean_workspaces = clean_database


# --- Unit Test Mocks ---


@pytest.fixture
def mock_async_engine():
    """Create a properly configured mock AsyncEngine with connection support"""
    mock_engine = AsyncMock(spec=AsyncEngine)
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=MagicMock())

    # properly mock async context manager for connection
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_conn
    mock_cm.__aexit__.return_value = None
    mock_engine.connect.return_value = mock_cm

    mock_engine.dispose = AsyncMock()

    return mock_engine


@pytest.fixture
def mock_session():
    """Create a properly configured mock AsyncSession"""
    session = AsyncMock(spec=AsyncSession)
    session.close = AsyncMock()

    # mock begin context manager for transaction support
    mock_begin_cm = AsyncMock()
    mock_begin_cm.__aenter__.return_value = None
    mock_begin_cm.__aexit__.return_value = False  # don't suppress exceptions
    session.begin.return_value = mock_begin_cm

    return session
