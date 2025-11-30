"""Pytest fixtures for database connection tests"""

import os
import pytest
import asyncio
from urllib.parse import urlparse
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from ragitect.services.database import DatabaseManager


@pytest.fixture(autouse=True)
def clean_db_manager(request):
    """
    Automatically clean up the database manager and enforce safety checks.

    This autouse fixture ensures that:
    1. Integration tests only run against a database ending in '_test'.
    2. The DatabaseManager engine is closed before and after each test
       to ensure a clean state.
    """
    # --- Safety Check for Integration Tests ---
    if "integration" in request.keywords:
        db_url = os.getenv("DATABASE_URL")
        # The test itself will skip if the URL is not set, so we only check if it *is* set.
        if db_url:
            parsed_url = urlparse(db_url)
            db_name = parsed_url.path.lstrip("/")
            if not db_name.endswith("_test"):
                pytest.fail(
                    f"\n\n*** SAFETY ABORT ***\n"
                    f"Attempting to run integration tests on a non-test database: '{db_name}'.\n"
                    f"To prevent data loss, the database name in DATABASE_URL must end with '_test'.\n"
                )
    # --- End Safety Check ---

    db_manager = DatabaseManager.get_instance()

    # Cleanup before test
    if db_manager._engine:
        asyncio.run(db_manager.close())

    yield db_manager

    # Cleanup after test
    if db_manager._engine:
        asyncio.run(db_manager.close())


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
