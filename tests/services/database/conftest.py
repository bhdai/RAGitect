"""Pytest fixtures for database connection tests"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from ragitect.services.database import DatabaseManager


@pytest.fixture(autouse=True)
def clean_db_manager():
    """Automatically clean up database manager before and after each test"""
    db_manager = DatabaseManager.get_instance()

    # cleanup before test (sync wrapper for async cleanup)
    if db_manager._engine:
        asyncio.run(db_manager.close())

    yield db_manager

    # cleanup after test (optional, but ensures clean state)
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
