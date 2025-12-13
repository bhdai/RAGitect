"""Pytest fixtures for database connection tests"""

import pytest
import asyncio
from ragitect.services.database import DatabaseManager


@pytest.fixture(autouse=True)
def clean_db_manager_singleton(request):
    """
    Automatically clean up the database manager singleton.

    This ensures that the singleton is reset before and after each test
    to prevent state leakage (like remaining open connections).

    Skips cleanup for integration tests which manage their own connection lifecycle.
    """
    # Skip for integration tests
    if "integration" in request.keywords:
        yield
        return

    db_manager = DatabaseManager.get_instance()

    # Cleanup before test
    if db_manager._engine:
        asyncio.run(db_manager.close())

    yield db_manager

    # Cleanup after test
    if db_manager._engine:
        asyncio.run(db_manager.close())
