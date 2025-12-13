"""Pytest fixtures for database connection tests"""

import pytest
import asyncio
from ragitect.services.database import DatabaseManager


@pytest.fixture(autouse=True)
def clean_db_manager_singleton():
    """
    Automatically clean up the database manager singleton.

    This ensures that the singleton is reset before and after each test
    to prevent state leakage (like remaining open connections).
    """
    db_manager = DatabaseManager.get_instance()

    # Cleanup before test
    if db_manager._engine:
        asyncio.run(db_manager.close())

    yield db_manager

    # Cleanup after test
    if db_manager._engine:
        asyncio.run(db_manager.close())
