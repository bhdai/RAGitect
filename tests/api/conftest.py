"""Pytest fixtures for API tests

Provides test isolation by mocking the database lifespan manager.
This allows API endpoint tests to run without requiring a real database connection.
"""

import pytest
import pytest_asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@asynccontextmanager
async def mock_lifespan(app: FastAPI):
    """Mock lifespan that doesn't initialize real database"""
    yield


@pytest.fixture(autouse=True)
def mock_database_manager():
    """
    Mock the DatabaseManager to prevent real DB connections during API tests.

    This fixture patches the lifespan function in main.py to use a mock
    that doesn't attempt real database connections.
    """
    with patch("main.lifespan", mock_lifespan):
        with patch("main.DatabaseManager") as mock_dm:
            mock_instance = AsyncMock()
            mock_dm.get_instance.return_value = mock_instance
            yield mock_instance


@pytest_asyncio.fixture
async def async_client(mock_database_manager) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for FastAPI app with mocked DB

    This fixture imports main.app after the lifespan is mocked to ensure
    the mock is in place before app initialization.
    """
    # Import app here to ensure mocks are in place
    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
