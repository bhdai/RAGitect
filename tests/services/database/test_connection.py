"""Tests for database connection layer"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncEngine

from ragitect.services.database import (
    DatabaseManager,
    check_connection,
    get_engine,
    get_session,
    get_session_factory,
    get_session_no_autocommit,
)
from ragitect.services.database.exceptions import ConnectionError as DBConnectionError


class TestDatabaseManager:
    """Test DatabaseManager singleton class"""

    def test_singleton_pattern(self):
        """Test that DatabaseManager follows singleton pattern"""
        instance1 = DatabaseManager.get_instance()
        instance2 = DatabaseManager.get_instance()

        assert instance1 is instance2, "DatabaseManager should return same instance"

    def test_singleton_new_method(self):
        """Test that __new__ also enforces singleton"""
        instance1 = DatabaseManager()
        instance2 = DatabaseManager()

        assert instance1 is instance2, (
            "DatabaseManager __new__ should return same instance"
        )

    @pytest.mark.asyncio
    async def test_initialize_creates_engine(self, clean_db_manager, mock_async_engine):
        """Test that initialize creates engine and session factory"""
        with patch(
            "ragitect.services.database.connection.create_async_engine"
        ) as mock_create_engine:
            mock_create_engine.return_value = mock_async_engine

            # Initialize
            result = await clean_db_manager.initialize(
                database_url="postgresql+asyncpg://test:test@localhost/test"
            )

            assert result is mock_async_engine
            assert clean_db_manager._engine is mock_async_engine
            assert clean_db_manager._session_factory is not None
            mock_create_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(
        self, clean_db_manager, mock_async_engine
    ):
        """Test that calling initialize when already initialized returns existing engine"""
        with patch(
            "ragitect.services.database.connection.create_async_engine"
        ) as mock_create_engine:
            mock_create_engine.return_value = mock_async_engine

            # First initialization
            first_result = await clean_db_manager.initialize(
                database_url="postgresql+asyncpg://test:test@localhost/test"
            )

            # Second initialization should not create new engine
            second_result = await clean_db_manager.initialize(
                database_url="postgresql+asyncpg://test:test@localhost/test"
            )

            assert first_result is second_result
            assert mock_create_engine.call_count == 1, "Should not create engine twice"

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, clean_db_manager):
        """Test that initialize raises DBConnectionError on connection failure"""
        with patch(
            "ragitect.services.database.connection.create_async_engine"
        ) as mock_create_engine:
            mock_create_engine.side_effect = Exception("Connection failed")

            with pytest.raises(DBConnectionError) as exc_info:
                await clean_db_manager.initialize(
                    database_url="postgresql+asyncpg://bad:bad@localhost/bad"
                )

            assert "Connection failed" in str(exc_info.value.original_error)

    @pytest.mark.asyncio
    async def test_initialize_uses_config_defaults(
        self, clean_db_manager, mock_async_engine
    ):
        """Test that initialize uses config values when parameters not provided"""
        with (
            patch(
                "ragitect.services.database.connection.create_async_engine"
            ) as mock_create_engine,
            patch(
                "ragitect.services.database.connection.DATABASE_URL",
                "postgresql+asyncpg://default:default@localhost/default",
            ),
            patch("ragitect.services.database.connection.DB_ECHO", True),
            patch("ragitect.services.database.connection.DB_POOL_SIZE", 10),
        ):
            mock_create_engine.return_value = mock_async_engine

            await clean_db_manager.initialize()

            call_kwargs = mock_create_engine.call_args[1]
            assert call_kwargs["echo"] is True
            assert call_kwargs["pool_size"] == 10

    @pytest.mark.asyncio
    async def test_initialize_for_testing(self, clean_db_manager, mock_async_engine):
        """Test that initialize_for_testing uses NullPool"""
        with patch(
            "ragitect.services.database.connection.create_async_engine"
        ) as mock_create_engine:
            from sqlalchemy.pool.impl import NullPool

            mock_create_engine.return_value = mock_async_engine

            await clean_db_manager.initialize_for_testing(
                database_url="postgresql+asyncpg://test:test@localhost/test"
            )

            call_kwargs = mock_create_engine.call_args[1]
            assert call_kwargs["poolclass"] is NullPool
            assert call_kwargs["echo"] is False

    @pytest.mark.asyncio
    async def test_initialize_for_testing_closes_existing_engine(
        self, clean_db_manager
    ):
        """Test that initialize_for_testing closes existing engine first"""
        with patch(
            "ragitect.services.database.connection.create_async_engine"
        ) as mock_create_engine:
            mock_engine1 = AsyncMock(spec=AsyncEngine)
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock(return_value=MagicMock())

            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = mock_conn
            mock_cm.__aexit__.return_value = None
            mock_engine1.connect.return_value = mock_cm
            mock_engine1.dispose = AsyncMock()

            mock_engine2 = AsyncMock(spec=AsyncEngine)

            # Return different engines for each call
            mock_create_engine.side_effect = [mock_engine1, mock_engine2]

            # Initialize regular engine
            await clean_db_manager.initialize(
                database_url="postgresql+asyncpg://test:test@localhost/test"
            )

            # Verify engine 1 was set
            assert clean_db_manager._engine is mock_engine1

            # Initialize for testing should close first
            await clean_db_manager.initialize_for_testing(
                database_url="postgresql+asyncpg://test2:test2@localhost/test2"
            )

            # Verify engine 1 was disposed
            mock_engine1.dispose.assert_called()
            # Verify engine 2 is now set
            assert clean_db_manager._engine is mock_engine2

    def test_get_engine_when_initialized(self):
        """Test get_engine returns engine when initialized"""
        db_manager = DatabaseManager.get_instance()
        mock_engine = MagicMock(spec=AsyncEngine)
        db_manager._engine = mock_engine

        result = db_manager.get_engine()

        assert result is mock_engine

    def test_get_engine_when_not_initialized(self):
        """Test get_engine raises RuntimeError when not initialized"""
        db_manager = DatabaseManager.get_instance()
        db_manager._engine = None

        with pytest.raises(RuntimeError) as exc_info:
            db_manager.get_engine()

        assert "not initialized" in str(exc_info.value).lower()

    def test_get_session_factory_when_initialized(self):
        """Test get_session_factory returns factory when initialized"""
        db_manager = DatabaseManager.get_instance()
        mock_factory = MagicMock()
        db_manager._session_factory = mock_factory

        result = db_manager.get_session_factory()

        assert result is mock_factory

    def test_get_session_factory_when_not_initialized(self):
        """Test get_session_factory raises RuntimeError when not initialized"""
        db_manager = DatabaseManager.get_instance()
        db_manager._session_factory = None

        with pytest.raises(RuntimeError) as exc_info:
            db_manager.get_session_factory()

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_close_disposes_engine(self):
        """Test close properly disposes engine"""
        db_manager = DatabaseManager.get_instance()
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_engine.dispose = AsyncMock()
        db_manager._engine = mock_engine
        db_manager._session_factory = MagicMock()

        await db_manager.close()

        mock_engine.dispose.assert_called_once()
        assert db_manager._engine is None
        assert db_manager._session_factory is None

    @pytest.mark.asyncio
    async def test_close_when_no_engine(self):
        """Test close does nothing when no engine exists"""
        db_manager = DatabaseManager.get_instance()
        db_manager._engine = None

        # Should not raise
        await db_manager.close()


class TestConvenienceFunctions:
    """Test convenience functions for engine and session access"""

    def test_get_engine_function(self):
        """Test get_engine convenience function"""
        db_manager = DatabaseManager.get_instance()
        mock_engine = MagicMock(spec=AsyncEngine)
        db_manager._engine = mock_engine

        result = get_engine()

        assert result is mock_engine

    def test_get_engine_function_not_initialized(self):
        """Test get_engine raises when not initialized"""
        db_manager = DatabaseManager.get_instance()
        db_manager._engine = None

        with pytest.raises(RuntimeError):
            get_engine()

    def test_get_session_factory_function(self):
        """Test get_session_factory convenience function"""
        db_manager = DatabaseManager.get_instance()
        mock_factory = MagicMock()
        db_manager._session_factory = mock_factory

        result = get_session_factory()

        assert result is mock_factory

    def test_get_session_factory_function_not_initialized(self):
        """Test get_session_factory raises when not initialized"""
        db_manager = DatabaseManager.get_instance()
        db_manager._session_factory = None

        with pytest.raises(RuntimeError):
            get_session_factory()


class TestSessionContextManagers:
    """Test session context managers"""

    @pytest.mark.asyncio
    async def test_get_session_yields_session(self, clean_db_manager, mock_session):
        """Test get_session yields a session"""
        mock_factory = MagicMock(return_value=mock_session)
        clean_db_manager._session_factory = mock_factory

        async with get_session() as session:
            assert session is mock_session

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_rolls_back_on_exception(
        self, clean_db_manager, mock_session
    ):
        """Test get_session rolls back on exception"""
        mock_factory = MagicMock(return_value=mock_session)
        clean_db_manager._session_factory = mock_factory

        with pytest.raises(Exception) as exc_info:
            async with get_session() as session:
                raise Exception("Test error")

        assert "Test error" in str(exc_info.value)
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_closes_session_finally(
        self, clean_db_manager, mock_session
    ):
        """Test get_session always closes session in finally block"""
        mock_factory = MagicMock(return_value=mock_session)
        clean_db_manager._session_factory = mock_factory

        try:
            async with get_session() as session:
                raise ValueError("Test exception")
        except ValueError:
            pass

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_no_autocommit_yields_session(
        self, clean_db_manager, mock_session
    ):
        """Test get_session_no_autocommit yields session without auto-begin"""
        mock_factory = MagicMock(return_value=mock_session)
        clean_db_manager._session_factory = mock_factory

        async with get_session_no_autocommit() as session:
            assert session is mock_session
            # Should not call begin automatically
            mock_session.begin.assert_not_called()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_no_autocommit_closes_on_exception(
        self, clean_db_manager, mock_session
    ):
        """Test get_session_no_autocommit closes session on exception"""
        mock_factory = MagicMock(return_value=mock_session)
        clean_db_manager._session_factory = mock_factory

        with pytest.raises(Exception):
            async with get_session_no_autocommit() as session:
                raise Exception("Test error")

        mock_session.close.assert_called_once()


class TestCheckConnection:
    """Test check_connection utility function"""

    @pytest.mark.asyncio
    async def test_check_connection_success(self, clean_db_manager, mock_async_engine):
        """Test check_connection returns True when connection succeeds"""
        clean_db_manager._engine = mock_async_engine

        result = await check_connection()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_connection_failure(self):
        """Test check_connection returns False when connection fails"""
        db_manager = DatabaseManager.get_instance()

        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_engine.connect = AsyncMock(side_effect=Exception("Connection error"))

        db_manager._engine = mock_engine

        result = await check_connection()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_connection_no_engine(self):
        """Test check_connection returns False when engine not initialized"""
        db_manager = DatabaseManager.get_instance()
        db_manager._engine = None

        result = await check_connection()

        assert result is False
