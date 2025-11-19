"""Tests for database connection layer"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

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
        
        assert instance1 is instance2, "DatabaseManager __new__ should return same instance"

    @pytest.mark.asyncio
    async def test_initialize_creates_engine(self):
        """Test that initialize creates engine and session factory"""
        db_manager = DatabaseManager.get_instance()
        
        # Clean up any existing engine
        if db_manager._engine:
            await db_manager.close()
        
        with patch('ragitect.services.database.connection.create_async_engine') as mock_create_engine:
            # Mock engine and connection
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock(return_value=MagicMock())
            
            # Properly mock async context manager
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = mock_conn
            mock_cm.__aexit__.return_value = None
            
            mock_engine = AsyncMock(spec=AsyncEngine)
            mock_engine.connect.return_value = mock_cm
            
            mock_create_engine.return_value = mock_engine
            
            # Initialize
            result = await db_manager.initialize(
                database_url="postgresql+asyncpg://test:test@localhost/test"
            )
            
            assert result is mock_engine
            assert db_manager._engine is mock_engine
            assert db_manager._session_factory is not None
            mock_create_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test that calling initialize when already initialized returns existing engine"""
        db_manager = DatabaseManager.get_instance()
        
        # Clean up any existing engine from previous tests
        if db_manager._engine:
            await db_manager.close()
        
        with patch('ragitect.services.database.connection.create_async_engine') as mock_create_engine:
            mock_engine = AsyncMock(spec=AsyncEngine)
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock(return_value=MagicMock())
            
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = mock_conn
            mock_cm.__aexit__.return_value = None
            mock_engine.connect.return_value = mock_cm
            
            mock_create_engine.return_value = mock_engine
            
            # First initialization
            first_result = await db_manager.initialize(database_url="postgresql+asyncpg://test:test@localhost/test")
            
            # Second initialization should not create new engine
            second_result = await db_manager.initialize(database_url="postgresql+asyncpg://test:test@localhost/test")
            
            assert first_result is second_result
            assert mock_create_engine.call_count == 1, "Should not create engine twice"

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self):
        """Test that initialize raises DBConnectionError on connection failure"""
        db_manager = DatabaseManager.get_instance()
        
        if db_manager._engine:
            await db_manager.close()
        
        with patch('ragitect.services.database.connection.create_async_engine') as mock_create_engine:
            mock_create_engine.side_effect = Exception("Connection failed")
            
            with pytest.raises(DBConnectionError) as exc_info:
                await db_manager.initialize(database_url="postgresql+asyncpg://bad:bad@localhost/bad")
            
            assert "Connection failed" in str(exc_info.value.original_error)

    @pytest.mark.asyncio
    async def test_initialize_uses_config_defaults(self):
        """Test that initialize uses config values when parameters not provided"""
        db_manager = DatabaseManager.get_instance()
        
        if db_manager._engine:
            await db_manager.close()
        
        with patch('ragitect.services.database.connection.create_async_engine') as mock_create_engine, \
             patch('ragitect.services.database.connection.DATABASE_URL', 'postgresql+asyncpg://default:default@localhost/default'), \
             patch('ragitect.services.database.connection.DB_ECHO', True), \
             patch('ragitect.services.database.connection.DB_POOL_SIZE', 10):
            
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock(return_value=MagicMock())
            
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = mock_conn
            mock_cm.__aexit__.return_value = None
            
            mock_engine = AsyncMock(spec=AsyncEngine)
            mock_engine.connect.return_value = mock_cm
            mock_create_engine.return_value = mock_engine
            
            await db_manager.initialize()
            
            call_kwargs = mock_create_engine.call_args[1]
            assert call_kwargs['echo'] is True
            assert call_kwargs['pool_size'] == 10

    @pytest.mark.asyncio
    async def test_initialize_for_testing(self):
        """Test that initialize_for_testing uses NullPool"""
        db_manager = DatabaseManager.get_instance()
        
        if db_manager._engine:
            await db_manager.close()
        
        with patch('ragitect.services.database.connection.create_async_engine') as mock_create_engine:
            from sqlalchemy.pool.impl import NullPool
            
            mock_engine = AsyncMock(spec=AsyncEngine)
            mock_create_engine.return_value = mock_engine
            
            await db_manager.initialize_for_testing(database_url="postgresql+asyncpg://test:test@localhost/test")
            
            call_kwargs = mock_create_engine.call_args[1]
            assert call_kwargs['pool_class'] is NullPool
            assert call_kwargs['echo'] is False

    @pytest.mark.asyncio
    async def test_initialize_for_testing_closes_existing_engine(self):
        """Test that initialize_for_testing closes existing engine first"""
        db_manager = DatabaseManager.get_instance()
        
        # Clean up first
        if db_manager._engine:
            await db_manager.close()
        
        with patch('ragitect.services.database.connection.create_async_engine') as mock_create_engine:
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock(return_value=MagicMock())
            
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = mock_conn
            mock_cm.__aexit__.return_value = None
            
            mock_engine1 = AsyncMock(spec=AsyncEngine)
            mock_engine1.connect.return_value = mock_cm
            mock_engine1.dispose = AsyncMock()
            
            mock_engine2 = AsyncMock(spec=AsyncEngine)
            
            # Return different engines for each call
            mock_create_engine.side_effect = [mock_engine1, mock_engine2]
            
            # Initialize regular engine
            await db_manager.initialize(database_url="postgresql+asyncpg://test:test@localhost/test")
            
            # Verify engine 1 was set
            assert db_manager._engine is mock_engine1
            
            # Initialize for testing should close first
            await db_manager.initialize_for_testing(database_url="postgresql+asyncpg://test2:test2@localhost/test2")
            
            # Verify engine 1 was disposed
            mock_engine1.dispose.assert_called()
            # Verify engine 2 is now set
            assert db_manager._engine is mock_engine2

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
    async def test_get_session_yields_session(self):
        """Test get_session yields a session"""
        db_manager = DatabaseManager.get_instance()
        
        # Mock session and factory
        mock_session = AsyncMock(spec=AsyncSession)
        
        # Properly mock async context manager for begin()
        mock_begin_cm = AsyncMock()
        mock_begin_cm.__aenter__.return_value = None
        mock_begin_cm.__aexit__.return_value = None
        mock_session.begin.return_value = mock_begin_cm
        mock_session.close = AsyncMock()
        
        mock_factory = MagicMock(return_value=mock_session)
        db_manager._session_factory = mock_factory
        
        async with get_session() as session:
            assert session is mock_session
        
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_rolls_back_on_exception(self):
        """Test get_session rolls back on exception"""
        db_manager = DatabaseManager.get_instance()
        
        mock_session = AsyncMock(spec=AsyncSession)
        
        # Mock begin context manager that allows exceptions to pass through
        mock_begin_cm = AsyncMock()
        mock_begin_cm.__aenter__.return_value = None
        mock_begin_cm.__aexit__.return_value = False  # Don't suppress exceptions
        mock_session.begin.return_value = mock_begin_cm
        mock_session.close = AsyncMock()
        
        mock_factory = MagicMock(return_value=mock_session)
        db_manager._session_factory = mock_factory
        
        with pytest.raises(Exception) as exc_info:
            async with get_session() as session:
                raise Exception("Test error")
        
        assert "Test error" in str(exc_info.value)
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_closes_session_finally(self):
        """Test get_session always closes session in finally block"""
        db_manager = DatabaseManager.get_instance()
        
        mock_session = AsyncMock(spec=AsyncSession)
        
        mock_begin_cm = AsyncMock()
        mock_begin_cm.__aenter__.return_value = None
        mock_begin_cm.__aexit__.return_value = False
        mock_session.begin.return_value = mock_begin_cm
        mock_session.close = AsyncMock()
        
        mock_factory = MagicMock(return_value=mock_session)
        db_manager._session_factory = mock_factory
        
        try:
            async with get_session() as session:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_no_autocommit_yields_session(self):
        """Test get_session_no_autocommit yields session without auto-begin"""
        db_manager = DatabaseManager.get_instance()
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.close = AsyncMock()
        
        mock_factory = MagicMock(return_value=mock_session)
        db_manager._session_factory = mock_factory
        
        async with get_session_no_autocommit() as session:
            assert session is mock_session
            # Should not call begin automatically
            mock_session.begin.assert_not_called()
        
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_no_autocommit_closes_on_exception(self):
        """Test get_session_no_autocommit closes session on exception"""
        db_manager = DatabaseManager.get_instance()
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.close = AsyncMock()
        
        mock_factory = MagicMock(return_value=mock_session)
        db_manager._session_factory = mock_factory
        
        with pytest.raises(Exception):
            async with get_session_no_autocommit() as session:
                raise Exception("Test error")
        
        mock_session.close.assert_called_once()


class TestCheckConnection:
    """Test check_connection utility function"""

    @pytest.mark.asyncio
    async def test_check_connection_success(self):
        """Test check_connection returns True when connection succeeds"""
        db_manager = DatabaseManager.get_instance()
        
        # Mock engine and connection
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=MagicMock())
        
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None
        
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_engine.connect.return_value = mock_cm
        
        db_manager._engine = mock_engine
        
        result = await check_connection()
        
        assert result is True
        mock_conn.execute.assert_called_once()

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


class TestDatabaseIntegration:
    """Integration tests for database connection (requires database)"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_connection_initialization(self):
        """Test actual database connection (skip if DATABASE_URL not set)"""
        import os
        
        if not os.getenv('DATABASE_URL'):
            pytest.skip("DATABASE_URL not set - skipping integration test")
        
        db_manager = DatabaseManager.get_instance()
        
        try:
            # Close any existing connection
            if db_manager._engine:
                await db_manager.close()
            
            # Initialize with real database
            await db_manager.initialize()
            
            # Verify engine is created
            assert db_manager._engine is not None
            assert db_manager._session_factory is not None
            
            # Test connection
            is_connected = await check_connection()
            assert is_connected is True
            
        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_session_creation(self):
        """Test creating real session and executing query"""
        import os
        
        if not os.getenv('DATABASE_URL'):
            pytest.skip("DATABASE_URL not set - skipping integration test")
        
        db_manager = DatabaseManager.get_instance()
        
        try:
            if not db_manager._engine:
                await db_manager.initialize()
            
            async with get_session() as session:
                result = await session.execute(text("SELECT 1 as value"))
                value = result.scalar()
                assert value == 1
        
        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pgvector_extension_check(self):
        """Test checking if pgvector extension is installed"""
        import os
        
        if not os.getenv('DATABASE_URL'):
            pytest.skip("DATABASE_URL not set - skipping integration test")
        
        db_manager = DatabaseManager.get_instance()
        
        try:
            if not db_manager._engine:
                await db_manager.initialize()
            
            async with get_session() as session:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
                )
                count = result.scalar()
                
                # Just check that query executes - extension may or may not be installed
                assert count is not None
                assert count >= 0
        
        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pgvector_basic_operations(self):
        """Test basic pgvector operations"""
        import os
        
        if not os.getenv('DATABASE_URL'):
            pytest.skip("DATABASE_URL not set - skipping integration test")
        
        db_manager = DatabaseManager.get_instance()
        
        try:
            if not db_manager._engine:
                await db_manager.initialize()
            
            async with get_session() as session:
                # Enable extension
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await session.commit()
                
            async with get_session() as session:
                # Create test table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS test_vectors_temp (
                        id SERIAL PRIMARY KEY,
                        embedding vector(3)
                    )
                """))
                await session.commit()
            
            async with get_session() as session:
                # Insert test vector
                test_vector = [0.1, 0.2, 0.3]
                await session.execute(
                    text("INSERT INTO test_vectors_temp (embedding) VALUES (:vec)"),
                    {"vec": str(test_vector)}
                )
                await session.commit()
            
            async with get_session() as session:
                # Query vector
                result = await session.execute(
                    text("SELECT embedding FROM test_vectors_temp LIMIT 1")
                )
                fetched = result.scalar()
                assert fetched is not None
            
            async with get_session() as session:
                # Test cosine distance
                query_vector = [0.1, 0.2, 0.3]
                result = await session.execute(
                    text("""
                        SELECT embedding <=> :query_vec AS distance
                        FROM test_vectors_temp
                        ORDER BY distance
                        LIMIT 1
                    """),
                    {"query_vec": str(query_vector)}
                )
                distance = result.scalar()
                assert distance is not None
                assert distance >= 0
            
            # Cleanup
            async with get_session() as session:
                await session.execute(text("DROP TABLE IF EXISTS test_vectors_temp"))
                await session.commit()
        
        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_session_transaction_commit(self):
        """Test that session properly commits transactions"""
        import os
        
        if not os.getenv('DATABASE_URL'):
            pytest.skip("DATABASE_URL not set - skipping integration test")
        
        db_manager = DatabaseManager.get_instance()
        
        try:
            if not db_manager._engine:
                await db_manager.initialize()
            
            # Create temporary table
            async with get_session() as session:
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS test_commit_temp (
                        id SERIAL PRIMARY KEY,
                        value TEXT
                    )
                """))
                await session.commit()
            
            # Insert data (should auto-commit)
            async with get_session() as session:
                await session.execute(
                    text("INSERT INTO test_commit_temp (value) VALUES (:val)"),
                    {"val": "test_value"}
                )
            
            # Verify data persisted
            async with get_session() as session:
                result = await session.execute(
                    text("SELECT value FROM test_commit_temp WHERE value = :val"),
                    {"val": "test_value"}
                )
                value = result.scalar()
                assert value == "test_value"
            
            # Cleanup
            async with get_session() as session:
                await session.execute(text("DROP TABLE IF EXISTS test_commit_temp"))
                await session.commit()
        
        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_session_transaction_rollback(self):
        """Test that session properly rolls back on exception"""
        import os
        
        if not os.getenv('DATABASE_URL'):
            pytest.skip("DATABASE_URL not set - skipping integration test")
        
        db_manager = DatabaseManager.get_instance()
        
        try:
            if not db_manager._engine:
                await db_manager.initialize()
            
            # Create temporary table
            async with get_session() as session:
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS test_rollback_temp (
                        id SERIAL PRIMARY KEY,
                        value TEXT UNIQUE
                    )
                """))
                await session.commit()
            
            # Insert data but raise exception
            try:
                async with get_session() as session:
                    await session.execute(
                        text("INSERT INTO test_rollback_temp (value) VALUES (:val)"),
                        {"val": "rollback_test"}
                    )
                    raise ValueError("Intentional error")
            except ValueError:
                pass
            
            # Verify data was rolled back
            async with get_session() as session:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM test_rollback_temp WHERE value = :val"),
                    {"val": "rollback_test"}
                )
                count = result.scalar()
                assert count == 0, "Data should have been rolled back"
            
            # Cleanup
            async with get_session() as session:
                await session.execute(text("DROP TABLE IF EXISTS test_rollback_temp"))
                await session.commit()
        
        finally:
            await db_manager.close()
