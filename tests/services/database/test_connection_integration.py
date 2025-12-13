"""Integration tests for database connection layer (requires database)"""

import pytest
from sqlalchemy import text
from ragitect.services.database import check_connection, get_session

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestDatabaseIntegration:
    """Integration tests for database connection (requires database)"""

    async def test_real_connection_initialization(self, setup_integration_database):
        """Test actual database connection (skip if DATABASE_URL not set)"""
        db_manager = setup_integration_database

        # Verify engine is created
        assert db_manager._engine is not None
        assert db_manager._session_factory is not None

        # Test connection
        is_connected = await check_connection()
        assert is_connected is True

    async def test_real_session_creation(self, setup_integration_database):
        """Test creating real session and executing query"""
        async with get_session() as session:
            result = await session.execute(text("SELECT 1 as value"))
            value = result.scalar()
            assert value == 1

    async def test_pgvector_extension_check(self, setup_integration_database):
        """Test checking if pgvector extension is installed"""
        async with get_session() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
            )
            count = result.scalar()

            # Just check that query executes - extension may or may not be installed
            assert count is not None
            assert count >= 0

    async def test_pgvector_basic_operations(self, setup_integration_database):
        """Test basic pgvector operations"""
        # setup_integration_database should have already created the extension
        # but let's verify we can use it

        async with get_session() as session:
            # Create test table
            await session.execute(
                text("""
                    CREATE TABLE IF NOT EXISTS test_vectors_temp (
                        id SERIAL PRIMARY KEY,
                        embedding vector(3)
                    )
                """)
            )

        try:
            async with get_session() as session:
                # Insert test vector
                test_vector = [0.1, 0.2, 0.3]
                await session.execute(
                    text("INSERT INTO test_vectors_temp (embedding) VALUES (:vec)"),
                    {"vec": str(test_vector)},
                )

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
                    {"query_vec": str(query_vector)},
                )
                distance = result.scalar()
                assert distance is not None
                assert distance >= 0
        finally:
            # Cleanup
            async with get_session() as session:
                await session.execute(text("DROP TABLE IF EXISTS test_vectors_temp"))
                await session.commit()

    async def test_session_transaction_commit(self, setup_integration_database):
        """Test that session properly commits transactions"""
        # Create temporary table
        async with get_session() as session:
            await session.execute(
                text("""
                    CREATE TABLE IF NOT EXISTS test_commit_temp (
                        id SERIAL PRIMARY KEY,
                        value TEXT
                    )
                """)
            )

        try:
            # Insert data (should auto-commit)
            async with get_session() as session:
                await session.execute(
                    text("INSERT INTO test_commit_temp (value) VALUES (:val)"),
                    {"val": "test_value"},
                )

            # Verify data persisted
            async with get_session() as session:
                result = await session.execute(
                    text("SELECT value FROM test_commit_temp WHERE value = :val"),
                    {"val": "test_value"},
                )
                value = result.scalar()
                assert value == "test_value"
        finally:
            # Cleanup
            async with get_session() as session:
                await session.execute(text("DROP TABLE IF EXISTS test_commit_temp"))
                await session.commit()

    async def test_session_transaction_rollback(self, setup_integration_database):
        """Test that session properly rolls back on exception"""
        # Create temporary table
        async with get_session() as session:
            await session.execute(
                text("""
                    CREATE TABLE IF NOT EXISTS test_rollback_temp (
                        id SERIAL PRIMARY KEY,
                        value TEXT UNIQUE
                    )
                """)
            )

        try:
            # Insert data but raise exception
            try:
                async with get_session() as session:
                    await session.execute(
                        text("INSERT INTO test_rollback_temp (value) VALUES (:val)"),
                        {"val": "rollback_test"},
                    )
                    raise ValueError("Intentional error")
            except ValueError:
                pass

            # Verify data was rolled back
            async with get_session() as session:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM test_rollback_temp WHERE value = :val"),
                    {"val": "rollback_test"},
                )
                count = result.scalar()
                assert count == 0, "Data should have been rolled back"
        finally:
            # Cleanup
            async with get_session() as session:
                await session.execute(text("DROP TABLE IF EXISTS test_rollback_temp"))
                await session.commit()
