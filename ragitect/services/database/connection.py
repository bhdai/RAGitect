"""Database connection management"""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool.impl import NullPool

from ragitect.services.config import (
    DATABASE_URL,
    DB_ECHO,
    DB_MAX_OVERFLOW,
    DB_POOL_RECYCLE,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT,
)
from ragitect.services.database.exceptions import ConnectionError as DBConnectionError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Singleton manager for database engine lifecycle

    Attributes:
        _instance: Singleton instance
        _engine: SQLAlchemy AsyncEngine
        _session_factory: Session factory for creating AsyncSession
    """

    _instance: "DatabaseManager | None" = None
    _engine: AsyncEngine | None = None
    _session_factory: async_sessionmaker[AsyncSession] | None = None

    def __new__(cls):
        """Get the singleton instance

        Returns:
            DatabaseManager: singleton instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        """Get the singletion instance

        Returns:
            DatabaseManager: singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(
        self,
        database_url: str | None = None,
        echo: bool | None = None,
        pool_size: int | None = None,
        max_overflow: int | None = None,
        poll_timeout: int | None = None,
        pool_recycle: int | None = None,
    ) -> AsyncEngine:
        """Initialize the database engine and session factory

        Args:
            database_url: PostgreSQL connection string (defaults to config)
            echo: enable sql query logging (defaullt to config)
            pool_size: connection pool size (default to config)
            max_overflow: max overflow connections (default to config)
            poll_timeout: connection pool timeout (default to config)
            pool_recycle: recycle connections after this many seconds (default to config)

        Returns:
            AsyncEngine: initialized database engine

        Raises:
            DBConnectionError: if connection cannot be established
        """
        if self._engine is not None:
            logger.info("Database engine already initalized")
            return self._engine

        url = database_url or DATABASE_URL
        echo = echo if echo is not None else DB_ECHO
        pool_size = pool_size or DB_POOL_SIZE
        max_overflow = max_overflow or DB_MAX_OVERFLOW
        poll_timeout = poll_timeout or DB_POOL_TIMEOUT
        pool_recycle = pool_recycle or DB_POOL_RECYCLE

        logger.info(
            f"Intializing database engine: {url.split('@')[-1]}"
        )  # hide the credentials

        try:
            self._engine = create_async_engine(
                url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=poll_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,  # verify connection before using
            )

            # test connection immediately
            async with self._engine.connect() as conn:
                _ = await conn.execute(text("SELECT 1"))

            # create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,  # don't expire object after commit
            )

            logger.info("Database engine initialized successfully")
            return self._engine
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {str(e)}")
            raise DBConnectionError(original_error=e)

    async def initialize_for_testing(self, database_url: str) -> AsyncEngine:
        """Intialize engine for testing with NullPool

        Args:
            database_url: test database_url

        Returns:
            AsyncEngine: initialized test database engine

        Raises:
            DBConnectionError: if connection cannot be established
        """
        if self._engine is not None:
            await self.close()

        logger.info("Initializing database engine for testing")

        try:
            self._engine = create_async_engine(
                database_url, echo=False, pool_class=NullPool
            )

            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            return self._engine
        except Exception as e:
            logger.error(f"Failed to initialize test database: {str(e)}")
            raise DBConnectionError(original_error=e)

    def get_engine(self) -> AsyncEngine:
        """Get the session factory

        Returns:
            AsyncEngine: database engine

        Raises:
            RuntimeError: if engine not initialized
        """
        if self._engine is None:
            raise RuntimeError(
                "Databse engine not initialized "
                + "Call DatabaseManager.get_instance().initialize() first"
            )
        return self._engine

    def get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory

        Returns:
            async_sessionmaker[AsyncSession]: session factory

        Raises:
            RuntimeError: if session factory not initialized
        """
        if self._session_factory is None:
            raise RuntimeError(
                "Session factory not initialized "
                + "Call DatabaseManager.get_instance().initialize() first"
            )
        return self._session_factory

    async def close(self) -> None:
        """Close the engine and dispose of connection pool"""
        if self._engine is not None:
            logger.info("Closing databse engine")
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database engine closed")


# convenience function for common operations
def get_engine() -> AsyncEngine:
    """Get the current database engine

    Returns:
        AsyncEngine: database engine

    Raises:
        RuntimeError: If engine not initialized
    """
    return DatabaseManager.get_instance().get_engine()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the session factory

    Returns:
        async_sessionmaker[AsyncSession]: session factory

    Raises:
        RuntimeError: If session factory not initialized
    """
    return DatabaseManager.get_instance().get_session_factory()


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database session with automatic transaction handling

    this context manager:
    1. creates a new session
    2. begins a transaction
    3. commits on successful completion
    4. rolls back on exception
    5. closes the section on final block

    Usage:
    >>> async with get_session() as session:
    :::     repo = WorkspaceRepository(session)
    :::     workspace = await repo.create(name="AI research")
    :::     # auto commit here if no exception

    Yields:
        AsyncSession: database session
    """
    factory = get_session_factory()
    session = factory()

    try:
        async with session.begin():
            yield session
    except Exception:
        logger.debug("Session rollback due to exception")
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_session_no_autocommit() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for session without automatic commit

    Use this function when you need manual transaction control

    Usage:
    >>> async with get_session_no_autocommit() as session:
    :::    async with session.begin():
    :::        repo = WorkspaceRepository(session)
    :::        workspace = await repo.create(name="AI Research")
    :::        # ... more operations ...
    :::        await session.commit()  # manual commit

    Yields:
        AsyncSession: database session
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        await session.close()


# testing and utility functions


async def check_connection() -> bool:
    """Verify database connection is working

    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            _ = await conn.execute(text("SELECT 1"))
        logger.info("Database connection check succeeded")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def create_table(engine: AsyncEngine | None = None) -> None:
    """Create all the tables defined in SQLAlchemy models

    WARNING: This is for testing or initial setup only.
    In production, use Alembic migrations instead

    Args:
        engine: Optional AsyncEngine to use (defaults to singleton engine)
    """
    from ragitect.services.database.models import Base

    engine = engine or get_engine()

    logger.info("Creating all database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("All database tables created successfully.")


async def drop_table(engine: AsyncEngine | None = None) -> None:
    """Drop all the tables defined in SQLAlchemy models

    WARNING: This is destructive! Only use for testing.

    Args:
        engine: Optional AsyncEngine to use (defaults to singleton engine)
    """
    from ragitect.services.database.models import Base

    engine = engine or get_engine()
    logger.info("Dropping all database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("All database tables dropped successfully.")


async def reset_database(engine: AsyncEngine | None = None) -> None:
    """Drop and recreate all tables in the database

    WARNING: This is destructive! Only use for testing.

    Args:
        engine: Optional AsyncEngine to use (defaults to singleton engine)
    """
    logger.info("Resetting the database...")
    engine = engine or get_engine()
    await drop_table(engine)
    await create_table(engine)
    logger.info("Database reset successfully.")
