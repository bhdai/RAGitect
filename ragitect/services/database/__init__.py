"""Database package - SQLAlchemy async ORM with pgvector support"""

from ragitect.services.database.connection import (
    DatabaseManager,
    check_connection,
    get_engine,
    get_session,
    get_session_factory,
    get_session_no_autocommit,
    create_table,
    drop_table,
    reset_database,
)

from ragitect.services.database.exceptions import (
    ConnectionError,
    DatabaseError,
    DuplicateError,
    NotFoundError,
    ValidationError,
)

from ragitect.services.database.models import (
    Base,
    Document,
    DocumentChunk,
    Workspace,
)

__all__ = [
    # connection management
    "DatabaseManager",
    "get_engine",
    "get_session_factory",
    "get_session",
    "get_session_no_autocommit",
    # utilities
    "check_connection",
    "create_table",
    "drop_table",
    "reset_database",
    # exceptions
    "DatabaseError",
    "NotFoundError",
    "DuplicateError",
    "ValidationError",
    "ConnectionError",
    # models
    "Base",
    "Document",
    "DocumentChunk",
    "Workspace",
]
