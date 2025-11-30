"""Custom exceptions for database operations"""

from typing import Any
from uuid import UUID


class DatabaseError(Exception):
    """Base exception for all the database related errors

    Attributes:
        message: str: Description of the error
    """

    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class NotFoundError(DatabaseError):
    """Raised when the entity is not found in the database but caller expect result

    Attributes:
        entity_type: type of the entity (e.g., "Workspace", "Document")
        entity_id: ID of the missing entry
    """

    entity_type: str
    entity_id: str | UUID

    def __init__(self, entity_type: str, entity_id: str | UUID):
        self.entity_type = entity_type
        self.entity_id = entity_id
        message = f"{self.entity_type} with ID {str(self.entity_id)} not found."
        super().__init__(message)


class DuplicateError(DatabaseError):
    """Raised when attempting to create an entity that violates unique constraints

    Attributes:
        entity_type: type of the entity (e.g., "Workspace", "Document")
        field: field name that violate the constraint
        value: the duplicate value
    """

    entity_type: str
    field: str
    value: Any  # pyright: ignore[reportExplicitAny]

    def __init__(self, entity_type: str, field: str, value: Any):  # pyright: ignore[reportExplicitAny, reportAny]
        self.entity_type = entity_type
        self.field = field
        self.value = value
        message = (
            f"{self.entity_type} with {self.field} = '{self.value}' already exists."  # pyright: ignore[reportAny]
        )
        super().__init__(message)


class ValidationError(DatabaseError):
    """Raised when invalid data is provided to a repository method

    Attributes:
        field: field name that failed validation
        reason: description of the validation failure
    """

    field: str
    reason: str

    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        message = f"Validation error on field '{self.field}': {self.reason}"
        super().__init__(message)


class ConnectionError(DatabaseError):
    """Raised when database connection/pool occur

    Attributes:
        original_error: the underlying exception that cause the connection failure
    """

    original_error: Exception

    def __init__(self, original_error: Exception):
        self.original_error = original_error
        message = f"Database connection error: {str(self.original_error)}"
        super().__init__(message)
