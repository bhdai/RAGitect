"""Base repository with common utilities"""

import logging
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio.session import AsyncSession

from ragitect.services.database.exceptions import NotFoundError, ValidationError
from ragitect.services.database.models import Base

logger = logging.getLogger(__name__)


ModelType = TypeVar(
    "ModelType", bound=Base
)  # only accept type that are children or subclasses of Base


class BaseRepository(Generic[ModelType]):
    """Base repository class with common database operations

    Provies common patterns for CRUD operations that can be inherited
    by specific repository classes

    Attributes:
        session: SQLAlchemy async session
        model_class: SQLAlchemy model class (e.g., Workspace, Document)

    Usage:
    >>>> class WorkspaceRepository(BaseRepository[Workspace]):
    ::::    def __init__(self, session: AsyncSession):
    ::::        super().__init__(session, Workspace)

    """

    session: AsyncSession
    model_class: type[ModelType]
    model_name: str

    def __init__(self, session: AsyncSession, model_class: type[ModelType]):
        self.session = session
        self.model_class = model_class
        self.model_name = model_class.__name__

    async def get_by_id(self, id: UUID) -> ModelType | None:
        """get entity by id

        Args:
            id: entity UUID

        Returns:
            Model instance or None if not found

        Raises:
            ValidationError: if id format is invalid
        """
        try:
            result = await self.session.get(self.model_class, id)

            if result:
                logger.debug(f"Found {self.model_name} with id={id}")
            else:
                logger.debug(f"{self.model_name} not found with id {id}")

            return result
        except Exception as e:
            logger.error(f"Error getting {self.model_name} by id {id}: {e}")
            raise ValidationError("id", f"Invalid UUID format: {e}")

    async def get_by_id_or_raise(self, id: UUID) -> ModelType:
        """Get entity by ID or raise NotFoundError if not found

        Args:
            id: Entity UUID

        Returns:
            Model instance

        Raises:
            NotFoundError: If entity not found
            ValidationError: If id format is invalid
        """
        result = await self.get_by_id(id)
        if result is None:
            raise NotFoundError(self.model_name, id)
        return result

    async def exists(self, id: UUID) -> bool:
        """check if entity exists by id

        Args:
            id: entity UUID

        Returns:
            True if entity exists, False otherwise
        """
        result = await self.get_by_id(id)
        return result is not None

    async def count(self) -> int:
        """Count total entity of this type

        Returns:
            total count
        """

        stmt = select(func.count()).select_from(self.model_class)
        result = await self.session.execute(stmt)
        count = result.scalar()

        logger.debug(f"Total {self.model_name} count: {count}")
        return count or 0

    async def refresh(self, instance: ModelType) -> ModelType:
        """Refresh the instance from database

        Useful after commit to get updated timestamps, etc

        Args:
            instance: Model instance to refresh

        Returns:
            Refreshed Model instance
        """
        await self.session.refresh(instance)
        return instance

    async def delete(self, instance: ModelType) -> bool:
        """Delete an entity instance

        Args:
            instance: Model instance to delete

        Returns:
            True if deleted successfully
        """
        try:
            await self.session.delete(instance)
            return True
        except Exception as e:
            logger.error(f"Error deleting {self.model_name}: {e}")
            raise

    async def delete_by_id(self, id: UUID) -> bool:
        """Delete entity by ID

        Args:
            id: Entity UUID

        Returns:
            True if deleted successfully

        Raises:
            NotFoundError: If entity not found
        """
        instance = await self.get_by_id_or_raise(id)
        return await self.delete(instance)

    def _log_operation(self, operation: str, details: str = "") -> None:
        """Log repository operation

        Args:
            operation: operation name (e.g., "create", "update", "delete")
            details: additional details to log
        """
        msg = f"{self.model_name}.{operation}"
        if details:
            msg += f" - {details}"
        logger.info(msg)
