import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload

from ragitect.services.database.exceptions import DuplicateError, NotFoundError
from ragitect.services.database.models import Workspace
from ragitect.services.database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class WorkspaceRepository(BaseRepository[Workspace]):
    """Repository for workspace operations

    Handles all database operations for Workspace entities including:
    - Creating workspace
    - Retrieving workspace by id or name
    - Listing all workspaces
    - Updating workspace details
    - Deleting workspace (cascade deletes documents and chunks)

    Usage:
    >>>> async with get_session() as session:
    ::::    repo = WorkspaceRepository(session)
    ::::    workspace = await repo.create("AI research", "ML papers")
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, Workspace)

    async def create(self, name: str, description: str | None = None) -> Workspace:
        """Create a new workspac

        Args:
            name: Workspace name (must be unique)
            description: Optional workspace description

        Returns:
            Created Workspace instance

        Raises:
            DuplicateError: If workspace with same name exisit
        """
        try:
            workspace = Workspace(name=name, description=description)

            self.session.add(workspace)
            await self.session.flush()  # flush to catch IntegrityError
            await self.session.refresh(Workspace)

            self._log_operation("creaet", f"name={name}")
            return workspace
        except IntegrityError:
            await self.session.rollback()
            logger.warning(f"Duplicate workspace name: {name}")
            raise DuplicateError("Workspace", "name", name)

    async def get_by_name(self, name: str) -> Workspace | None:
        """Get workspace by name

        Args:
            name: Workspace name

        Returns:
            Workspace instance or None if not found
        """
        stmt = select(Workspace).where(Workspace.name == name)
        result = await self.session.execute(stmt)
        workspace = result.scalar_one_or_none()

        if workspace:
            logger.debug(f"Found Workspace with name='{name}'")
        else:
            logger.debug(f"Workspace not found with name '{name}'")

        return workspace

    async def get_by_name_or_raise(self, name: str) -> Workspace:
        """Get workspace by name or raise NotFoundError

        Args:
            name: Workspace name

        Returns:
            Workspace instance

        Raises:
            NotFoundError: If workspace not found
        """
        from ragitect.services.database.exceptions import NotFoundError

        workspace = await self.get_by_name(name)
        if workspace is None:
            raise NotFoundError("Workspace", name)
        return workspace

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Workspace]:
        """Get all the workspace with pagination

        Args:
            skip: number of workspace to skip (for pagination)
            limit: Maximum number of workspace to return

        Returns:
            list of Workspace instances
        """
        stmt = (
            select(Workspace)
            .order_by(Workspace.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        workspaces = result.scalars().all()

        logger.debug(
            f"Retrieved {len(workspaces)} workspaces (skip={skip}, limit={limit})"
        )
        return list(workspaces)

    async def update(
        self,
        workspace_id: UUID,
        name: str | None = None,
        description: str | None = None,
    ) -> Workspace:
        """update workspace details

        Args:
            workspace_id: workspace UUID
            name: new name (optional)
            description: new description (optional)

        Returns:
            updated workspace instance

        Raises:
            NotFoundError: if workspace not found
            DuplicateError: if new name violates unique constraint
        """
        workspace = await self.get_by_id_or_raise(workspace_id)

        try:
            if name is not None:
                workspace.name = name
            if description is not None:
                workspace.description = description

            await self.session.flush()  # flush to catch IntegrityError
            await self.session.refresh(workspace)

            self._log_operation("update", f"id={workspace_id}")
            return workspace

        except IntegrityError:
            await self.session.rollback()
            raise DuplicateError("Workspace", "name", "name")

    async def exists_by_name(self, name: str) -> bool:
        """Check if workspace with given name exists

        Args:
            name: workspace name

        Returns:
            True if exists, False otherwise
        """
        workspace = await self.get_by_name(name)
        return workspace is not None

    async def count_document(self, workspace_id: UUID) -> int:
        """Count documents in a workspace

        Args:
            workspace_id: Workspace UUID

        Returns:
            Number of documents in workspace

        Raises:
            NotFoundError: if workspace not found
        """
        from ragitect.services.database.models import Document

        _ = await self.get_by_id_or_raise(workspace_id)  # verify workspace exists

        stmt = (
            select(func.count())
            .select_from(Document)
            .where(Document.workspace_id == workspace_id)
        )
        result = await self.session.execute(stmt)
        count = result.scalar()

        logger.debug(f"Workspace {workspace_id} has {count} documents")
        return count or 0

    async def get_with_documents(self, workspace_id: UUID) -> Workspace:
        """get workspace with documents eagerly loaded

        Args:
            workspace_id: workspace UUID

        Returns:
            workspace instance with document loadeded

        Raises:
            NotFoundError: if workspace not found
        """
        stmt = (
            select(Workspace)
            .where(Workspace.id == workspace_id)
            .options(selectinload(Workspace.documents))
        )
        result = await self.session.execute(stmt)
        workspace = result.scalar_one_or_none()

        if workspace is None:
            raise NotFoundError("Workspace", workspace_id)

        logger.debug(
            f"Loaded workspace {workspace_id} with {len(workspace.documents)} documents"
        )

        return workspace
