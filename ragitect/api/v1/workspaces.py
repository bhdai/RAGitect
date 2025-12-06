"""Workspace API endpoints

Provides REST API endpoints for workspace CRUD operations:
- POST /api/v1/workspaces - Create a new workspace
- GET /api/v1/workspaces - List all workspaces
- GET /api/v1/workspaces/{id} - Get a single workspace by ID
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.api.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceListResponse,
    WorkspaceResponse,
)
from ragitect.services.database.connection import get_session
from ragitect.services.database.exceptions import DuplicateError, NotFoundError
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new workspace",
    description="Create a new workspace with a unique name and optional description.",
)
async def create_workspace(
    data: WorkspaceCreate,
    session: AsyncSession = Depends(get_session),
) -> WorkspaceResponse:
    """Create a new workspace.

    Args:
        data: WorkspaceCreate schema with name and optional description
        session: Database session (injected by FastAPI)

    Returns:
        WorkspaceResponse with created workspace data (camelCase serialization)

    Raises:
        HTTPException 409: If workspace with same name already exists
    """
    logger.info(f"Creating workspace: {data.name}")

    repo = WorkspaceRepository(session)

    try:
        workspace = await repo.create(
            name=data.name,
            description=data.description,
        )
        logger.info(f"Created workspace: {workspace.id}")

        return WorkspaceResponse.model_validate(workspace)

    except DuplicateError as e:
        logger.warning(f"Duplicate workspace name: {data.name}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Workspace with name '{data.name}' already exists",
        ) from e


@router.get(
    "",
    response_model=WorkspaceListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all workspaces",
    description="Retrieve a list of all workspaces with pagination support.",
)
async def list_workspaces(
    skip: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
) -> WorkspaceListResponse:
    """List all workspaces with pagination.

    Args:
        skip: Number of workspaces to skip (for pagination)
        limit: Maximum number of workspaces to return
        session: Database session (injected by FastAPI)

    Returns:
        WorkspaceListResponse with list of workspaces and total count
    """
    logger.info(f"Listing workspaces: skip={skip}, limit={limit}")

    repo = WorkspaceRepository(session)

    workspaces = await repo.get_all(skip=skip, limit=limit)
    total = await repo.count()

    workspace_responses = [WorkspaceResponse.model_validate(ws) for ws in workspaces]

    logger.info(f"Found {len(workspaces)} workspaces (total: {total})")

    return WorkspaceListResponse(
        workspaces=workspace_responses,
        total=total,
    )


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a workspace by ID",
    description="Retrieve a single workspace by its unique identifier.",
)
async def get_workspace(
    workspace_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> WorkspaceResponse:
    """Get a single workspace by ID.

    Args:
        workspace_id: Workspace UUID
        session: Database session (injected by FastAPI)

    Returns:
        WorkspaceResponse with workspace data (camelCase serialization)

    Raises:
        HTTPException 404: If workspace not found
    """
    logger.info(f"Getting workspace: {workspace_id}")

    repo = WorkspaceRepository(session)

    try:
        workspace = await repo.get_by_id_or_raise(workspace_id)
        logger.info(f"Found workspace: {workspace.name}")

        return WorkspaceResponse.model_validate(workspace)

    except NotFoundError as e:
        logger.warning(f"Workspace not found: {workspace_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace not found: {workspace_id}",
        ) from e
