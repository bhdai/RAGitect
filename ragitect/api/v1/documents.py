"""Document API endpoints

Provides REST API endpoints for document operations:
- POST /api/v1/workspaces/{workspace_id}/documents - Upload documents
- GET /api/v1/workspaces/{workspace_id}/documents - List documents (future)
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.api.schemas.document import DocumentListResponse, DocumentUploadResponse
from ragitect.services.database.connection import get_async_session
from ragitect.services.database.exceptions import NotFoundError
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.document_upload_service import DocumentUploadService
from ragitect.services.processor.factory import UnsupportedFormatError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces", tags=["documents"])


@router.post(
    "/{workspace_id}/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload documents to workspace",
    description="Upload one or more files to a workspace. Supports PDF, DOCX, TXT, MD, and other formats.",
)
async def upload_documents(
    workspace_id: UUID,
    files: list[UploadFile] = File(...),
    session: AsyncSession = Depends(get_async_session),
) -> DocumentListResponse:
    """Upload documents to workspace

    Args:
        workspace_id: Target workspace UUID
        files: List of uploaded files (multipart/form-data)
        session: Database session (injected by FastAPI)

    Returns:
        DocumentListResponse with uploaded documents and total count

    Raises:
        HTTPException 404: If workspace not found
        HTTPException 400: If file format not supported
    """
    logger.info(f"Uploading {len(files)} files to workspace {workspace_id}")

    # Validate workspace exists
    workspace_repo = WorkspaceRepository(session)
    try:
        _ = await workspace_repo.get_by_id_or_raise(workspace_id)
    except NotFoundError as e:
        logger.warning(f"Workspace not found: {workspace_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace not found: {workspace_id}",
        ) from e

    # Upload documents
    upload_service = DocumentUploadService(session)

    try:
        documents = await upload_service.upload_documents(workspace_id, files)

        # Convert to response format
        document_responses = [
            DocumentUploadResponse(
                id=doc.id,
                file_name=doc.file_name,
                file_type=doc.file_type,
                status=doc.metadata_.get("status", "uploaded")
                if doc.metadata_
                else "uploaded",
                created_at=doc.processed_at,
            )
            for doc in documents
        ]

        logger.info(
            f"Successfully uploaded {len(documents)} documents to workspace {workspace_id}"
        )

        # Commit transaction
        await session.commit()

        return DocumentListResponse(
            documents=document_responses,
            total=len(document_responses),
        )

    except UnsupportedFormatError as e:
        logger.warning(f"Unsupported file format: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
