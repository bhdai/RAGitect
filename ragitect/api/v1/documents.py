"""Document API endpoints

Provides REST API endpoints for document operations:
- POST /api/v1/workspaces/{workspace_id}/documents - Upload documents
- GET /api/v1/workspaces/{workspace_id}/documents - List documents (future)
"""

import logging
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.api.schemas.document import (
    DocumentListResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from ragitect.services.database.connection import get_async_session
from ragitect.services.database.exceptions import NotFoundError
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.document_processing_service import DocumentProcessingService
from ragitect.services.document_upload_service import DocumentUploadService
from ragitect.services.exceptions import FileSizeExceededError
from ragitect.services.processor.factory import UnsupportedFormatError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces", tags=["documents"])


async def process_document_background(document_id: UUID) -> None:
    """Background task wrapper for document processing

    Args:
        document_id: Document UUID to process
    """
    from ragitect.services.database.connection import get_session_factory

    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            processing_service = DocumentProcessingService(session)
            await processing_service.process_document(document_id)
        except Exception as e:
            logger.error(f"Background processing failed for {document_id}: {e}")
            # Error already logged and status updated in service


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
    background_tasks: BackgroundTasks = BackgroundTasks(),
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

        # Trigger background processing for each document
        for doc in documents:
            background_tasks.add_task(
                process_document_background,
                document_id=doc.id,
            )
            logger.info(f"Scheduled background processing for document {doc.id}")

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

    except FileSizeExceededError as e:
        logger.warning(f"File size exceeded: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        ) from e


@router.get(
    "/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document processing status",
    description="Retrieve the current processing status of a document",
)
async def get_document_status(
    document_id: UUID,
    session: AsyncSession = Depends(get_async_session),
) -> DocumentStatusResponse:
    """Get document processing status

    Args:
        document_id: Document UUID
        session: Database session (injected by FastAPI)

    Returns:
        DocumentStatusResponse with current status

    Raises:
        HTTPException 404: If document not found
    """
    logger.info(f"Fetching status for document {document_id}")

    document_repo = DocumentRepository(session)
    try:
        document = await document_repo.get_by_id_or_raise(document_id)

        # Extract status from metadata
        doc_status = (
            document.metadata_.get("status", "uploaded")
            if document.metadata_
            else "uploaded"
        )

        return DocumentStatusResponse(
            id=document.id,
            status=doc_status,
            file_name=document.file_name,
        )

    except NotFoundError as e:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        ) from e
