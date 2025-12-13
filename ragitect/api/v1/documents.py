"""Document API endpoints

Provides REST API endpoints for document operations:
- POST /api/v1/workspaces/{workspace_id}/documents - Upload documents
- GET /api/v1/workspaces/{workspace_id}/documents - List documents
- GET /api/v1/documents/{document_id} - Get document detail
- DELETE /api/v1/documents/{document_id} - Delete document
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
    DocumentDetailResponse,
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

# Router for workspace-scoped document endpoints
router = APIRouter(prefix="/workspaces", tags=["documents"])

# Router for document-specific endpoints (without /workspaces prefix)
documents_router = APIRouter(prefix="/documents", tags=["documents"])


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
        DocumentStatusResponse with current status and phase

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

        # Determine current phase for progress indication
        phase: str | None = None
        if doc_status == "processing":
            phase = "parsing"
        elif doc_status == "embedding":
            phase = "embedding"
        # ready, error, uploaded have no active phase

        return DocumentStatusResponse(
            id=document.id,
            status=doc_status,
            file_name=document.file_name,
            phase=phase,
        )

    except NotFoundError as e:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        ) from e


# ============================================================================
# List Documents Endpoint (workspace-scoped)
# ============================================================================


@router.get(
    "/{workspace_id}/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List documents in workspace",
    description="Retrieve all documents belonging to a workspace with pagination",
)
async def list_workspace_documents(
    workspace_id: UUID,
    skip: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(get_async_session),
) -> DocumentListResponse:
    """List documents in a workspace

    Args:
        workspace_id: Target workspace UUID
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
        session: Database session (injected by FastAPI)

    Returns:
        DocumentListResponse with list of documents and total count

    Raises:
        HTTPException 404: If workspace not found
    """
    logger.info(f"Listing documents for workspace {workspace_id}")

    # Verify workspace exists
    workspace_repo = WorkspaceRepository(session)
    try:
        await workspace_repo.get_by_id_or_raise(workspace_id)
    except NotFoundError as e:
        logger.warning(f"Workspace not found: {workspace_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace not found: {workspace_id}",
        ) from e

    # Get documents
    document_repo = DocumentRepository(session)
    documents = await document_repo.get_by_workspace(
        workspace_id=workspace_id,
        skip=skip,
        limit=limit,
    )
    total_count = await document_repo.get_by_workspace_count(workspace_id)

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

    logger.info(f"Found {len(documents)} documents in workspace {workspace_id}")

    return DocumentListResponse(
        documents=document_responses,
        total=total_count,
    )


# ============================================================================
# Document-specific endpoints (documents_router)
# ============================================================================


@documents_router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document with content",
    description="Retrieve a document with its full processed content for viewing",
)
async def get_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_async_session),
) -> DocumentDetailResponse:
    """Get document detail with processed content

    Args:
        document_id: Document UUID
        session: Database session (injected by FastAPI)

    Returns:
        DocumentDetailResponse with full document info including processed_content

    Raises:
        HTTPException 404: If document not found
    """
    logger.info(f"Fetching document detail: {document_id}")

    document_repo = DocumentRepository(session)
    try:
        document = await document_repo.get_by_id_or_raise(document_id)

        # Extract status from metadata
        doc_status = (
            document.metadata_.get("status", "uploaded")
            if document.metadata_
            else "uploaded"
        )

        return DocumentDetailResponse(
            id=document.id,
            file_name=document.file_name,
            file_type=document.file_type,
            status=doc_status,
            processed_content=document.processed_content,
            summary=document.summary,
            created_at=document.processed_at,
        )

    except NotFoundError as e:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        ) from e


@documents_router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Delete a document and all its associated chunks and embeddings (cascade)",
)
async def delete_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Delete a document

    Cascade delete is handled automatically by SQLAlchemy relationships.
    All document chunks and embeddings are removed with the document.

    Args:
        document_id: Document UUID
        session: Database session (injected by FastAPI)

    Raises:
        HTTPException 404: If document not found
    """
    logger.info(f"Deleting document: {document_id}")

    document_repo = DocumentRepository(session)
    try:
        deleted = await document_repo.delete_by_id(document_id)
        if deleted:
            await session.commit()
            logger.info(f"Successfully deleted document {document_id}")

    except NotFoundError as e:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        ) from e
