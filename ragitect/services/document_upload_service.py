"""Document Upload Service

Handles document upload operations including file validation and storage.
Separated from document_processor.py which handles text extraction
"""

import logging
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.services.database.models import Document
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.exceptions import FileSizeExceededError
from ragitect.services.processor.factory import ProcessorFactory

logger = logging.getLogger(__name__)

# Maximum file size: 50MB
# NOTE: Frontend also validates this limit (frontend/src/components/IngestionDropzone.tsx)
# Both layers enforce independently for better UX (client-side) and security (server-side)
MAX_FILE_SIZE = 50 * 1024 * 1024


class DocumentUploadService:
    """Service for uploading documents to workspaces

    This service handles the upload phase only:
    - File type validation using ProcessorFactory
    - Raw file storage
    - Document metadata creation with status="uploaded"
    """

    def __init__(self, session: AsyncSession):
        """Initialize the upload service

        Args:
            session: SQLAlchemy async session
        """
        self.session = session
        self.repository = DocumentRepository(session)
        self.processor_factory = ProcessorFactory()

    async def upload_document(self, workspace_id: UUID, file: UploadFile) -> Document:
        """Upload a single document to workspace

        Args:
            workspace_id: Target workspace UUID
            file: Uploaded file from FastAPI

        Returns:
            Document: Created document with status="uploaded"

        Raises:
            UnsupportedFormatError: If file format is not supported
            FileSizeExceededError: If file size exceeds limit
        """
        logger.info(f"Uploading document {file.filename} to workspace {workspace_id}")

        # Validate file format using ProcessorFactory
        # This raises UnsupportedFormatError if format not supported
        _ = self.processor_factory.get_processor(file.filename or "unknown")

        # Read file bytes with size validation
        file_bytes = await file.read()

        # Validate file size after reading
        if len(file_bytes) > MAX_FILE_SIZE:
            file_size_mb = len(file_bytes) / (1024 * 1024)
            max_size_mb = MAX_FILE_SIZE / (1024 * 1024)
            raise FileSizeExceededError(
                filename=file.filename or "unknown",
                file_size_mb=file_size_mb,
                max_size_mb=max_size_mb,
            )

        # Extract file type (extension)
        file_type = Path(file.filename or "").suffix.lower()

        # Create document record with status="uploaded"
        document = await self.repository.create_from_upload(
            workspace_id=workspace_id,
            file_name=file.filename or "unknown",
            file_type=file_type,
            file_bytes=file_bytes,
        )

        logger.info(
            f"Successfully uploaded document {document.id}: {file.filename} "
            f"({len(file_bytes)} bytes)"
        )

        return document

    async def upload_documents(
        self, workspace_id: UUID, files: list[UploadFile]
    ) -> list[Document]:
        """Upload multiple documents to workspace

        Args:
            workspace_id: Target workspace UUID
            files: List of uploaded files

        Returns:
            List of created documents

        Raises:
            UnsupportedFormatError: If any file format is not supported
            FileSizeExceededError: If any file size exceeds limit

        Note:
            Database operations must be sequential due to SQLAlchemy session
            limitations. File validation can be done concurrently.
        """
        logger.info(f"Uploading {len(files)} documents to workspace {workspace_id}")

        # Process uploads sequentially to avoid SQLAlchemy session conflicts
        # asyncio.gather causes "Session is already flushing" errors
        documents = []
        for file in files:
            document = await self.upload_document(workspace_id, file)
            documents.append(document)

        logger.info(
            f"Successfully uploaded {len(documents)} documents to workspace {workspace_id}"
        )

        return documents
