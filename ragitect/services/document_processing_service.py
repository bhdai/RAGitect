"""Document Processing Service

Handles background document processing including text extraction and status management.
"""

import asyncio
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.document_processor import process_file_bytes

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """Service for background document processing

    Orchestrates the document processing workflow:
    1. Fetch document from DB
    2. Update status to "processing"
    3. Get file bytes from metadata
    4. Extract text using process_file_bytes()
    5. Store processed content in DB
    6. Update status to "ready" (or "error" on failure)
    7. Clear file bytes from metadata

    Usage:
        >>> async with get_session() as session:
        ...     service = DocumentProcessingService(session)
        ...     await service.process_document(document_id)
    """

    def __init__(self, session: AsyncSession):
        """Initialize DocumentProcessingService

        Args:
            session: SQLAlchemy async session
        """
        self.session = session
        self.repo = DocumentRepository(session)

    async def process_document(self, document_id: UUID) -> None:
        """Process document: extract text and update status

        Args:
            document_id: Document UUID to process

        Raises:
            NotFoundError: If document doesn't exist
            ValueError: If document not in 'uploaded' state or file bytes missing
            Exception: Any processing errors (status updated to 'error')
        """
        try:
            # Fetch document
            document = await self.repo.get_by_id_or_raise(document_id)

            # Guard: Only process if status is "uploaded"
            current_status = (
                document.metadata_.get("status") if document.metadata_ else None
            )
            if current_status != "uploaded":
                logger.warning(
                    f"Document {document_id} not in 'uploaded' state: {current_status}"
                )
                return

            logger.info(
                f"Starting processing for document {document_id}: {document.file_name}"
            )

            # Update status to processing
            await self.repo.update_status(document_id, "processing")
            await self.session.commit()

            # Get file bytes from metadata
            file_bytes = await self.repo.get_file_bytes(document_id)

            # Extract text using existing processor
            # Run in thread pool to avoid blocking event loop (process_file_bytes is CPU-intensive)
            text, metadata = await asyncio.to_thread(
                process_file_bytes, file_bytes, document.file_name
            )

            logger.info(
                f"Extracted {len(text)} characters from {document.file_name} "
                f"(file_type: {metadata.get('file_type')})"
            )

            # Store processed content
            await self.repo.update_processed_content(document_id, text)

            # Update status to ready
            await self.repo.update_status(document_id, "ready")

            # Clear file bytes to free storage
            await self.repo.clear_file_bytes(document_id)

            await self.session.commit()

            logger.info(f"Successfully processed document {document_id}")

        except Exception as e:
            logger.error(
                f"Processing failed for document {document_id}: {e}", exc_info=True
            )

            # Update status to error
            try:
                await self.repo.update_status(document_id, "error")
                await self.session.commit()
            except Exception as status_error:
                logger.error(f"Failed to update status to error: {status_error}")

            # Re-raise original exception
            raise
