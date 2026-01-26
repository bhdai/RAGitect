"""Document Processing Service

Handles background document processing including text extraction,
embedding generation, and status management.

Flow:
    1. Fetch document from DB
    2. Update status to "processing"
    3. Extract text using docling/processors
    4. Update status to "embedding"
    5. Split text into chunks
    6. Generate embeddings for chunks
    7. Store chunks with embeddings
    8. Update status to "ready"
"""

import asyncio
import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.services.config import load_document_config
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.document_processor import process_file_bytes, split_document
from ragitect.services.embedding import embed_documents, get_embedding_model_from_config

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """Service for background document processing

    Orchestrates the document processing workflow:
    1. Fetch document from DB
    2. Update status to "processing"
    3. Get file bytes from metadata
    4. Extract text using process_file_bytes()
    5. Store processed content in DB
    6. Update status to "embedding"
    7. Split text into chunks
    8. Generate embeddings for all chunks
    9. Store chunks with embeddings via add_chunks()
    10. Update status to "ready" (or "error" on failure)
    11. Clear file bytes from metadata

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
        """Process document: extract text, generate embeddings, and update status

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
                + f"(file_type: {metadata.get('file_type')})"
            )

            # Store processed content
            await self.repo.update_processed_content(document_id, text)
            await self.session.commit()

            # === NEW: Embedding Generation Phase ===

            # Update status to embedding
            await self.repo.update_status(document_id, "embedding")
            await self.session.commit()

            # Load document configuration
            doc_config = load_document_config()

            # Split text into chunks
            # All documents are in Markdown format at this point:
            # - DoclingProcessor converts complex formats (PDF, DOCX, etc.) to Markdown
            # - SimpleProcessor keeps text-based formats (.txt, .md) as-is
            # Therefore, always use Markdown-aware splitting for optimal structure preservation
            chunks = split_document(
                text,
                chunk_size=doc_config.chunk_size,
                overlap=doc_config.chunk_overlap,
            )
            logger.info(f"Split into {len(chunks)} chunks for document {document_id}")

            # Generate embeddings if there are chunks
            if chunks:
                try:
                    embedding_model = await get_embedding_model_from_config(self.session)
                    embeddings = await embed_documents(embedding_model, chunks)
                    logger.info(
                        f"Generated {len(embeddings)} embeddings for document {document_id}"
                    )

                    # Prepare chunk data for storage: (content, embedding, metadata)
                    chunk_data: list[tuple[str, list[float], dict[str, Any] | None]] = [
                        (chunk, embedding, {"chunk_index": i})
                        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                    ]

                    # Store chunks with embeddings
                    await self.repo.add_chunks(document_id, chunk_data)
                    logger.info(
                        f"Stored {len(chunk_data)} chunks for document {document_id}"
                    )

                except Exception as embed_error:
                    logger.error(
                        f"Embedding generation failed for document {document_id}: {embed_error}",
                        exc_info=True,
                    )
                    raise
            else:
                logger.warning(
                    f"No chunks generated for document {document_id} (empty text)"
                )

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
