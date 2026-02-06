"""URL Processing Service

Handles background URL document processing including URL fetching,
content extraction, embedding generation, and status management.

Implements AC2, AC3 from Story 5.5:
- Background task function for URL-based documents
- Status tracking in document metadata

Flow:
    1. Update status to "fetching"
    2. Get appropriate processor from ProcessorFactory (url/youtube/pdf)
    3. Fetch URL and convert to Markdown
    4. Store processed content in DB
    5. Update status to "processing"
    6. Split text into chunks
    7. Update status to "embedding"
    8. Generate embeddings for all chunks
    9. Store chunks with embeddings
    10. Update status to "ready" (or "error" on failure)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.services.config import load_document_config
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.document_processor import split_document
from ragitect.services.embedding import embed_documents, get_embedding_model_from_config
from ragitect.services.processor.factory import ProcessorFactory
from ragitect.services.retry import with_retry

logger = logging.getLogger(__name__)

# Concurrency limit for URL fetches (NFR-P5, AC5)
# Will be implemented in Task 8
_url_fetch_semaphore = asyncio.Semaphore(5)


class URLProcessingService:
    """Service for background URL document processing

    Orchestrates the URL processing workflow:
    1. Update status to "fetching" with fetch_started_at timestamp
    2. Get appropriate processor from ProcessorFactory
    3. Fetch URL and convert to Markdown via processor
    4. Store processed content in DB
    5. Update status to "processing" with fetch_completed_at timestamp
    6. Split text into chunks
    7. Update status to "embedding"
    8. Generate embeddings for all chunks
    9. Store chunks with embeddings via add_chunks()
    10. Update status to "ready" (or "error" on failure)

    This mirrors DocumentProcessingService but handles URL-based sources
    instead of file uploads.

    Usage:
        >>> async with get_session() as session:
        ...     service = URLProcessingService(session)
        ...     await service.process_url_document(document_id, url, "url")
    """

    def __init__(self, session: AsyncSession):
        """Initialize URLProcessingService

        Args:
            session: SQLAlchemy async session
        """
        self.session = session
        self.repo = DocumentRepository(session)
        self.processor_factory = ProcessorFactory()

    async def process_url_document(
        self,
        document_id: UUID,
        url: str,
        source_type: Literal["url", "youtube", "pdf"],
    ) -> None:
        """Process URL document: fetch, extract, chunk, embed (AC2)

        Args:
            document_id: Document UUID to process
            url: URL to fetch content from
            source_type: Type of URL source ("url", "youtube", "pdf")

        Note:
            This method catches all exceptions internally and updates
            document status to "error" with error message in metadata.
            Exceptions are logged but not re-raised to avoid crashing
            the background task.
        """
        async with _url_fetch_semaphore:
            try:
                # Update status to fetching with timestamp (AC3)
                await self._update_status_with_metadata(
                    document_id,
                    "fetching",
                    {"fetch_started_at": datetime.now(timezone.utc).isoformat()},
                )
                await self.session.commit()

                # Get appropriate processor from factory
                processor = self.processor_factory.get_processor(url, source_type)

                # Fetch and convert to Markdown with retry logic (AC4)
                markdown = await with_retry(processor.process, url)

                logger.info(
                    f"Fetched {len(markdown)} chars from {url[:50]}..."
                )

                # Update status to processing with fetch_completed_at (AC3)
                await self._update_status_with_metadata(
                    document_id,
                    "processing",
                    {"fetch_completed_at": datetime.now(timezone.utc).isoformat()},
                )
                await self.session.commit()

                # Store processed content in document
                await self.repo.update_processed_content(document_id, markdown)
                await self.session.commit()

                # Chunking and embedding (reuse pattern from DocumentProcessingService)
                await self._chunk_and_embed(document_id, markdown)

                # Final status update
                await self.repo.update_status(document_id, "ready")
                await self.session.commit()

                logger.info(f"Successfully processed URL document {document_id}")

            except Exception as e:
                logger.error(
                    f"URL processing failed for {document_id}: {e}",
                    exc_info=True,
                )
                await self._set_error_status(document_id, str(e))

    async def _update_status_with_metadata(
        self,
        document_id: UUID,
        status: str,
        extra_metadata: dict[str, Any],
    ) -> None:
        """Update status and merge additional metadata (AC3)

        Args:
            document_id: Document UUID
            status: New status value
            extra_metadata: Additional metadata fields to merge
        """
        document = await self.repo.get_by_id_or_raise(document_id)
        metadata = dict(document.metadata_) if document.metadata_ else {}
        metadata["status"] = status
        metadata.update(extra_metadata)
        await self.repo.update_metadata(document_id, metadata)

    async def _chunk_and_embed(self, document_id: UUID, text: str) -> None:
        """Split text and generate embeddings

        Args:
            document_id: Document UUID
            text: Processed text content to chunk and embed
        """
        # Update status to embedding
        await self.repo.update_status(document_id, "embedding")
        await self.session.commit()

        # Load document configuration for chunk size/overlap
        doc_config = load_document_config()

        # Split text into chunks using Markdown-aware splitter
        chunks = split_document(
            text,
            chunk_size=doc_config.chunk_size,
            overlap=doc_config.chunk_overlap,
        )

        logger.info(f"Split into {len(chunks)} chunks for document {document_id}")

        if chunks:
            # Get embedding model from config
            embedding_model = await get_embedding_model_from_config(self.session)

            # Generate embeddings for all chunks
            embeddings = await embed_documents(embedding_model, chunks)

            # Prepare chunk data: (content, embedding, metadata)
            chunk_data: list[tuple[str, list[float], dict[str, Any] | None]] = [
                (chunk, embedding, {"chunk_index": i})
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]

            # Store chunks with embeddings
            await self.repo.add_chunks(document_id, chunk_data)
            logger.info(f"Stored {len(chunk_data)} chunks for document {document_id}")
        else:
            logger.warning(
                f"No chunks generated for URL document {document_id} (empty content)"
            )

    async def _set_error_status(
        self, document_id: UUID, error_message: str
    ) -> None:
        """Set error status with user-friendly message (AC6)

        Args:
            document_id: Document UUID
            error_message: Raw error message to humanize
        """
        try:
            document = await self.repo.get_by_id_or_raise(document_id)
            metadata = dict(document.metadata_) if document.metadata_ else {}
            metadata["status"] = "error"
            metadata["error_message"] = self._humanize_error(error_message)
            await self.repo.update_metadata(document_id, metadata)
            await self.session.commit()
        except Exception as update_error:
            logger.error(f"Failed to set error status: {update_error}")

    def _humanize_error(self, error_message: str) -> str:
        """Convert technical error to user-friendly message (AC6)

        Args:
            error_message: Raw technical error message

        Returns:
            User-friendly error message (< 500 chars)
        """
        error_lower = error_message.lower()

        # Timeout errors
        if "timeout" in error_lower:
            return "Request timed out (30s limit). The server may be slow or unavailable."

        # HTTP status errors
        if "404" in error_lower:
            return "Page not found (404). Please verify the URL is correct."

        if "403" in error_lower:
            return "Access denied (403). The content may be restricted."

        if "401" in error_lower:
            return "Authentication required (401). The content may be private."

        if "500" in error_lower or "502" in error_lower or "503" in error_lower:
            return "Server error. The website may be temporarily unavailable."

        # Connection errors
        if "connect" in error_lower:
            return "Could not connect to the URL. Please check the address and try again."

        # YouTube-specific errors
        if "transcript" in error_lower and "disabled" in error_lower:
            return "Transcripts are disabled for this YouTube video."

        if "transcript" in error_lower:
            return "Could not retrieve transcript for this YouTube video."

        # PDF-specific errors
        if "pdf" in error_lower and "invalid" in error_lower:
            return "URL does not point to a valid PDF file."

        if "pdf" in error_lower and "download" in error_lower:
            return "Failed to download PDF. Please check the URL."

        if "pdf" in error_lower and "process" in error_lower:
            return "Failed to process PDF content. The file may be corrupted or password-protected."

        # Content extraction errors
        if "extract" in error_lower or "content" in error_lower:
            return "Could not extract content from the web page."

        # Generic fallback - truncate to keep user-friendly
        truncated = error_message[:200]
        if len(error_message) > 200:
            truncated += "..."

        return f"Processing failed: {truncated}"
