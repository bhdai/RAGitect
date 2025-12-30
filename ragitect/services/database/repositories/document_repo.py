"""Document repository for CRUD operations"""

import base64
from datetime import UTC, datetime
import hashlib
import logging
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.sql.functions import func

from ragitect.services.database.exceptions import DuplicateError
from ragitect.services.database.models import Document, DocumentChunk
from ragitect.services.database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository[Document]):
    """Repository for document operations

    Handles all the database operations for document entities including:
    - Creating documents with metadata
    - Adding chunks to documents
    - Retrieving document by workspace
    - Duplicate detection via content hashing
    - batch operations

    Usage:
    >>>> async with get_session() as session:
    ::::    repo = DocumentRepository(session)
    ::::    document = await repo.create(
    ::::        workspace_id=workspace.id,
    ::::        file_name="paper.pdf",
    ::::        processed_content="...",
    ::::        embeddings=[0.1] * 768
    ::::    )
    """

    def __init__(self, session: AsyncSession):
        """Initialize DocumentRepository

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(session, Document)

    async def create(
        self,
        workspace_id: UUID,
        file_name: str,
        processed_content: str | None = None,
        content_hash: str | None = None,
        unique_identifier_hash: str | None = None,
        file_type: str | None = None,
        summary: str | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """create a new document

        Args:
            workspace_id: parent workspace id
            file_name: original file name
            processed_content: full text after ETL processing (optional for upload)
            content_hash: content hash for duplicate detection
            unique_identifier_hash: unique identifier hash
            file_type: file type / extension
            summary: short summary of the document
            embedding: vector embedding of the document
            metadata: additional metadata as key-value pairs

        Returns:
            Document: created document instance

        Raises:
            DuplicateError: if a document with the same unique identifier hash already exists
        """
        if content_hash is None and processed_content is not None:
            content_hash = hashlib.sha256(processed_content.encode()).hexdigest()

        if unique_identifier_hash is None:
            unique_identifier_hash = hashlib.sha256(
                f"{workspace_id}:{content_hash}".encode()
            ).hexdigest()

        try:
            document = Document(
                workspace_id=workspace_id,
                file_name=file_name,
                file_type=file_type,
                content_hash=content_hash,
                unique_identifier_hash=unique_identifier_hash,
                processed_content=processed_content,
                summary=summary,
                embedding=embedding,
                metadata_=metadata or {},
            )

            self.session.add(document)
            await self.session.flush()
            await self.session.refresh(document)

            self._log_operation(
                "create", f"workspace_id={workspace_id}, file_name={file_name}"
            )
            return document
        except IntegrityError:
            await self.session.rollback()
            logger.warning(
                f"Duplicate document: {file_name} in workspace {workspace_id}"
            )
            raise DuplicateError(
                "Document", "unique_identifier_hash", unique_identifier_hash
            )

    async def get_by_workspace(
        self,
        workspace_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Document]:
        """Get all documents in a workspace

        Args:
            workspace_id: Workspace UUID
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return

        Returns:
            List of Document instances
        """
        stmt = (
            select(Document)
            .where(Document.workspace_id == workspace_id)
            .order_by(Document.processed_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        documents = result.scalars().all()

        logger.debug(
            f"Retrieved {len(documents)} documents from workspace {workspace_id}"
        )
        return list(documents)

    async def check_duplicate(
        self, workspace_id: UUID, file_name: str, content_hash: str
    ) -> tuple[bool, Document | None]:
        """Check if document is duplicate

        Args:
            workspace_id: workspace UUID
            file_name: file name to check
            content_hash: content hash to check

        Returns:
            tuple: (is_duplicate: bool, document: Document | None)
        """
        stmt = select(Document).where(
            Document.workspace_id == workspace_id, Document.content_hash == content_hash
        )
        result = await self.session.execute(stmt)
        document = result.scalar_one_or_none()

        is_duplicate = document is not None

        if is_duplicate:
            logger.info(f"Duplicate detected: {file_name} in workspace {workspace_id}")

        return is_duplicate, document

    async def add_chunks(
        self,
        document_id: UUID,
        chunks: list[tuple[str, list[float], dict[str, Any] | None]],
    ) -> list[DocumentChunk]:
        """add chunks to a document

        Args:
            document_id: parent document UUID
            chunks: list of (content, embedding, metadata) tuples

        Returns:
            List of created DocumentChunk instances

        Raises:
            NotFoundError: if the parent document does not exist
        """
        document = await self.get_by_id_or_raise(document_id)
        workspace_id = document.workspace_id

        chunk_objects = []
        for idx, (content, embedding, metadata) in enumerate(chunks):
            chunk = DocumentChunk(
                document_id=document_id,
                workspace_id=workspace_id,
                chunk_index=idx,
                content=content,
                embedding=embedding,
                metadata_=metadata or {},
            )
            chunk_objects.append(chunk)

        self.session.add_all(chunk_objects)
        await self.session.flush()

        for chunk in chunk_objects:
            await self.session.refresh(chunk)

        logger.info(f"Added {len(chunk_objects)} chunks to document {document_id}")
        return chunk_objects

    async def get_chunks(self, document_id: UUID) -> list[DocumentChunk]:
        """Get all chunks for a document

        Args:
            document_id: document UUID

        Returns:
            List of DocumentChunk instances

        Raises:
            NotFoundError: if the document does not exist
        """
        _ = await self.get_by_id_or_raise(document_id)

        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        logger.debug(f"Retrieved {len(chunks)} chunks for document {document_id}")
        return list(chunks)

    async def count_chunks(self, document_id: UUID) -> int:
        """count chunks in a document

        Args:
            document_id: Document UUID

        Returns:
            number of chunks in the document

        Raises:
            NotFoundError: if the document does not exist
        """
        _ = await self.get_by_id_or_raise(document_id)

        stmt = (
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
        )
        result = await self.session.execute(stmt)
        count = result.scalar()

        return count or 0

    async def update_embedding(
        self, document_id: UUID, embedding: list[float]
    ) -> Document:
        """update document embedding

        Args:
            document_id: document UUID
            embedding: new embedding vector

        Returns:
            updated Document instance

        Raises:
            NotFoundError: if the document does not exist
        """
        document = await self.get_by_id_or_raise(document_id)
        document.embedding = embedding

        await self.session.flush()
        await self.session.refresh(document)

        self._log_operation("update_embedding", f"document_id={document_id}")
        return document

    async def update_metadata(
        self, document_id: UUID, metadata: dict[str, Any]
    ) -> Document:
        """Update document metadata.

        Args:
            document_id: Document UUID
            metadata: New metadata dictionary

        Returns:
            Updated Document instance

        Raises:
            NotFoundError: If document doesn't exist
        """
        document = await self.get_by_id_or_raise(document_id)
        document.metadata_ = metadata

        await self.session.flush()
        await self.session.refresh(document)

        self._log_operation("update_metadata", f"document_id={document_id}")
        return document

    async def get_by_workspace_count(self, workspace_id: UUID) -> int:
        """Count documents in workspace.

        Args:
            workspace_id: Workspace UUID

        Returns:
            Number of documents
        """
        stmt = (
            select(func.count())
            .select_from(Document)
            .where(Document.workspace_id == workspace_id)
        )
        result = await self.session.execute(stmt)
        count = result.scalar()

        return count or 0

    async def create_from_upload(
        self,
        workspace_id: UUID,
        file_name: str,
        file_type: str,
        file_bytes: bytes,
    ) -> Document:
        """Create document from file upload (before processing)

        Args:
            workspace_id: Parent workspace UUID
            file_name: Original file name
            file_type: File extension/type
            file_bytes: Raw file bytes

        Returns:
            Document: Created document instance with status="uploaded"

        Raises:
            DuplicateError: If document with same unique identifier exists
        """
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        unique_hash = hashlib.sha256(
            f"{workspace_id}:{file_name}:{datetime.now(UTC).isoformat()}".encode()
        ).hexdigest()

        # Store base64-encoded bytes for later processing
        file_bytes_b64 = base64.b64encode(file_bytes).decode("utf-8")

        return await self.create(
            workspace_id=workspace_id,
            file_name=file_name,
            file_type=file_type,
            content_hash=content_hash,
            unique_identifier_hash=unique_hash,
            processed_content=None,  # Not processed yet
            metadata={
                "status": "uploaded",
                "original_size": len(file_bytes),
                "file_bytes_b64": file_bytes_b64,  # Store for processing
            },
        )

    async def update_status(self, document_id: UUID, status: str) -> Document:
        """Update document processing status

        Args:
            document_id: Document UUID
            status: New status (uploaded, processing, ready, error)

        Returns:
            Updated Document instance

        Raises:
            NotFoundError: If document doesn't exist
        """
        document = await self.get_by_id_or_raise(document_id)
        # Create a NEW dict to ensure SQLAlchemy detects the change
        # (in-place mutation of JSONB dicts may not be tracked)
        metadata = dict(document.metadata_) if document.metadata_ else {}
        metadata["status"] = status
        return await self.update_metadata(document_id, metadata)

    async def update_processed_content(
        self, document_id: UUID, content: str
    ) -> Document:
        """Update document with processed text content

        This method will be called by  document processor
        after text extraction.

        Args:
            document_id: Document UUID
            content: Extracted text content

        Returns:
            Updated Document instance

        Raises:
            NotFoundError: If document doesn't exist
        """
        document = await self.get_by_id_or_raise(document_id)
        document.processed_content = content

        await self.session.flush()
        await self.session.refresh(document)

        self._log_operation("update_processed_content", f"document_id={document_id}")
        return document

    async def get_file_bytes(self, document_id: UUID) -> bytes:
        """Get file bytes from metadata for processing

        Args:
            document_id: Document UUID

        Returns:
            Raw file bytes decoded from base64

        Raises:
            NotFoundError: If document doesn't exist
            ValueError: If file_bytes_b64 not found in metadata
        """
        document = await self.get_by_id_or_raise(document_id)
        metadata = document.metadata_ or {}

        if "file_bytes_b64" not in metadata:
            raise ValueError(
                f"Document {document_id} has no stored file bytes in metadata"
            )

        file_bytes_b64 = metadata["file_bytes_b64"]
        return base64.b64decode(file_bytes_b64)

    async def clear_file_bytes(self, document_id: UUID) -> Document:
        """Clear file bytes from metadata after successful processing

        This frees storage by removing the base64-encoded bytes.
        Note: Re-fetches document to get current metadata state to avoid
        overwriting status changes made by other methods.

        Args:
            document_id: Document UUID

        Returns:
            Updated Document instance

        Raises:
            NotFoundError: If document doesn't exist
        """
        # Re-fetch to get current metadata state (status may have changed)
        document = await self.get_by_id_or_raise(document_id)
        # Make a copy of current metadata to avoid mutation issues
        metadata = dict(document.metadata_) if document.metadata_ else {}

        # Remove file bytes if present
        if "file_bytes_b64" in metadata:
            del metadata["file_bytes_b64"]
            await self.update_metadata(document_id, metadata)
            logger.info(f"Cleared file bytes for document {document_id}")

        return document

    async def create_from_url(
        self,
        workspace_id: UUID,
        source_url: str,
        source_type: str,
    ) -> Document:
        """Create document placeholder from URL submission (Story 5.1)

        Creates a document record with status="backlog" to indicate
        the URL has been submitted but not yet fetched/processed.
        Actual URL fetching happens in Story 5.5 background tasks.

        Args:
            workspace_id: Parent workspace UUID
            source_url: The submitted URL (validated by caller)
            source_type: Type of URL source ("url", "youtube", "pdf")

        Returns:
            Document: Created document instance with status="backlog"

        Raises:
            DuplicateError: If document with same URL already exists in workspace
        """
        # Generate file name from URL (use last path segment or domain)
        parsed = urlparse(source_url)
        path_segments = [s for s in parsed.path.split("/") if s]
        if path_segments:
            file_name = path_segments[-1]
        else:
            file_name = parsed.netloc

        # Add source type prefix for clarity
        if source_type == "youtube":
            # Prefer YouTube video id when available
            video_id = parse_qs(parsed.query or "").get("v", [""])[0]
            if video_id:
                file_name = f"[YouTube] {video_id}"
            else:
                file_name = f"[YouTube] {file_name}"
        elif source_type == "pdf":
            if not file_name.lower().endswith(".pdf"):
                file_name = f"{file_name}.pdf"

        # Hash based on URL for duplicate detection
        # NOTE: unique_identifier_hash must be deterministic so re-submitting the same
        # URL in the same workspace triggers the DB uniqueness constraint.
        content_hash = hashlib.sha256(source_url.encode()).hexdigest()
        unique_hash = hashlib.sha256(
            f"url:{workspace_id}:{source_type}:{source_url}".encode()
        ).hexdigest()

        # Determine file type from source type
        file_type_map = {
            "url": "html",
            "youtube": "youtube",
            "pdf": "pdf",
        }
        file_type = file_type_map.get(source_type, "unknown")

        return await self.create(
            workspace_id=workspace_id,
            file_name=file_name,
            file_type=file_type,
            content_hash=content_hash,
            unique_identifier_hash=unique_hash,
            processed_content=None,  # Not fetched yet
            metadata={
                "status": "backlog",  # Queued for fetching
                "source_type": source_type,
                "source_url": source_url,
                "submitted_at": datetime.now(UTC).isoformat(),
            },
        )
