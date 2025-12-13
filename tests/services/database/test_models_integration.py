"""Integration tests for database ORM models.

Requirements:
    - Running PostgreSQL database (ragitect_test)
    - DATABASE_URL environment variable set
"""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ragitect.services.database import get_session
from ragitect.services.database.models import Document, DocumentChunk, Workspace

# Module-level markers
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestWorkspaceModelIntegration:
    """Integration tests for Workspace model (requires database)"""

    async def test_create_workspace(self, clean_database):
        """Test creating a workspace in database"""
        async with get_session() as session:
            workspace = Workspace(name="Test Workspace", description="A test workspace")
            session.add(workspace)
            await session.flush()

            assert workspace.id is not None
            assert isinstance(workspace.id, uuid.UUID)
            assert workspace.name == "Test Workspace"
            assert workspace.description == "A test workspace"
            assert isinstance(workspace.created_at, datetime)
            assert isinstance(workspace.updated_at, datetime)

    async def test_workspace_name_unique_constraint(self, clean_database):
        """Test workspace name unique constraint is enforced"""
        # Create first workspace
        async with get_session() as session:
            workspace1 = Workspace(name="Duplicate Name")
            session.add(workspace1)

        # Try to create second workspace with same name
        with pytest.raises(IntegrityError) as exc_info:
            async with get_session() as session:
                workspace2 = Workspace(name="Duplicate Name")
                session.add(workspace2)
                await session.flush()

        assert (
            "unique" in str(exc_info.value).lower()
            or "duplicate" in str(exc_info.value).lower()
        )

    async def test_workspace_name_not_empty_constraint(self, clean_database):
        """Test workspace name cannot be empty"""
        with pytest.raises(IntegrityError) as exc_info:
            async with get_session() as session:
                workspace = Workspace(name="")
                session.add(workspace)
                await session.flush()

        assert (
            "check" in str(exc_info.value).lower()
            or "constraint" in str(exc_info.value).lower()
        )


class TestDocumentModelIntegration:
    """Integration tests for Document model (requires database)"""

    async def test_create_document(self, clean_database):
        """Test creating a document in database"""
        # Create workspace first
        async with get_session() as session:
            workspace = Workspace(name="Doc Test Workspace")
            session.add(workspace)
            await session.flush()
            workspace_id = workspace.id

        # Create document
        async with get_session() as session:
            test_embedding = [0.1] * 768  # 768-dimensional vector
            document = Document(
                workspace_id=workspace_id,
                file_name="test.pdf",
                file_type="pdf",
                content_hash="abc123",
                unique_identifier_hash="unique123",
                processed_content="Test content",
                summary="Test summary",
                embedding=test_embedding,
                metadata_={"page_count": 10},
            )
            session.add(document)
            await session.flush()

            assert document.id is not None
            assert isinstance(document.id, uuid.UUID)
            assert document.workspace_id == workspace_id
            assert document.file_name == "test.pdf"
            assert document.file_type == "pdf"
            assert document.metadata_ == {"page_count": 10}

    async def test_document_cascade_delete_from_workspace(self, clean_database):
        """Test that deleting workspace cascades to documents"""
        # Create workspace and document
        async with get_session() as session:
            workspace = Workspace(name="Cascade Test Workspace")
            session.add(workspace)
            await session.flush()
            workspace_id = workspace.id

            test_embedding = [0.1] * 768
            document = Document(
                workspace_id=workspace_id,
                file_name="cascade_test.pdf",
                content_hash="cascade_hash",
                unique_identifier_hash="cascade_unique",
                embedding=test_embedding,
            )
            session.add(document)
            await session.flush()
            document_id = document.id

        # Delete workspace
        async with get_session() as session:
            workspace = await session.get(Workspace, workspace_id)
            await session.delete(workspace)

        # Verify document is also deleted
        async with get_session() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM documents WHERE id = :id"),
                {"id": str(document_id)},
            )
            count = result.scalar()
            assert count == 0, "Document should be cascade deleted"


class TestDocumentChunkModelIntegration:
    """Integration tests for DocumentChunk model (requires database)"""

    async def test_create_document_chunk(self, clean_database):
        """Test creating a document chunk in database"""
        # Create workspace and document
        async with get_session() as session:
            workspace = Workspace(name="Chunk Test Workspace")
            session.add(workspace)
            await session.flush()
            workspace_id = workspace.id

            doc_embedding = [0.1] * 768
            document = Document(
                workspace_id=workspace_id,
                file_name="chunk_test.pdf",
                content_hash="chunk_hash",
                unique_identifier_hash="chunk_unique",
                embedding=doc_embedding,
            )
            session.add(document)
            await session.flush()
            document_id = document.id

        # Create chunk
        async with get_session() as session:
            chunk_embedding = [0.2] * 768
            chunk = DocumentChunk(
                document_id=document_id,
                workspace_id=workspace_id,
                chunk_index=0,
                content="This is chunk content",
                embedding=chunk_embedding,
                metadata_={"page": 1},
            )
            session.add(chunk)
            await session.flush()

            assert chunk.id is not None
            assert isinstance(chunk.id, uuid.UUID)
            assert chunk.document_id == document_id
            assert chunk.workspace_id == workspace_id
            assert chunk.chunk_index == 0
            assert chunk.content == "This is chunk content"
            assert chunk.metadata_ == {"page": 1}

    async def test_document_chunk_unique_constraint(self, clean_database):
        """Test document_id + chunk_index unique constraint"""
        # Create workspace and document
        async with get_session() as session:
            workspace = Workspace(name="Unique Chunk Test")
            session.add(workspace)
            await session.flush()
            workspace_id = workspace.id

            doc_embedding = [0.1] * 768
            document = Document(
                workspace_id=workspace_id,
                file_name="unique_chunk.pdf",
                content_hash="unique_chunk_hash",
                unique_identifier_hash="unique_chunk_id",
                embedding=doc_embedding,
            )
            session.add(document)
            await session.flush()
            document_id = document.id

        # Create first chunk
        async with get_session() as session:
            chunk_embedding = [0.2] * 768
            chunk1 = DocumentChunk(
                document_id=document_id,
                workspace_id=workspace_id,
                chunk_index=0,
                content="First chunk",
                embedding=chunk_embedding,
            )
            session.add(chunk1)

        # Try to create duplicate chunk with same index
        with pytest.raises(IntegrityError) as exc_info:
            async with get_session() as session:
                chunk2 = DocumentChunk(
                    document_id=document_id,
                    workspace_id=workspace_id,
                    chunk_index=0,  # Same index
                    content="Second chunk",
                    embedding=chunk_embedding,
                )
                session.add(chunk2)
                await session.flush()

        assert (
            "unique" in str(exc_info.value).lower()
            or "duplicate" in str(exc_info.value).lower()
        )

    async def test_document_chunk_negative_index_constraint(self, clean_database):
        """Test chunk_index cannot be negative"""
        # Create workspace and document
        async with get_session() as session:
            workspace = Workspace(name="Negative Index Test")
            session.add(workspace)
            await session.flush()
            workspace_id = workspace.id

            doc_embedding = [0.1] * 768
            document = Document(
                workspace_id=workspace_id,
                file_name="negative_test.pdf",
                content_hash="negative_hash",
                unique_identifier_hash="negative_unique",
                embedding=doc_embedding,
            )
            session.add(document)
            await session.flush()
            document_id = document.id

        # Try to create chunk with negative index
        with pytest.raises(IntegrityError) as exc_info:
            async with get_session() as session:
                chunk_embedding = [0.2] * 768
                chunk = DocumentChunk(
                    document_id=document_id,
                    workspace_id=workspace_id,
                    chunk_index=-1,  # Negative index
                    content="Negative chunk",
                    embedding=chunk_embedding,
                )
                session.add(chunk)
                await session.flush()

        assert (
            "check" in str(exc_info.value).lower()
            or "constraint" in str(exc_info.value).lower()
        )

    async def test_document_chunk_empty_content_constraint(self, clean_database):
        """Test chunk content cannot be empty"""
        # Create workspace and document
        async with get_session() as session:
            workspace = Workspace(name="Empty Content Test")
            session.add(workspace)
            await session.flush()
            workspace_id = workspace.id

            doc_embedding = [0.1] * 768
            document = Document(
                workspace_id=workspace_id,
                file_name="empty_content.pdf",
                content_hash="empty_hash",
                unique_identifier_hash="empty_unique",
                embedding=doc_embedding,
            )
            session.add(document)
            await session.flush()
            document_id = document.id

        # Try to create chunk with empty content
        with pytest.raises(IntegrityError) as exc_info:
            async with get_session() as session:
                chunk_embedding = [0.2] * 768
                chunk = DocumentChunk(
                    document_id=document_id,
                    workspace_id=workspace_id,
                    chunk_index=0,
                    content="",  # Empty content
                    embedding=chunk_embedding,
                )
                session.add(chunk)
                await session.flush()

        assert (
            "check" in str(exc_info.value).lower()
            or "constraint" in str(exc_info.value).lower()
        )

    async def test_document_chunk_cascade_delete(self, clean_database):
        """Test that deleting document cascades to chunks"""
        # Create workspace, document, and chunk
        async with get_session() as session:
            workspace = Workspace(name="Cascade Chunk Test")
            session.add(workspace)
            await session.flush()
            workspace_id = workspace.id

            doc_embedding = [0.1] * 768
            document = Document(
                workspace_id=workspace_id,
                file_name="cascade_chunk.pdf",
                content_hash="cascade_chunk_hash",
                unique_identifier_hash="cascade_chunk_unique",
                embedding=doc_embedding,
            )
            session.add(document)
            await session.flush()
            document_id = document.id

            chunk_embedding = [0.2] * 768
            chunk = DocumentChunk(
                document_id=document_id,
                workspace_id=workspace_id,
                chunk_index=0,
                content="Cascade chunk",
                embedding=chunk_embedding,
            )
            session.add(chunk)
            await session.flush()
            chunk_id = chunk.id

        # Delete document
        async with get_session() as session:
            document = await session.get(Document, document_id)
            await session.delete(document)

        # Verify chunk is also deleted
        async with get_session() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM document_chunks WHERE id = :id"),
                {"id": str(chunk_id)},
            )
            count = result.scalar()
            assert count == 0, "Chunk should be cascade deleted"
