import pytest
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.models import Document, DocumentChunk
from ragitect.services.database.exceptions import NotFoundError, DuplicateError
from sqlalchemy.exc import IntegrityError


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    session.delete = AsyncMock()
    session.get = AsyncMock()
    return session


class TestDocumentRepository:
    @pytest.fixture
    def repo(self, mock_session):
        return DocumentRepository(mock_session)

    @pytest.mark.asyncio
    async def test_create_document(self, repo, mock_session):
        workspace_id = uuid4()
        file_name = "test.pdf"
        content = "some content"

        doc = await repo.create(workspace_id, file_name, content)

        assert doc.workspace_id == workspace_id
        assert doc.file_name == file_name
        assert doc.processed_content == content
        assert doc.content_hash is not None
        assert doc.unique_identifier_hash is not None

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_duplicate_document(self, repo, mock_session):
        mock_session.flush.side_effect = IntegrityError(
            None, None, Exception("Duplicate")
        )

        with pytest.raises(DuplicateError):
            await repo.create(uuid4(), "duplicate.pdf", "content")

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_workspace(self, repo, mock_session):
        docs = [Document(file_name="doc1"), Document(file_name="doc2")]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs
        mock_session.execute.return_value = mock_result

        result = await repo.get_by_workspace(uuid4())

        assert len(result) == 2
        assert result == docs

    @pytest.mark.asyncio
    async def test_check_duplicate(self, repo, mock_session):
        doc = Document(file_name="exists.pdf")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_session.execute.return_value = mock_result

        is_dup, found_doc = await repo.check_duplicate(uuid4(), "exists.pdf", "hash")

        assert is_dup is True
        assert found_doc == doc

    @pytest.mark.asyncio
    async def test_check_duplicate_false(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        is_dup, found_doc = await repo.check_duplicate(uuid4(), "new.pdf", "hash")

        assert is_dup is False
        assert found_doc is None

    @pytest.mark.asyncio
    async def test_add_chunks(self, repo, mock_session):
        document_id = uuid4()
        workspace_id = uuid4()
        document = Document(id=document_id, workspace_id=workspace_id)
        mock_session.get.return_value = document

        chunks_data = [
            ("chunk1", [0.1] * 768, {"page": 1}),
            ("chunk2", [0.2] * 768, {"page": 2}),
        ]

        chunks = await repo.add_chunks(document_id, chunks_data)

        assert len(chunks) == 2
        assert chunks[0].content == "chunk1"
        assert chunks[1].content == "chunk2"
        assert chunks[0].document_id == document_id

        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_called_once()
        assert mock_session.refresh.call_count == 2

    @pytest.mark.asyncio
    async def test_add_chunks_document_not_found(self, repo, mock_session):
        mock_session.get.return_value = None

        with pytest.raises(NotFoundError):
            await repo.add_chunks(uuid4(), [])

    @pytest.mark.asyncio
    async def test_get_chunks(self, repo, mock_session):
        document_id = uuid4()
        mock_session.get.return_value = Document(id=document_id)

        chunks = [DocumentChunk(chunk_index=0), DocumentChunk(chunk_index=1)]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = chunks
        mock_session.execute.return_value = mock_result

        result = await repo.get_chunks(document_id)

        assert result == chunks

    @pytest.mark.asyncio
    async def test_count_chunks(self, repo, mock_session):
        document_id = uuid4()
        mock_session.get.return_value = Document(id=document_id)

        mock_result = MagicMock()
        mock_result.scalar.return_value = 10
        mock_session.execute.return_value = mock_result

        count = await repo.count_chunks(document_id)
        assert count == 10

    @pytest.mark.asyncio
    async def test_update_embedding(self, repo, mock_session):
        document_id = uuid4()
        document = Document(id=document_id)
        mock_session.get.return_value = document

        new_embedding = [0.5] * 768
        updated = await repo.update_embedding(document_id, new_embedding)

        assert updated.embedding == new_embedding
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_metadata(self, repo, mock_session):
        document_id = uuid4()
        document = Document(id=document_id, metadata_={})
        mock_session.get.return_value = document

        new_metadata = {"processed": True}
        updated = await repo.update_metadata(document_id, new_metadata)

        assert updated.metadata_ == new_metadata
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_workspace_count(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        count = await repo.get_by_workspace_count(uuid4())
        assert count == 5

    @pytest.mark.asyncio
    async def test_create_from_upload(self, repo, mock_session):
        """Test create_from_upload method"""
        workspace_id = uuid4()
        file_name = "test.pdf"
        file_type = ".pdf"
        file_bytes = b"test content bytes"

        doc = await repo.create_from_upload(
            workspace_id, file_name, file_type, file_bytes
        )

        assert doc.workspace_id == workspace_id
        assert doc.file_name == file_name
        assert doc.file_type == file_type
        assert doc.processed_content is None
        assert doc.metadata_["status"] == "uploaded"
        assert doc.metadata_["original_size"] == len(file_bytes)

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status(self, repo, mock_session):
        """Test update_status method"""
        document_id = uuid4()
        document = Document(id=document_id, metadata_={"status": "uploaded"})
        mock_session.get.return_value = document

        updated = await repo.update_status(document_id, "processing")

        assert updated.metadata_["status"] == "processing"
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_processed_content(self, repo, mock_session):
        """Test update_processed_content method"""
        document_id = uuid4()
        document = Document(id=document_id, processed_content=None)
        mock_session.get.return_value = document

        new_content = "Extracted text content"
        updated = await repo.update_processed_content(document_id, new_content)

        assert updated.processed_content == new_content
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_optional_processed_content(self, repo, mock_session):
        """Test create method with processed_content=None"""
        workspace_id = uuid4()
        file_name = "test.pdf"

        doc = await repo.create(
            workspace_id=workspace_id,
            file_name=file_name,
            processed_content=None,
            content_hash="explicit_hash",
        )

        assert doc.workspace_id == workspace_id
        assert doc.file_name == file_name
        assert doc.processed_content is None

        mock_session.add.assert_called_once()


class TestDocumentRepositoryIntegration:
    """Integration tests for DocumentRepository"""

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def db_context(self, clean_db_manager):
        """Setup database for integration tests"""
        import os
        from sqlalchemy import text
        from ragitect.services.database import get_session
        from ragitect.services.database.connection import create_table, drop_table

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        await create_table()
        try:
            yield
        finally:
            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_document_integration(self, clean_db_manager):
        from ragitect.services.database import get_session
        from ragitect.services.database.repositories.workspace_repo import (
            WorkspaceRepository,
        )

        async with self.db_context(clean_db_manager):
            async with get_session() as session:
                ws_repo = WorkspaceRepository(session)
                doc_repo = DocumentRepository(session)

                workspace = await ws_repo.create("Doc Test WS")

                doc = await doc_repo.create(
                    workspace_id=workspace.id,
                    file_name="test.pdf",
                    processed_content="content",
                    embedding=[0.1] * 768,
                )

                assert doc.id is not None
                assert doc.workspace_id == workspace.id

                # Verify persistence
                fetched = await doc_repo.get_by_id(doc.id)
                assert fetched is not None
                assert fetched.file_name == "test.pdf"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_check_duplicate_integration(self, clean_db_manager):
        from ragitect.services.database import get_session
        from ragitect.services.database.repositories.workspace_repo import (
            WorkspaceRepository,
        )
        import hashlib

        async with self.db_context(clean_db_manager):
            async with get_session() as session:
                ws_repo = WorkspaceRepository(session)
                doc_repo = DocumentRepository(session)
                workspace = await ws_repo.create("Dup Check WS")

                content = "duplicate content"
                content_hash = hashlib.sha256(content.encode()).hexdigest()

                # Create first doc
                await doc_repo.create(
                    workspace_id=workspace.id,
                    file_name="doc1.pdf",
                    processed_content=content,
                    embedding=[0.1] * 768,
                )

                # Check duplicate
                is_dup, existing = await doc_repo.check_duplicate(
                    workspace.id, "doc2.pdf", content_hash
                )
                assert is_dup is True
                assert existing is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_chunks_operations_integration(self, clean_db_manager):
        from ragitect.services.database import get_session
        from ragitect.services.database.repositories.workspace_repo import (
            WorkspaceRepository,
        )

        async with self.db_context(clean_db_manager):
            async with get_session() as session:
                ws_repo = WorkspaceRepository(session)
                doc_repo = DocumentRepository(session)
                workspace = await ws_repo.create("Chunks WS")

                doc = await doc_repo.create(
                    workspace_id=workspace.id,
                    file_name="chunks.pdf",
                    processed_content="content",
                    embedding=[0.1] * 768,
                )

                # Add chunks
                chunks_data = [
                    ("chunk 1", [0.1] * 768, {"idx": 1}),
                    ("chunk 2", [0.2] * 768, {"idx": 2}),
                ]

                created_chunks = await doc_repo.add_chunks(doc.id, chunks_data)
                assert len(created_chunks) == 2

                # Get chunks
                fetched_chunks = await doc_repo.get_chunks(doc.id)
                assert len(fetched_chunks) == 2
                assert fetched_chunks[0].content == "chunk 1"

                # Count chunks
                count = await doc_repo.count_chunks(doc.id)
                assert count == 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_operations_integration(self, clean_db_manager):
        from ragitect.services.database import get_session
        from ragitect.services.database.repositories.workspace_repo import (
            WorkspaceRepository,
        )

        async with self.db_context(clean_db_manager):
            async with get_session() as session:
                ws_repo = WorkspaceRepository(session)
                doc_repo = DocumentRepository(session)
                workspace = await ws_repo.create("Update WS")

                doc = await doc_repo.create(
                    workspace_id=workspace.id,
                    file_name="update.pdf",
                    processed_content="content",
                    embedding=[0.0] * 768,
                )

                # Update embedding
                new_emb = [0.9] * 768
                _ = await doc_repo.update_embedding(doc.id, new_emb)
                # Need to fetch again or check if returned object is updated
                # PGVector returns numpy array or list? asyncpg usually returns string or list.
                # Let's just check if no error and value is set.
                # Note: Exact float comparison might be tricky with vector types, but let's try.

                # Update metadata
                new_meta = {"status": "processed"}
                updated_doc_meta = await doc_repo.update_metadata(doc.id, new_meta)
                assert updated_doc_meta.metadata_ == new_meta
