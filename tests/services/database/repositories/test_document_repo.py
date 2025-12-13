"""Unit tests for DocumentRepository."""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.models import Document, DocumentChunk
from ragitect.services.database.exceptions import NotFoundError, DuplicateError
from sqlalchemy.exc import IntegrityError

pytestmark = [pytest.mark.asyncio]


class TestDocumentRepository:
    @pytest.fixture
    def repo(self, mock_session):
        return DocumentRepository(mock_session)

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

    async def test_create_duplicate_document(self, repo, mock_session):
        mock_session.flush.side_effect = IntegrityError(
            None, None, Exception("Duplicate")
        )

        with pytest.raises(DuplicateError):
            await repo.create(uuid4(), "duplicate.pdf", "content")

        mock_session.rollback.assert_called_once()

    async def test_get_by_workspace(self, repo, mock_session):
        docs = [Document(file_name="doc1"), Document(file_name="doc2")]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs
        mock_session.execute.return_value = mock_result

        result = await repo.get_by_workspace(uuid4())

        assert len(result) == 2
        assert result == docs

    async def test_check_duplicate(self, repo, mock_session):
        doc = Document(file_name="exists.pdf")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_session.execute.return_value = mock_result

        is_dup, found_doc = await repo.check_duplicate(uuid4(), "exists.pdf", "hash")

        assert is_dup is True
        assert found_doc == doc

    async def test_check_duplicate_false(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        is_dup, found_doc = await repo.check_duplicate(uuid4(), "new.pdf", "hash")

        assert is_dup is False
        assert found_doc is None

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

    async def test_add_chunks_document_not_found(self, repo, mock_session):
        mock_session.get.return_value = None

        with pytest.raises(NotFoundError):
            await repo.add_chunks(uuid4(), [])

    async def test_get_chunks(self, repo, mock_session):
        document_id = uuid4()
        mock_session.get.return_value = Document(id=document_id)

        chunks = [DocumentChunk(chunk_index=0), DocumentChunk(chunk_index=1)]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = chunks
        mock_session.execute.return_value = mock_result

        result = await repo.get_chunks(document_id)

        assert result == chunks

    async def test_count_chunks(self, repo, mock_session):
        document_id = uuid4()
        mock_session.get.return_value = Document(id=document_id)

        mock_result = MagicMock()
        mock_result.scalar.return_value = 10
        mock_session.execute.return_value = mock_result

        count = await repo.count_chunks(document_id)
        assert count == 10

    async def test_update_embedding(self, repo, mock_session):
        document_id = uuid4()
        document = Document(id=document_id)
        mock_session.get.return_value = document

        new_embedding = [0.5] * 768
        updated = await repo.update_embedding(document_id, new_embedding)

        assert updated.embedding == new_embedding
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    async def test_update_metadata(self, repo, mock_session):
        document_id = uuid4()
        document = Document(id=document_id, metadata_={})
        mock_session.get.return_value = document

        new_metadata = {"processed": True}
        updated = await repo.update_metadata(document_id, new_metadata)

        assert updated.metadata_ == new_metadata
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    async def test_get_by_workspace_count(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        count = await repo.get_by_workspace_count(uuid4())
        assert count == 5

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

    async def test_update_status(self, repo, mock_session):
        """Test update_status method"""
        document_id = uuid4()
        document = Document(id=document_id, metadata_={"status": "uploaded"})
        mock_session.get.return_value = document

        updated = await repo.update_status(document_id, "processing")

        assert updated.metadata_["status"] == "processing"
        mock_session.flush.assert_called_once()

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
