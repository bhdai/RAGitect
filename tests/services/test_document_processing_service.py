"""Unit tests for DocumentProcessingService

Tests document processing workflow including status transitions,
text extraction integration, and error handling.
"""

import base64
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, MagicMock, patch

from ragitect.services.database.models import Document
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.document_processing_service import DocumentProcessingService


@pytest.fixture
def mock_document_repo(mocker):
    """Mock DocumentRepository for testing"""
    repo = mocker.Mock(spec=DocumentRepository)
    repo.get_by_id_or_raise = mocker.AsyncMock()
    repo.update_status = mocker.AsyncMock()
    repo.get_file_bytes = mocker.AsyncMock()
    repo.update_processed_content = mocker.AsyncMock()
    repo.clear_file_bytes = mocker.AsyncMock()
    repo.add_chunks = mocker.AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_session(mocker):
    """Mock AsyncSession for testing"""
    session = mocker.Mock(spec=AsyncSession)
    session.commit = mocker.AsyncMock()
    session.rollback = mocker.AsyncMock()
    return session


@pytest.fixture
def document_processing_service(mock_session, mock_document_repo, mocker):
    """Create DocumentProcessingService with mocked dependencies"""
    service = DocumentProcessingService(mock_session)
    # Replace repo with mock
    mocker.patch.object(service, "repo", mock_document_repo)
    return service


@pytest.fixture
def sample_document():
    """Sample document with uploaded status"""
    doc_id = uuid4()
    file_bytes = b"Sample PDF content"
    file_bytes_b64 = base64.b64encode(file_bytes).decode("utf-8")

    return Document(
        id=doc_id,
        workspace_id=uuid4(),
        file_name="test.pdf",
        file_type=".pdf",
        content_hash="abc123",
        unique_identifier_hash="unique123",
        processed_content=None,
        metadata_={
            "status": "uploaded",
            "original_size": len(file_bytes),
            "file_bytes_b64": file_bytes_b64,
        },
    )


class TestDocumentProcessingService:
    """Test suite for DocumentProcessingService"""

    @pytest.mark.asyncio
    async def test_process_document_success(
        self,
        document_processing_service,
        mock_document_repo,
        mock_session,
        sample_document,
    ):
        """Test successful document processing flow including embedding generation"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_document
        mock_document_repo.get_file_bytes.return_value = b"Sample PDF content"
        mock_document_repo.add_chunks = AsyncMock(return_value=[])

        # Mock process_file_bytes to return extracted text
        with patch(
            "ragitect.services.document_processing_service.process_file_bytes"
        ) as mock_process:
            mock_process.return_value = (
                "Extracted text content",
                {"file_type": ".pdf", "file_name": "test.pdf"},
            )

            # Mock embedding functions
            with patch(
                "ragitect.services.document_processing_service.split_document",
                return_value=["Chunk 1"],
            ):
                mock_model = MagicMock()
                mock_model.aembed_documents = AsyncMock(return_value=[[0.1] * 768])
                with patch(
                    "ragitect.services.document_processing_service.create_embeddings_model",
                    return_value=mock_model,
                ):
                    with patch(
                        "ragitect.services.document_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768]),
                    ):
                        # Act
                        await document_processing_service.process_document(
                            sample_document.id
                        )

        # Assert - verify status transitions (now 3: processing, embedding, ready)
        assert mock_document_repo.update_status.call_count == 3
        mock_document_repo.update_status.assert_any_call(
            sample_document.id, "processing"
        )
        mock_document_repo.update_status.assert_any_call(
            sample_document.id, "embedding"
        )
        mock_document_repo.update_status.assert_any_call(sample_document.id, "ready")

        # Assert - verify text extraction called
        mock_process.assert_called_once_with(b"Sample PDF content", "test.pdf")

        # Assert - verify processed content stored
        mock_document_repo.update_processed_content.assert_called_once_with(
            sample_document.id, "Extracted text content"
        )

        # Assert - verify chunks stored
        mock_document_repo.add_chunks.assert_called_once()

        # Assert - verify file bytes cleared
        mock_document_repo.clear_file_bytes.assert_called_once_with(sample_document.id)

        # Assert - verify session commits (now 4: processing, processed_content, embedding, ready)
        assert mock_session.commit.call_count == 4

    @pytest.mark.asyncio
    async def test_process_document_not_uploaded_status(
        self, document_processing_service, mock_document_repo, sample_document
    ):
        """Test processing skipped if document not in 'uploaded' state"""
        # Arrange - set status to 'processing' instead of 'uploaded'
        sample_document.metadata_["status"] = "processing"
        mock_document_repo.get_by_id_or_raise.return_value = sample_document

        # Act
        await document_processing_service.process_document(sample_document.id)

        # Assert - no processing should occur
        assert mock_document_repo.update_status.call_count == 0
        assert mock_document_repo.get_file_bytes.call_count == 0

    @pytest.mark.asyncio
    async def test_process_document_error_handling(
        self,
        document_processing_service,
        mock_document_repo,
        mock_session,
        sample_document,
    ):
        """Test error handling and status update to 'error'"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_document
        mock_document_repo.get_file_bytes.side_effect = ValueError("File bytes missing")

        # Act & Assert - expect exception to be re-raised
        with pytest.raises(ValueError, match="File bytes missing"):
            await document_processing_service.process_document(sample_document.id)

        # Assert - verify status updated to processing, then error
        assert mock_document_repo.update_status.call_count >= 2
        mock_document_repo.update_status.assert_any_call(
            sample_document.id, "processing"
        )
        mock_document_repo.update_status.assert_any_call(sample_document.id, "error")

    @pytest.mark.asyncio
    async def test_process_document_extraction_failure(
        self,
        document_processing_service,
        mock_document_repo,
        mock_session,
        sample_document,
    ):
        """Test handling of text extraction failures"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_document
        mock_document_repo.get_file_bytes.return_value = b"Corrupted content"

        # Mock process_file_bytes to raise exception
        with patch(
            "ragitect.services.document_processing_service.process_file_bytes"
        ) as mock_process:
            mock_process.side_effect = Exception("Extraction failed")

            # Act & Assert
            with pytest.raises(Exception, match="Extraction failed"):
                await document_processing_service.process_document(sample_document.id)

        # Assert - status should be updated to error
        mock_document_repo.update_status.assert_any_call(sample_document.id, "error")

    @pytest.mark.asyncio
    async def test_process_document_no_metadata(
        self, document_processing_service, mock_document_repo, sample_document
    ):
        """Test processing skipped if document has no metadata"""
        # Arrange
        sample_document.metadata_ = None
        mock_document_repo.get_by_id_or_raise.return_value = sample_document

        # Act
        await document_processing_service.process_document(sample_document.id)

        # Assert - no processing should occur (no status)
        assert mock_document_repo.update_status.call_count == 0

    @pytest.mark.asyncio
    async def test_process_document_integration_with_repo_methods(
        self,
        document_processing_service,
        mock_document_repo,
        mock_session,
        sample_document,
    ):
        """Test that all repository methods are called correctly"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_document
        mock_document_repo.get_file_bytes.return_value = b"Sample content"

        with patch(
            "ragitect.services.document_processing_service.process_file_bytes"
        ) as mock_process:
            mock_process.return_value = ("Extracted text", {"file_type": ".pdf"})

            # Mock embedding functions
            with patch(
                "ragitect.services.document_processing_service.split_document",
                return_value=["Chunk"],
            ):
                with patch(
                    "ragitect.services.document_processing_service.create_embeddings_model"
                ):
                    with patch(
                        "ragitect.services.document_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768]),
                    ):
                        # Act
                        await document_processing_service.process_document(
                            sample_document.id
                        )

        # Assert - verify call sequence
        mock_document_repo.get_by_id_or_raise.assert_called_once_with(
            sample_document.id
        )
        mock_document_repo.get_file_bytes.assert_called_once_with(sample_document.id)
        mock_document_repo.update_processed_content.assert_called_once()
        mock_document_repo.clear_file_bytes.assert_called_once()
