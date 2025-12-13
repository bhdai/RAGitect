"""Unit tests for DocumentUploadService"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import UploadFile

from ragitect.services.document_upload_service import (
    DocumentUploadService,
    MAX_FILE_SIZE,
)
from ragitect.services.exceptions import FileSizeExceededError
from ragitect.services.processor.factory import UnsupportedFormatError


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session"""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_upload_file():
    """Mock FastAPI UploadFile"""
    file = MagicMock(spec=UploadFile)
    file.filename = "test.pdf"
    file.read = AsyncMock(return_value=b"test file content")
    return file


@pytest.mark.asyncio
async def test_upload_document_success(mock_session, mock_upload_file):
    """Test successful document upload"""
    service = DocumentUploadService(mock_session)
    workspace_id = uuid4()

    # Mock ProcessorFactory to accept the file
    with patch.object(service.processor_factory, "get_processor") as mock_get_proc:
        mock_processor = MagicMock()
        mock_get_proc.return_value = mock_processor

        # Mock repository create_from_upload
        with patch.object(service.repository, "create_from_upload") as mock_create:
            from ragitect.services.database.models import Document

            expected_doc = Document(
                id=uuid4(),
                workspace_id=workspace_id,
                file_name="test.pdf",
                file_type=".pdf",
                processed_content=None,
                metadata_={"status": "uploaded", "original_size": 17},
            )
            mock_create.return_value = expected_doc

            result = await service.upload_document(workspace_id, mock_upload_file)

            assert result == expected_doc
            mock_get_proc.assert_called_once_with("test.pdf")
            mock_upload_file.read.assert_called_once()
            mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_upload_document_unsupported_format(mock_session):
    """Test upload with unsupported format"""
    service = DocumentUploadService(mock_session)
    workspace_id = uuid4()

    unsupported_file = MagicMock(spec=UploadFile)
    unsupported_file.filename = "test.exe"

    # ProcessorFactory will raise UnsupportedFormatError
    with patch.object(service.processor_factory, "get_processor") as mock_get_proc:
        mock_get_proc.side_effect = UnsupportedFormatError("Unsupported format: .exe")

        with pytest.raises(UnsupportedFormatError):
            await service.upload_document(workspace_id, unsupported_file)


@pytest.mark.asyncio
async def test_upload_documents_multiple(mock_session):
    """Test uploading multiple documents"""
    service = DocumentUploadService(mock_session)
    workspace_id = uuid4()

    # Create multiple mock files
    files = []
    for i in range(3):
        file = MagicMock(spec=UploadFile)
        file.filename = f"test{i}.pdf"
        file.read = AsyncMock(return_value=b"content")
        files.append(file)

    with patch.object(service.processor_factory, "get_processor"):
        with patch.object(service.repository, "create_from_upload") as mock_create:
            from ragitect.services.database.models import Document

            # Return different docs for each call
            mock_create.side_effect = [
                Document(id=uuid4(), file_name=f"test{i}.pdf") for i in range(3)
            ]

            results = await service.upload_documents(workspace_id, files)

            assert len(results) == 3
            assert mock_create.call_count == 3


@pytest.mark.asyncio
async def test_upload_document_file_type_extraction(mock_session):
    """Test file type extraction from filename"""
    service = DocumentUploadService(mock_session)
    workspace_id = uuid4()

    file = MagicMock(spec=UploadFile)
    file.filename = "document.DOCX"  # uppercase extension
    file.read = AsyncMock(return_value=b"content")

    with patch.object(service.processor_factory, "get_processor"):
        with patch.object(service.repository, "create_from_upload") as mock_create:
            from ragitect.services.database.models import Document

            mock_create.return_value = Document(id=uuid4())

            await service.upload_document(workspace_id, file)

            # Verify file_type was lowercased
            call_args = mock_create.call_args
            assert call_args.kwargs["file_type"] == ".docx"


@pytest.mark.asyncio
async def test_upload_document_exceeds_size_limit(mock_session):
    """Test upload with file exceeding size limit"""
    service = DocumentUploadService(mock_session)
    workspace_id = uuid4()

    # Create file that exceeds 50MB limit
    large_file = MagicMock(spec=UploadFile)
    large_file.filename = "large.pdf"
    large_file.read = AsyncMock(return_value=b"x" * (MAX_FILE_SIZE + 1))

    with patch.object(service.processor_factory, "get_processor"):
        with pytest.raises(FileSizeExceededError) as exc_info:
            await service.upload_document(workspace_id, large_file)

        assert exc_info.value.filename == "large.pdf"
        assert exc_info.value.max_size_mb == 50.0
        assert "too large" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_upload_document_at_size_limit(mock_session):
    """Test upload with file at exactly the size limit"""
    service = DocumentUploadService(mock_session)
    workspace_id = uuid4()

    # Create file at exactly 50MB
    at_limit_file = MagicMock(spec=UploadFile)
    at_limit_file.filename = "atlimit.pdf"
    at_limit_file.read = AsyncMock(return_value=b"x" * MAX_FILE_SIZE)

    with patch.object(service.processor_factory, "get_processor"):
        with patch.object(service.repository, "create_from_upload") as mock_create:
            from ragitect.services.database.models import Document

            mock_create.return_value = Document(id=uuid4())

            # Should succeed at exactly the limit
            result = await service.upload_document(workspace_id, at_limit_file)

            assert result is not None
            mock_create.assert_called_once()
