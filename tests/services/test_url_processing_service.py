"""Unit tests for URLProcessingService

Tests URL document processing workflow including:
- Status transitions (backlog → fetching → processing → embedding → ready)
- Metadata field updates (source_type, source_url, timestamps)
- Integration with ProcessorFactory and DocumentProcessingService patterns
- Error handling and retry logic
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, MagicMock, patch

from ragitect.services.database.models import Document
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.url_processing_service import URLProcessingService


pytestmark = [pytest.mark.asyncio]


@pytest.fixture
def mock_document_repo(mocker):
    """Mock DocumentRepository for testing"""
    repo = mocker.Mock(spec=DocumentRepository)
    repo.get_by_id_or_raise = mocker.AsyncMock()
    repo.update_status = mocker.AsyncMock()
    repo.update_metadata = mocker.AsyncMock()
    repo.update_processed_content = mocker.AsyncMock()
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
def url_processing_service(mock_session, mock_document_repo, mocker):
    """Create URLProcessingService with mocked dependencies"""
    service = URLProcessingService(mock_session)
    # Replace repo with mock
    mocker.patch.object(service, "repo", mock_document_repo)
    return service


@pytest.fixture
def sample_url_document():
    """Sample document in backlog status awaiting URL fetching"""
    doc_id = uuid4()
    return Document(
        id=doc_id,
        workspace_id=uuid4(),
        file_name="[URL] example.com-article",
        file_type="html",
        content_hash="url_hash_123",
        unique_identifier_hash="unique_url_123",
        processed_content=None,
        metadata_={
            "status": "backlog",
            "source_type": "url",
            "source_url": "https://example.com/article",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        },
    )


@pytest.fixture
def sample_youtube_document():
    """Sample YouTube document in backlog status"""
    doc_id = uuid4()
    return Document(
        id=doc_id,
        workspace_id=uuid4(),
        file_name="[YouTube] dQw4w9WgXcQ",
        file_type="youtube",
        content_hash="yt_hash_123",
        unique_identifier_hash="unique_yt_123",
        processed_content=None,
        metadata_={
            "status": "backlog",
            "source_type": "youtube",
            "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        },
    )


@pytest.fixture
def sample_pdf_url_document():
    """Sample PDF URL document in backlog status"""
    doc_id = uuid4()
    return Document(
        id=doc_id,
        workspace_id=uuid4(),
        file_name="paper.pdf",
        file_type="pdf",
        content_hash="pdf_hash_123",
        unique_identifier_hash="unique_pdf_123",
        processed_content=None,
        metadata_={
            "status": "backlog",
            "source_type": "pdf",
            "source_url": "https://arxiv.org/pdf/1706.03762.pdf",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        },
    )


class TestURLProcessingServiceStatusFlow:
    """Test status progression through URL processing lifecycle (AC2, AC3)"""

    async def test_status_updated_to_fetching_at_start(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Status should be updated to 'fetching' at the start of processing"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(return_value="# Test Content\n\nBody")

        with patch.object(
            url_processing_service.processor_factory,
            "get_processor",
            return_value=mock_processor,
        ):
            with patch(
                "ragitect.services.url_processing_service.split_document",
                return_value=["Chunk 1"],
            ):
                with patch(
                    "ragitect.services.url_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=MagicMock()),
                ):
                    with patch(
                        "ragitect.services.url_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768]),
                    ):
                        # Act
                        await url_processing_service.process_url_document(
                            sample_url_document.id,
                            "https://example.com/article",
                            "url",
                        )

        # Assert - verify first metadata update includes fetching status
        update_calls = mock_document_repo.update_metadata.call_args_list
        assert len(update_calls) >= 1

        # First update should set status to "fetching"
        first_metadata = update_calls[0][0][1]
        assert first_metadata.get("status") == "fetching"
        assert "fetch_started_at" in first_metadata

    async def test_status_progresses_through_all_stages(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Status should progress: fetching → processing → embedding → ready"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(return_value="# Markdown Content")

        # Track status updates
        statuses_seen = []

        def track_metadata(doc_id, metadata):
            if "status" in metadata:
                statuses_seen.append(metadata["status"])
            return sample_url_document

        mock_document_repo.update_metadata.side_effect = track_metadata
        mock_document_repo.update_status.side_effect = lambda doc_id, status: (
            statuses_seen.append(status),
            sample_url_document,
        )[1]

        with patch.object(
            url_processing_service.processor_factory,
            "get_processor",
            return_value=mock_processor,
        ):
            with patch(
                "ragitect.services.url_processing_service.split_document",
                return_value=["Chunk"],
            ):
                with patch(
                    "ragitect.services.url_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=MagicMock()),
                ):
                    with patch(
                        "ragitect.services.url_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768]),
                    ):
                        # Act
                        await url_processing_service.process_url_document(
                            sample_url_document.id,
                            "https://example.com/article",
                            "url",
                        )

        # Assert - verify proper status sequence
        assert "fetching" in statuses_seen
        assert "processing" in statuses_seen
        assert "embedding" in statuses_seen
        assert "ready" in statuses_seen

        # Verify order: fetching before processing before embedding before ready
        fetching_idx = statuses_seen.index("fetching")
        processing_idx = statuses_seen.index("processing")
        embedding_idx = statuses_seen.index("embedding")
        ready_idx = statuses_seen.index("ready")

        assert fetching_idx < processing_idx < embedding_idx < ready_idx

    async def test_metadata_includes_fetch_timestamps(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Metadata should include fetch_started_at and fetch_completed_at timestamps"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(return_value="# Content")

        captured_metadata = []
        mock_document_repo.update_metadata.side_effect = lambda doc_id, metadata: (
            captured_metadata.append(metadata.copy()),
            sample_url_document,
        )[1]

        with patch.object(
            url_processing_service.processor_factory,
            "get_processor",
            return_value=mock_processor,
        ):
            with patch(
                "ragitect.services.url_processing_service.split_document",
                return_value=["Chunk"],
            ):
                with patch(
                    "ragitect.services.url_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=MagicMock()),
                ):
                    with patch(
                        "ragitect.services.url_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768]),
                    ):
                        # Act
                        await url_processing_service.process_url_document(
                            sample_url_document.id,
                            "https://example.com/article",
                            "url",
                        )

        # Assert - check timestamps
        all_metadata_keys = set()
        for m in captured_metadata:
            all_metadata_keys.update(m.keys())

        assert "fetch_started_at" in all_metadata_keys
        assert "fetch_completed_at" in all_metadata_keys


class TestURLProcessingServiceProcessorIntegration:
    """Test integration with ProcessorFactory and URL processors (AC2)"""

    async def test_uses_correct_processor_for_web_url(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Should use WebURLProcessor for source_type='url'"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document

        with patch.object(
            url_processing_service.processor_factory, "get_processor"
        ) as mock_get_processor:
            mock_processor = AsyncMock()
            mock_processor.process = AsyncMock(return_value="# Content")
            mock_get_processor.return_value = mock_processor

            with patch(
                "ragitect.services.url_processing_service.split_document",
                return_value=["Chunk"],
            ):
                with patch(
                    "ragitect.services.url_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=MagicMock()),
                ):
                    with patch(
                        "ragitect.services.url_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768]),
                    ):
                        # Act
                        await url_processing_service.process_url_document(
                            sample_url_document.id,
                            "https://example.com/article",
                            "url",
                        )

            # Assert
            mock_get_processor.assert_called_once_with(
                "https://example.com/article", "url"
            )

    async def test_uses_correct_processor_for_youtube(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_youtube_document,
    ):
        """Should use YouTubeProcessor for source_type='youtube'"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_youtube_document

        with patch.object(
            url_processing_service.processor_factory, "get_processor"
        ) as mock_get_processor:
            mock_processor = AsyncMock()
            mock_processor.process = AsyncMock(return_value="# YouTube Transcript")
            mock_get_processor.return_value = mock_processor

            with patch(
                "ragitect.services.url_processing_service.split_document",
                return_value=["Chunk"],
            ):
                with patch(
                    "ragitect.services.url_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=MagicMock()),
                ):
                    with patch(
                        "ragitect.services.url_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768]),
                    ):
                        # Act
                        await url_processing_service.process_url_document(
                            sample_youtube_document.id,
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                            "youtube",
                        )

            # Assert
            mock_get_processor.assert_called_once_with(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"
            )

    async def test_uses_correct_processor_for_pdf_url(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_pdf_url_document,
    ):
        """Should use PDFURLProcessor for source_type='pdf'"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_pdf_url_document

        with patch.object(
            url_processing_service.processor_factory, "get_processor"
        ) as mock_get_processor:
            mock_processor = AsyncMock()
            mock_processor.process = AsyncMock(return_value="# PDF Content")
            mock_get_processor.return_value = mock_processor

            with patch(
                "ragitect.services.url_processing_service.split_document",
                return_value=["Chunk"],
            ):
                with patch(
                    "ragitect.services.url_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=MagicMock()),
                ):
                    with patch(
                        "ragitect.services.url_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768]),
                    ):
                        # Act
                        await url_processing_service.process_url_document(
                            sample_pdf_url_document.id,
                            "https://arxiv.org/pdf/1706.03762.pdf",
                            "pdf",
                        )

            # Assert
            mock_get_processor.assert_called_once_with(
                "https://arxiv.org/pdf/1706.03762.pdf", "pdf"
            )


class TestURLProcessingServiceContentStorage:
    """Test that processor output is stored and chunked correctly (AC2)"""

    async def test_processor_output_stored_in_document(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Processor Markdown output should be stored in document"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document
        expected_markdown = "# Article Title\n\nThis is the article content."

        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(return_value=expected_markdown)

        with patch.object(
            url_processing_service.processor_factory,
            "get_processor",
            return_value=mock_processor,
        ):
            with patch(
                "ragitect.services.url_processing_service.split_document",
                return_value=["Chunk 1", "Chunk 2"],
            ):
                with patch(
                    "ragitect.services.url_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=MagicMock()),
                ):
                    with patch(
                        "ragitect.services.url_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768, [0.2] * 768]),
                    ):
                        # Act
                        await url_processing_service.process_url_document(
                            sample_url_document.id,
                            "https://example.com/article",
                            "url",
                        )

        # Assert
        mock_document_repo.update_processed_content.assert_called_once_with(
            sample_url_document.id, expected_markdown
        )

    async def test_chunks_stored_with_embeddings(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Chunks should be created and stored with embeddings"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(return_value="# Content\n\nParagraph 1")

        with patch.object(
            url_processing_service.processor_factory,
            "get_processor",
            return_value=mock_processor,
        ):
            with patch(
                "ragitect.services.url_processing_service.split_document",
                return_value=["Chunk 1", "Chunk 2"],
            ):
                with patch(
                    "ragitect.services.url_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=MagicMock()),
                ):
                    with patch(
                        "ragitect.services.url_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768, [0.2] * 768]),
                    ):
                        # Act
                        await url_processing_service.process_url_document(
                            sample_url_document.id,
                            "https://example.com/article",
                            "url",
                        )

        # Assert - chunks should be added
        mock_document_repo.add_chunks.assert_called_once()
        call_args = mock_document_repo.add_chunks.call_args[0]
        assert call_args[0] == sample_url_document.id  # document_id
        chunk_data = call_args[1]
        assert len(chunk_data) == 2  # Two chunks


class TestURLProcessingServiceErrorHandling:
    """Test error state handling (AC6)"""

    async def test_fetch_error_updates_status_to_error(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Fetch failure should update status to 'error' with message"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        with patch.object(
            url_processing_service.processor_factory,
            "get_processor",
            return_value=mock_processor,
        ):
            with patch(
                "ragitect.services.url_processing_service.with_retry",
                side_effect=Exception("Connection timeout"),
            ):
                # Act
                await url_processing_service.process_url_document(
                    sample_url_document.id,
                    "https://example.com/article",
                    "url",
                )

        # Assert - status should be updated to error
        error_update_calls = [
            call
            for call in mock_document_repo.update_metadata.call_args_list
            if call[0][1].get("status") == "error"
        ]
        assert len(error_update_calls) >= 1

        # Error message should be in metadata
        error_metadata = error_update_calls[-1][0][1]
        assert "error_message" in error_metadata
        assert len(error_metadata["error_message"]) > 0

    async def test_error_message_is_user_friendly(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Error message should be user-friendly, not technical stack trace"""
        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document
        mock_processor = AsyncMock()
        # Simulate a timeout error
        mock_processor.process = AsyncMock(
            side_effect=Exception("httpx.TimeoutException: timeout of 30s exceeded")
        )

        with patch.object(
            url_processing_service.processor_factory,
            "get_processor",
            return_value=mock_processor,
        ):
            with patch(
                "ragitect.services.url_processing_service.with_retry",
                side_effect=Exception("httpx.TimeoutException: timeout of 30s exceeded"),
            ):
                # Act
                await url_processing_service.process_url_document(
                    sample_url_document.id,
                    "https://example.com/article",
                    "url",
                )

        # Assert - error message should be concise and user-friendly
        error_calls = [
            call
            for call in mock_document_repo.update_metadata.call_args_list
            if call[0][1].get("status") == "error"
        ]
        if error_calls:
            error_msg = error_calls[-1][0][1].get("error_message", "")
            # Should not contain python traceback markers
            assert "Traceback" not in error_msg
            # Should be reasonably short (under 500 chars for user display)
            assert len(error_msg) < 500


class TestURLProcessingServiceConcurrency:
    """Test concurrency limiting with semaphore (AC5)"""

    async def test_semaphore_limits_concurrent_fetches(
        self,
        mock_session,
        mocker,
    ):
        """Should limit concurrent URL fetches to 5"""
        from ragitect.services.url_processing_service import _url_fetch_semaphore

        # Verify semaphore is configured with limit of 5
        assert _url_fetch_semaphore._value == 5

    async def test_semaphore_acquired_and_released(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Semaphore should be acquired at start and released at end"""
        from ragitect.services.url_processing_service import _url_fetch_semaphore

        # Record initial semaphore value
        initial_value = _url_fetch_semaphore._value

        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(return_value="# Content")

        with patch.object(
            url_processing_service.processor_factory,
            "get_processor",
            return_value=mock_processor,
        ):
            with patch("ragitect.services.url_processing_service.split_document", return_value=["Chunk"]):
                with patch("ragitect.services.url_processing_service.get_embedding_model_from_config", new=AsyncMock(return_value=MagicMock())):
                    with patch("ragitect.services.url_processing_service.embed_documents", new=AsyncMock(return_value=[[0.1] * 768])):
                        # Act
                        await url_processing_service.process_url_document(
                            sample_url_document.id,
                            "https://example.com/article",
                            "url",
                        )

        # Assert - semaphore should be back to initial value
        assert _url_fetch_semaphore._value == initial_value

    async def test_semaphore_released_on_error(
        self,
        url_processing_service,
        mock_document_repo,
        mock_session,
        sample_url_document,
    ):
        """Semaphore should be released even on error"""
        from ragitect.services.url_processing_service import _url_fetch_semaphore

        # Record initial semaphore value
        initial_value = _url_fetch_semaphore._value

        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_url_document

        with patch.object(
            url_processing_service.processor_factory,
            "get_processor",
            side_effect=Exception("Processor error"),
        ):
            # Act - should not raise (error handled internally)
            await url_processing_service.process_url_document(
                sample_url_document.id,
                "https://example.com/article",
                "url",
            )

        # Assert - semaphore should be back to initial value
        assert _url_fetch_semaphore._value == initial_value

    async def test_semaphore_used_via_context_manager(self, mocker):
        """Verify semaphore is used via async context manager for safe release"""
        # This test verifies implementation pattern
        from ragitect.services import url_processing_service

        # Read the source code to verify semaphore is used with async with
        import inspect
        source = inspect.getsource(url_processing_service.URLProcessingService.process_url_document)

        # Verify the pattern "async with _url_fetch_semaphore:" is present
        assert "async with _url_fetch_semaphore:" in source
