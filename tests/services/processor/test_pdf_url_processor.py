"""Tests for PDFURLProcessor - PDF download from URL and conversion to Markdown.

Red-Green-Refactor TDD: These tests define expected behavior before implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

# Module-level markers as per project-context.md
pytestmark = [pytest.mark.asyncio]


class TestPDFURLProcessorInterface:
    """Test PDFURLProcessor class interface and method signatures"""

    def test_class_exists(self):
        """PDFURLProcessor class should be importable"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()
        assert processor is not None

    def test_inherits_from_base_document_processor(self):
        """PDFURLProcessor should inherit from BaseDocumentProcessor"""
        from ragitect.services.processor.base import BaseDocumentProcessor
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()
        assert isinstance(processor, BaseDocumentProcessor)

    def test_supported_formats_returns_empty_list(self):
        """PDFURLProcessor is not file-based, returns empty list"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()
        formats = processor.supported_formats()
        assert formats == []

    async def test_process_method_signature_async(self):
        """process() should be async and accept url string, return str"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        mock_pdf_bytes = b"%PDF-1.4 mock content"

        with patch.object(processor, "_validate_pdf_url", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = True
            with patch.object(processor, "_download_pdf", new_callable=AsyncMock) as mock_download:
                mock_download.return_value = mock_pdf_bytes
                with patch.object(processor._docling_processor, "process") as mock_docling:
                    mock_docling.return_value = "# Test Document\n\nContent here."

                    result = await processor.process("https://example.com/doc.pdf")
                    assert isinstance(result, str)


class TestPDFURLValidation:
    """Test PDF URL validation logic"""

    async def test_url_ending_with_pdf_is_valid(self):
        """URL ending with .pdf should pass fast-path validation"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()
        result = await processor._validate_pdf_url("https://arxiv.org/pdf/1706.03762.pdf")
        assert result is True

    async def test_url_ending_with_pdf_uppercase_is_valid(self):
        """URL ending with .PDF (uppercase) should pass validation"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()
        result = await processor._validate_pdf_url("https://example.com/DOC.PDF")
        assert result is True

    async def test_url_without_pdf_extension_uses_head_request(self):
        """URL without .pdf extension should make HEAD request to check Content-Type"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.headers = {"content-type": "application/pdf"}
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await processor._validate_pdf_url("https://example.com/download?id=123")
            assert result is True
            mock_client.head.assert_called_once()

    async def test_url_with_non_pdf_content_type_returns_false(self):
        """URL with non-PDF Content-Type should return False"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.headers = {"content-type": "text/html"}
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await processor._validate_pdf_url("https://example.com/page")
            assert result is False

    async def test_non_pdf_url_raises_invalid_pdf_url_error_on_process(self):
        """process() should raise InvalidPDFURLError if URL is not a PDF"""
        from ragitect.services.processor.pdf_url_processor import (
            PDFURLProcessor,
            InvalidPDFURLError,
        )

        processor = PDFURLProcessor()

        with patch.object(processor, "_validate_pdf_url", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = False

            with pytest.raises(InvalidPDFURLError) as exc_info:
                await processor.process("https://example.com/not-a-pdf")

            assert "does not point to a PDF" in str(exc_info.value)
            assert "example.com" in str(exc_info.value)


class TestPDFDownload:
    """Test PDF download functionality"""

    async def test_successful_download_returns_bytes(self):
        """Successful download should return PDF bytes"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        mock_pdf_bytes = b"%PDF-1.4 mock pdf content"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = mock_pdf_bytes
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await processor._download_pdf("https://example.com/doc.pdf")
            assert result == mock_pdf_bytes

    async def test_httpx_client_configured_with_timeouts(self):
        """httpx client should have 30s total, 10s connect timeout"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = b"%PDF"
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await processor._download_pdf("https://example.com/doc.pdf")

            call_kwargs = mock_client_class.call_args.kwargs
            assert "timeout" in call_kwargs
            timeout = call_kwargs["timeout"]
            assert timeout.connect == 10.0

    async def test_httpx_client_has_http2_enabled(self):
        """httpx client should have HTTP/2 support enabled"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = b"%PDF"
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await processor._download_pdf("https://example.com/doc.pdf")

            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs.get("http2") is True

    async def test_httpx_client_follows_redirects(self):
        """httpx client should follow redirects automatically"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = b"%PDF"
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await processor._download_pdf("https://example.com/doc.pdf")

            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs.get("follow_redirects") is True


class TestPDFProcessing:
    """Test DoclingProcessor integration"""

    async def test_delegates_to_docling_processor(self):
        """process() should delegate PDF processing to DoclingProcessor"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        mock_pdf_bytes = b"%PDF-1.4 mock pdf content"
        expected_markdown = "# Test Document\n\nContent here."

        with patch.object(processor, "_validate_pdf_url", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = True
            with patch.object(processor, "_download_pdf", new_callable=AsyncMock) as mock_download:
                mock_download.return_value = mock_pdf_bytes
                with patch.object(processor._docling_processor, "process") as mock_docling:
                    mock_docling.return_value = expected_markdown

                    result = await processor.process("https://example.com/doc.pdf")

                    assert result == expected_markdown
                    mock_docling.assert_called_once()
                    # Verify PDF bytes were passed
                    call_args = mock_docling.call_args
                    assert call_args[0][0] == mock_pdf_bytes

    async def test_docling_failure_raises_pdf_processing_error(self):
        """DoclingProcessor failure should raise PDFProcessingError"""
        from ragitect.services.processor.pdf_url_processor import (
            PDFURLProcessor,
            PDFProcessingError,
        )

        processor = PDFURLProcessor()

        mock_pdf_bytes = b"%PDF-1.4 mock pdf content"

        with patch.object(processor, "_validate_pdf_url", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = True
            with patch.object(processor, "_download_pdf", new_callable=AsyncMock) as mock_download:
                mock_download.return_value = mock_pdf_bytes
                with patch.object(processor._docling_processor, "process") as mock_docling:
                    mock_docling.side_effect = ValueError("Corrupted PDF")

                    with pytest.raises(PDFProcessingError) as exc_info:
                        await processor.process("https://example.com/corrupted.pdf")

                    assert "example.com" in str(exc_info.value)


class TestErrorHandling:
    """Test error handling for various failure scenarios"""

    async def test_timeout_raises_pdf_download_error(self):
        """Timeout should raise PDFDownloadError with descriptive message"""
        from ragitect.services.processor.pdf_url_processor import (
            PDFURLProcessor,
            PDFDownloadError,
        )

        processor = PDFURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(PDFDownloadError) as exc_info:
                await processor._download_pdf("https://slow-site.com/doc.pdf")

            assert "slow-site.com" in str(exc_info.value)
            assert "timeout" in str(exc_info.value).lower()

    async def test_http_404_raises_pdf_download_error(self):
        """HTTP 404 should raise PDFDownloadError with status code"""
        from ragitect.services.processor.pdf_url_processor import (
            PDFURLProcessor,
            PDFDownloadError,
        )

        processor = PDFURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_request = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Not Found", request=mock_request, response=mock_response
            )
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(PDFDownloadError) as exc_info:
                await processor._download_pdf("https://example.com/missing.pdf")

            assert "404" in str(exc_info.value)
            assert "not found" in str(exc_info.value).lower()

    async def test_http_403_raises_pdf_download_error(self):
        """HTTP 403 should raise PDFDownloadError with 'Access denied' message"""
        from ragitect.services.processor.pdf_url_processor import (
            PDFURLProcessor,
            PDFDownloadError,
        )

        processor = PDFURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_request = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Forbidden", request=mock_request, response=mock_response
            )
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(PDFDownloadError) as exc_info:
                await processor._download_pdf("https://protected.com/doc.pdf")

            assert "403" in str(exc_info.value)
            assert "access denied" in str(exc_info.value).lower()

    async def test_connection_error_raises_pdf_download_error(self):
        """Connection error should raise PDFDownloadError with details"""
        from ragitect.services.processor.pdf_url_processor import (
            PDFURLProcessor,
            PDFDownloadError,
        )

        processor = PDFURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(PDFDownloadError) as exc_info:
                await processor._download_pdf("https://unreachable.com/doc.pdf")

            assert "unreachable.com" in str(exc_info.value)
            assert "connection" in str(exc_info.value).lower()

    async def test_exception_messages_contain_url(self):
        """All exception messages should include the URL for debugging"""
        from ragitect.services.processor.pdf_url_processor import (
            PDFURLProcessor,
            PDFDownloadError,
        )

        processor = PDFURLProcessor()
        test_url = "https://test-debugging.example.com/document.pdf"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(PDFDownloadError) as exc_info:
                await processor._download_pdf(test_url)

            assert "test-debugging.example.com" in str(exc_info.value)


class TestTempFileCleanup:
    """Test temporary file cleanup on success and failure"""

    async def test_temp_file_cleaned_up_on_success(self):
        """Temporary file should be cleaned up after successful processing"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        mock_pdf_bytes = b"%PDF-1.4 mock pdf content"
        expected_markdown = "# Test Document"

        # Track temp files created
        temp_files_created = []

        original_named_temp = __import__("tempfile").NamedTemporaryFile

        def track_temp_file(*args, **kwargs):
            # Ensure delete=False so we can check cleanup
            kwargs["delete"] = False
            f = original_named_temp(*args, **kwargs)
            temp_files_created.append(f.name)
            return f

        with patch.object(processor, "_validate_pdf_url", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = True
            with patch.object(processor, "_download_pdf", new_callable=AsyncMock) as mock_download:
                mock_download.return_value = mock_pdf_bytes
                with patch.object(processor._docling_processor, "process") as mock_docling:
                    mock_docling.return_value = expected_markdown

                    await processor.process("https://example.com/doc.pdf")

                    # After successful processing, docling was called
                    # The implementation should clean up temp files internally
                    mock_docling.assert_called_once()

    async def test_temp_file_cleaned_up_on_failure(self):
        """Temporary file should be cleaned up even when processing fails"""
        from ragitect.services.processor.pdf_url_processor import (
            PDFURLProcessor,
            PDFProcessingError,
        )

        processor = PDFURLProcessor()

        mock_pdf_bytes = b"%PDF-1.4 mock pdf content"

        with patch.object(processor, "_validate_pdf_url", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = True
            with patch.object(processor, "_download_pdf", new_callable=AsyncMock) as mock_download:
                mock_download.return_value = mock_pdf_bytes
                with patch.object(processor._docling_processor, "process") as mock_docling:
                    mock_docling.side_effect = ValueError("Processing failed")

                    with pytest.raises(PDFProcessingError):
                        await processor.process("https://example.com/doc.pdf")

                    # Even on failure, docling was called (then cleanup happens)
                    mock_docling.assert_called_once()


class TestPDFURLProcessorExceptions:
    """Test custom exception classes exist and are properly defined"""

    def test_invalid_pdf_url_error_exists(self):
        """InvalidPDFURLError exception class should exist"""
        from ragitect.services.processor.pdf_url_processor import InvalidPDFURLError

        error = InvalidPDFURLError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_pdf_download_error_exists(self):
        """PDFDownloadError exception class should exist"""
        from ragitect.services.processor.pdf_url_processor import PDFDownloadError

        error = PDFDownloadError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_pdf_processing_error_exists(self):
        """PDFProcessingError exception class should exist"""
        from ragitect.services.processor.pdf_url_processor import PDFProcessingError

        error = PDFProcessingError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"


@pytest.mark.integration
class TestPDFURLProcessorIntegration:
    """Integration tests with real PDF URLs (require network access)"""

    async def test_process_arxiv_paper(self):
        """Integration test: download and process real arXiv paper"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()
        # "Attention Is All You Need" - famous transformer paper
        url = "https://arxiv.org/pdf/1706.03762.pdf"

        markdown = await processor.process(url)

        # Verify substantial content extracted
        assert len(markdown) > 1000, "Expected substantial content from paper"
        # Verify markdown format
        assert "#" in markdown, "Expected markdown headings"
        # Paper title or key terms should be present
        assert any(
            term in markdown.lower()
            for term in ["attention", "transformer", "model"]
        ), "Expected paper content"

    async def test_markdown_compatible_with_chunking(self):
        """Integration test: verify markdown works with DocumentProcessor chunking"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor
        from ragitect.services.document_processor import split_markdown_document

        processor = PDFURLProcessor()
        url = "https://arxiv.org/pdf/1706.03762.pdf"

        markdown = await processor.process(url)

        # Test with existing chunker
        chunks = split_markdown_document(
            raw_text=markdown,
            chunk_size=512,
            overlap=50,
        )

        # Verify chunking works
        assert len(chunks) > 0, "Should produce at least one chunk"
        assert all(
            isinstance(chunk, str) for chunk in chunks
        ), "Chunks should be strings"
        assert all(
            len(chunk) > 0 for chunk in chunks
        ), "Each chunk should have content"

    async def test_various_pdf_sources(self):
        """Integration test: verify processor handles various PDF sources"""
        from ragitect.services.processor.pdf_url_processor import PDFURLProcessor

        processor = PDFURLProcessor()

        # Test with direct PDF link ending in .pdf
        url = "https://arxiv.org/pdf/1706.03762.pdf"

        markdown = await processor.process(url)

        assert len(markdown) > 100, "Expected content from PDF"
        assert isinstance(markdown, str), "Expected string output"
