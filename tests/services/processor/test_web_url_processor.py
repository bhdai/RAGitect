"""Tests for WebURLProcessor - Web page fetching and markdown extraction

Red-Green-Refactor TDD: These tests define expected behavior before implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

# Module-level markers as per project-context.md
pytestmark = [pytest.mark.asyncio]


class TestWebURLProcessorInterface:
    """Test WebURLProcessor class interface and method signatures"""

    def test_class_exists(self):
        """WebURLProcessor class should be importable"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()
        assert processor is not None

    def test_inherits_from_base_document_processor(self):
        """WebURLProcessor should inherit from BaseDocumentProcessor"""
        from ragitect.services.processor.base import BaseDocumentProcessor
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()
        assert isinstance(processor, BaseDocumentProcessor)

    def test_supported_formats_returns_empty_list(self):
        """WebURLProcessor is not file-based, returns empty list"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()
        formats = processor.supported_formats()
        assert formats == []

    async def test_process_method_signature_async(self):
        """process() should be async and accept url string"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()
        # This will fail until implemented - verifies async signature
        with patch.object(processor, "_fetch_url", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = "<html><body>Test</body></html>"
            with patch("trafilatura.extract") as mock_extract:
                mock_extract.return_value = "# Test"
                result = await processor.process("https://example.com")
                assert isinstance(result, str)


class TestWebURLProcessorFetching:
    """Test HTTP fetching functionality"""

    async def test_successful_fetch_returns_markdown(self):
        """Successful fetch and extraction returns markdown string"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()

        mock_html = "<html><body><article><h1>Test Article</h1><p>Content here.</p></article></body></html>"
        expected_markdown = "# Test Article\n\nContent here."

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = mock_html
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch("trafilatura.extract") as mock_extract:
                mock_extract.return_value = expected_markdown

                result = await processor.process("https://example.com/article")

                assert result == expected_markdown
                mock_client.get.assert_called_once()
                mock_extract.assert_called_once()

    async def test_httpx_client_configured_with_timeouts(self):
        """httpx client should have 30s total, 10s connect timeout"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = "<html></html>"
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch("trafilatura.extract", return_value="# Test"):
                await processor.process("https://example.com")

                # Verify client was created with correct config
                call_kwargs = mock_client_class.call_args.kwargs
                assert "timeout" in call_kwargs
                timeout = call_kwargs["timeout"]
                assert timeout.connect == 10.0
                assert timeout.read == 30.0 or timeout.pool == 30.0

    async def test_httpx_client_has_http2_enabled(self):
        """httpx client should have HTTP/2 support enabled"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = "<html></html>"
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch("trafilatura.extract", return_value="# Test"):
                await processor.process("https://example.com")

                call_kwargs = mock_client_class.call_args.kwargs
                assert call_kwargs.get("http2") is True

    async def test_httpx_client_follows_redirects(self):
        """httpx client should follow redirects automatically"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = "<html></html>"
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch("trafilatura.extract", return_value="# Test"):
                await processor.process("https://example.com")

                call_kwargs = mock_client_class.call_args.kwargs
                assert call_kwargs.get("follow_redirects") is True


class TestWebURLProcessorContentExtraction:
    """Test content extraction via trafilatura"""

    async def test_trafilatura_extracts_main_content(self):
        """trafilatura should be called with html and return markdown"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()

        html_with_nav = """
        <html>
            <nav>Navigation menu</nav>
            <article>
                <h1>Main Article</h1>
                <p>This is the main content.</p>
            </article>
            <footer>Footer content</footer>
        </html>
        """
        expected_markdown = "# Main Article\n\nThis is the main content."

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = html_with_nav
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch("trafilatura.extract") as mock_extract:
                mock_extract.return_value = expected_markdown

                result = await processor.process("https://example.com")

                # Verify trafilatura was called with html content
                mock_extract.assert_called_once()
                call_args = mock_extract.call_args
                # First positional arg should be the HTML content
                assert call_args[0][0] == html_with_nav

                # Verify result is the markdown
                assert result == expected_markdown


class TestWebURLProcessorErrorHandling:
    """Test error handling for various failure scenarios"""

    async def test_timeout_raises_url_fetch_error(self):
        """Timeout should raise URLFetchError with descriptive message"""
        from ragitect.services.processor.web_url_processor import (
            WebURLProcessor,
            URLFetchError,
        )

        processor = WebURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(URLFetchError) as exc_info:
                await processor.process("https://slow-site.com")

            assert "slow-site.com" in str(exc_info.value)
            assert "timeout" in str(exc_info.value).lower()

    async def test_connection_error_raises_url_fetch_error(self):
        """Connection error should raise URLFetchError with details"""
        from ragitect.services.processor.web_url_processor import (
            WebURLProcessor,
            URLFetchError,
        )

        processor = WebURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(URLFetchError) as exc_info:
                await processor.process("https://unreachable.com")

            assert "unreachable.com" in str(exc_info.value)

    async def test_http_404_raises_url_fetch_error(self):
        """HTTP 404 should raise URLFetchError with status code"""
        from ragitect.services.processor.web_url_processor import (
            WebURLProcessor,
            URLFetchError,
        )

        processor = WebURLProcessor()

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

            with pytest.raises(URLFetchError) as exc_info:
                await processor.process("https://example.com/missing")

            assert "404" in str(exc_info.value)
            assert "example.com" in str(exc_info.value)

    async def test_http_500_raises_url_fetch_error(self):
        """HTTP 500 should raise URLFetchError with status code"""
        from ragitect.services.processor.web_url_processor import (
            WebURLProcessor,
            URLFetchError,
        )

        processor = WebURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_request = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Internal Server Error", request=mock_request, response=mock_response
            )
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(URLFetchError) as exc_info:
                await processor.process("https://example.com/error")

            assert "500" in str(exc_info.value)

    async def test_trafilatura_returns_none_raises_content_extraction_error(self):
        """Empty extraction should raise ContentExtractionError"""
        from ragitect.services.processor.web_url_processor import (
            WebURLProcessor,
            ContentExtractionError,
        )

        processor = WebURLProcessor()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.text = "<html><body></body></html>"
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch("trafilatura.extract", return_value=None):
                with pytest.raises(ContentExtractionError) as exc_info:
                    await processor.process("https://empty-page.com")

                assert "empty-page.com" in str(exc_info.value)

    async def test_exception_messages_contain_url(self):
        """All exception messages should include the URL for debugging"""
        from ragitect.services.processor.web_url_processor import (
            WebURLProcessor,
            URLFetchError,
        )

        processor = WebURLProcessor()
        test_url = "https://test-debugging.example.com/path"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(URLFetchError) as exc_info:
                await processor.process(test_url)

            # URL should be in the error message for debugging
            assert "test-debugging.example.com" in str(exc_info.value)


class TestWebURLProcessorExceptions:
    """Test custom exception classes exist and are properly defined"""

    def test_url_fetch_error_exists(self):
        """URLFetchError exception class should exist"""
        from ragitect.services.processor.web_url_processor import URLFetchError

        error = URLFetchError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_content_extraction_error_exists(self):
        """ContentExtractionError exception class should exist"""
        from ragitect.services.processor.web_url_processor import ContentExtractionError

        error = ContentExtractionError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"


@pytest.mark.integration
class TestWebURLProcessorIntegration:
    """Integration tests with real web pages (require network access)"""

    async def test_process_wikipedia_article(self):
        """Integration test: fetch real Wikipedia page and extract markdown"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

        # Fetch real page
        markdown = await processor.process(url)

        # Verify main content present
        assert "Python" in markdown
        assert "#" in markdown  # Has headings
        assert len(markdown) > 1000  # Substantial content extracted

        # Verify markdown format
        assert any(
            marker in markdown for marker in ["#", "##", "###"]
        )  # Has markdown headings

    async def test_process_removes_navigation_boilerplate(self):
        """Integration test: verify navigation/boilerplate is stripped"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor

        processor = WebURLProcessor()
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

        markdown = await processor.process(url)
        markdown_lower = markdown.lower()

        # These Wikipedia navigation elements should NOT be in extracted content
        # Note: Some may appear in actual article text, so we check for common nav patterns
        navigation_patterns = [
            "jump to navigation",
            "jump to search",
            "personal tools",
            "[edit]",  # Wikipedia edit links
        ]

        # At least most navigation patterns should be absent
        nav_found = sum(1 for p in navigation_patterns if p in markdown_lower)
        assert nav_found <= 1, f"Too many nav patterns found: {nav_found}"

    async def test_markdown_compatible_with_chunking(self):
        """Integration test: verify markdown works with DocumentProcessor chunking"""
        from ragitect.services.processor.web_url_processor import WebURLProcessor
        from ragitect.services.document_processor import split_markdown_document

        processor = WebURLProcessor()
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

        # Fetch and extract
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

