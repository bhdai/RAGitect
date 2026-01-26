"""Web URL Processor - Fetches web pages and converts to clean Markdown.

This processor handles web URL ingestion by:
1. Fetching HTML content via httpx with proper timeout/HTTP2/redirect configuration
2. Extracting main article content using trafilatura (removes nav, ads, footers)
3. Converting to Markdown format for downstream chunking and embedding

Usage:
    processor = WebURLProcessor()
    markdown = await processor.process("https://example.com/article")

Note:
    This processor inherits from BaseDocumentProcessor but overrides with an
    async signature. The async process(url: str) method is used for URL fetching.
    The sync process(file_bytes, file_name) method raises NotImplementedError.

    Integration with ProcessorFactory happens in Story 5.5.
"""

import logging
from typing import override

import httpx
import trafilatura

from ragitect.services.processor.base import BaseDocumentProcessor

logger = logging.getLogger(__name__)


class URLFetchError(Exception):
    """Raised when HTTP request fails (timeout, connection error, 4xx/5xx).

    Attributes:
        url: The URL that failed to fetch
        message: Descriptive error message including URL and error type
    """

    pass


class ContentExtractionError(Exception):
    """Raised when content extraction fails (trafilatura returns None).

    Attributes:
        url: The URL where content extraction failed
        message: Descriptive error message
    """

    pass


class WebURLProcessor(BaseDocumentProcessor):
    """Processor for fetching web pages and converting to Markdown.

    Inherits from BaseDocumentProcessor but provides an async process(url: str)
    method instead of the sync process(file_bytes, file_name) method.

    Implements async web page fetching with:
    - 30 second total timeout, 10 second connect timeout (NFR-P4)
    - HTTP/2 support for improved performance
    - Automatic redirect following
    - Connection pooling (max 20 keepalive connections)

    Content extraction uses trafilatura to:
    - Extract main article content
    - Remove navigation, ads, headers, footers
    - Strip potentially malicious elements (scripts, iframes) for security (NFR-S5)
    - Capture article metadata (title, author, date) when available

    Example:
        >>> processor = WebURLProcessor()
        >>> markdown = await processor.process("https://en.wikipedia.org/wiki/Python")
        >>> print(markdown[:100])
        # Python (programming language)
        ...
    """

    @override
    def supported_formats(self) -> list[str]:
        """Return list of supported file extensions.

        WebURLProcessor is not file-based, so returns empty list.
        URL-based routing is handled separately from file extension routing.

        Returns:
            Empty list (not file-based)
        """
        return []

    async def process(self, url: str) -> str:
        """Fetch web page and convert to clean Markdown.

        Args:
            url: HTTP or HTTPS URL to fetch

        Returns:
            Markdown string with main article content extracted

        Raises:
            URLFetchError: If HTTP request fails (timeout, connection error, 4xx/5xx)
            ContentExtractionError: If content extraction fails (empty page)
        """
        logger.info(f"Processing web URL: {url}")

        # Fetch HTML content
        html_content = await self._fetch_url(url)

        # Extract main content and convert to Markdown
        markdown = self._extract_content(html_content, url)

        logger.info(f"Successfully processed {url} - {len(markdown)} chars extracted")
        return markdown

    async def _fetch_url(self, url: str) -> str:
        """Fetch HTML content from URL with configured httpx client.

        Args:
            url: URL to fetch

        Returns:
            HTML content as string

        Raises:
            URLFetchError: On timeout, connection error, or HTTP error status
        """
        # Configure timeout: 30s total, 10s connect (NFR-P4)
        timeout = httpx.Timeout(30.0, connect=10.0)

        # Configure connection limits for pooling (NFR-R3)
        limits = httpx.Limits(max_keepalive_connections=20)

        # Set User-Agent to avoid 403 from sites that block automated requests
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; RAGitect/1.0; +https://github.com/bhdai/ragitect)"
        }

        async with httpx.AsyncClient(
            timeout=timeout,
            http2=True,  # Enable HTTP/2 support
            follow_redirects=True,  # Auto-follow redirects
            limits=limits,  # Connection pooling
            headers=headers,  # Default headers for all requests
        ) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
            except httpx.TimeoutException as e:
                logger.error(f"Timeout fetching {url}: {e}")
                raise URLFetchError(f"Timeout fetching {url} (30s limit)")
            except httpx.ConnectError as e:
                logger.error(f"Connection error fetching {url}: {e}")
                raise URLFetchError(f"Connection error fetching {url}: {str(e)}")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP {e.response.status_code} fetching {url}")
                raise URLFetchError(f"HTTP {e.response.status_code} fetching {url}")

    def _extract_content(self, html_content: str, url: str) -> str:
        """Extract main article content and convert to Markdown.

        Uses trafilatura for:
        - Main content extraction (removes nav, ads, footers)
        - Script/iframe stripping (NFR-S5 security)
        - Direct Markdown output

        Args:
            html_content: Raw HTML content
            url: Original URL (for error messages)

        Returns:
            Markdown string with extracted content

        Raises:
            ContentExtractionError: If extraction returns None/empty
        """
        # Extract main content with trafilatura
        # output_format="markdown" gives us direct Markdown output
        markdown = trafilatura.extract(
            html_content,
            output_format="markdown",
            include_comments=False,  # Exclude comments
            include_tables=True,  # Keep tables
            include_images=True,  # Keep image references
            include_links=True,  # Keep hyperlinks
            no_fallback=False,  # Use fallback extraction if main method fails
        )

        if markdown is None:
            logger.error(f"Failed to extract content from {url}")
            raise ContentExtractionError(f"Failed to extract content from {url}")

        return markdown
