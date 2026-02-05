"""PDF URL Processor - Downloads PDF files from URLs and converts to Markdown.

This processor handles PDF URL ingestion by:
1. Validating that the URL points to a PDF file (extension or Content-Type)
2. Downloading PDF content via httpx with proper timeout/HTTP2/redirect configuration
3. Delegating to DoclingProcessor for PDF → Markdown conversion

Usage:
    processor = PDFURLProcessor()
    markdown = await processor.process("https://arxiv.org/pdf/1706.03762.pdf")

Supported URL patterns:
    - arXiv papers: https://arxiv.org/pdf/2301.12345.pdf
    - Direct PDF links: https://example.com/document.pdf
    - University research papers: https://stanford.edu/papers/paper.pdf
    - Documentation: https://docs.example.com/manual.pdf

Note:
    This processor inherits from BaseDocumentProcessor but overrides with an
    async signature. The async process(url: str) method is used for URL fetching.

    Integration with ProcessorFactory happens in Story 5.5.

Exceptions:
    InvalidPDFURLError: URL does not point to a PDF file
    PDFDownloadError: HTTP request failed (timeout, connection error, 4xx/5xx)
    PDFProcessingError: DoclingProcessor failed to process PDF
"""

import asyncio
import logging
from typing import override
from urllib.parse import unquote, urlparse

import httpx

from ragitect.services.processor.base import BaseDocumentProcessor
from ragitect.services.processor.docling_processor import DoclingProcessor

logger = logging.getLogger(__name__)


class InvalidPDFURLError(Exception):
    """Raised when URL does not point to a valid PDF file.

    Causes:
    - URL does not end with .pdf AND Content-Type is not application/pdf
    - HEAD request fails to validate content type

    Attributes:
        url: The URL that failed validation
        message: Descriptive error message including URL
    """

    pass


class PDFDownloadError(Exception):
    """Raised when PDF download fails.

    Causes:
    - HTTP timeout (30s limit)
    - Connection error
    - HTTP 4xx/5xx status codes

    Attributes:
        url: The URL that failed to download
        message: Descriptive error message including URL and error type
    """

    pass


class PDFProcessingError(Exception):
    """Raised when DoclingProcessor fails to process PDF.

    Causes:
    - Corrupted PDF
    - Password-protected PDF
    - Image-only PDF without OCR

    Attributes:
        url: The URL of the PDF that failed processing
        message: Descriptive error message including URL and error details
    """

    pass


class PDFURLProcessor(BaseDocumentProcessor):
    """Processor for downloading PDFs from URLs and converting to Markdown.

    Inherits from BaseDocumentProcessor but provides an async process(url: str)
    method instead of the sync process(file_bytes, file_name) method.

    Implements async PDF download with:
    - 30 second total timeout, 10 second connect timeout (NFR-P4)
    - HTTP/2 support for improved performance
    - Automatic redirect following
    - Connection pooling (max 20 keepalive connections)

    PDF URL validation:
    - Fast path: URL ends with .pdf extension
    - Fallback: HEAD request to check Content-Type: application/pdf

    PDF processing delegated to DoclingProcessor for:
    - Robust PDF parsing with Docling library
    - Table structure detection
    - Clean Markdown output

    Example:
        >>> processor = PDFURLProcessor()
        >>> markdown = await processor.process("https://arxiv.org/pdf/1706.03762.pdf")
        >>> print(markdown[:100])
        # Attention Is All You Need
        ...
    """

    def __init__(self) -> None:
        """Initialize PDFURLProcessor with DoclingProcessor for PDF conversion."""
        self._docling_processor = DoclingProcessor()

    @override
    def supported_formats(self) -> list[str]:
        """Return list of supported file extensions.

        PDFURLProcessor is not file-based, so returns empty list.
        URL-based routing is handled separately from file extension routing.

        Returns:
            Empty list (not file-based)
        """
        return []

    async def process(self, url: str) -> str:
        """Download PDF from URL and convert to Markdown.

        Args:
            url: HTTP or HTTPS URL pointing to a PDF file

        Returns:
            Markdown string with PDF content extracted

        Raises:
            InvalidPDFURLError: If URL does not point to a PDF file
            PDFDownloadError: If HTTP request fails (timeout, connection error, 4xx/5xx)
            PDFProcessingError: If DoclingProcessor fails to process PDF

        Example:
            >>> processor = PDFURLProcessor()
            >>> markdown = await processor.process("https://arxiv.org/pdf/1706.03762.pdf")
        """
        logger.info(f"Processing PDF URL: {url}")

        # Validate URL points to PDF
        if not await self._validate_pdf_url(url):
            raise InvalidPDFURLError(f"URL does not point to a PDF file: {url}")

        # Download PDF bytes
        pdf_bytes = await self._download_pdf(url)
        logger.info(f"Downloaded {len(pdf_bytes)} bytes from {url}")

        # Extract filename from URL for DoclingProcessor
        file_name = self._extract_filename(url)

        # Delegate to DoclingProcessor (sync, runs in executor via asyncio.to_thread)
        try:
            markdown = await asyncio.to_thread(
                self._docling_processor.process,
                pdf_bytes,
                file_name,
            )
        except Exception as e:
            logger.error(f"Failed to process PDF from {url}: {e}")
            raise PDFProcessingError(f"Failed to process PDF from {url}: {e}") from e

        logger.info(f"Successfully processed PDF {url} - {len(markdown)} chars extracted")
        return markdown

    async def _validate_pdf_url(self, url: str) -> bool:
        """Validate that URL points to a PDF file.

        Two-tier validation:
        1. Fast path: Check if URL ends with .pdf
        2. Fallback: HEAD request to check Content-Type

        Args:
            url: URL to validate

        Returns:
            True if URL points to a PDF, False otherwise
        """
        # Fast path: URL ends with .pdf (case-insensitive)
        if url.lower().rstrip("/").endswith(".pdf"):
            return True

        # Fallback: HEAD request to check Content-Type
        timeout = httpx.Timeout(10.0, connect=5.0)
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; RAGitect/1.0; +https://github.com/bhdai/ragitect)"
        }

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            try:
                response = await client.head(url)
                content_type = response.headers.get("content-type", "").lower()
                return "application/pdf" in content_type
            except httpx.HTTPError:
                # If HEAD fails, return False - caller will handle
                return False

    async def _download_pdf(self, url: str) -> bytes:
        """Download PDF bytes from URL.

        Args:
            url: URL to download PDF from

        Returns:
            PDF bytes

        Raises:
            PDFDownloadError: On timeout, connection error, or HTTP error status
        """
        # Configure timeout: 30s total, 10s connect (NFR-P4)
        timeout = httpx.Timeout(30.0, connect=10.0)

        # Configure connection limits for pooling
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
                return response.content
            except httpx.TimeoutException as e:
                logger.error(f"Timeout downloading PDF from {url}: {e}")
                raise PDFDownloadError(
                    f"Timeout downloading PDF from {url} (30s limit)"
                ) from e
            except httpx.ConnectError as e:
                logger.error(f"Connection error downloading PDF from {url}: {e}")
                raise PDFDownloadError(
                    f"Connection error downloading PDF from {url}: {str(e)}"
                ) from e
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                logger.error(f"HTTP {status} downloading PDF from {url}")
                if status == 404:
                    raise PDFDownloadError(
                        f"PDF not found (404): {url}"
                    ) from e
                elif status == 403:
                    raise PDFDownloadError(
                        f"Access denied (403): {url}"
                    ) from e
                else:
                    raise PDFDownloadError(
                        f"HTTP {status} downloading PDF from {url}"
                    ) from e

    def _extract_filename(self, url: str) -> str:
        """Extract filename from URL path.

        Args:
            url: URL to extract filename from

        Returns:
            Filename with .pdf extension
        """
        parsed = urlparse(url)
        path = unquote(parsed.path)

        # Get last path segment
        segments = [s for s in path.split("/") if s]
        if segments:
            filename = segments[-1]
            # Ensure .pdf extension
            if not filename.lower().endswith(".pdf"):
                filename = f"{filename}.pdf"
            return filename

        # Fallback
        return "downloaded.pdf"
