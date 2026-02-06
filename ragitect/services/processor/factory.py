"""
Document Processor Factory

Selects the appropriate document processor based on file type or URL source type.
Uses a simple two-processor strategy for files:
- SimpleProcessor for text-based formats (lightweight, fast)
- DoclingProcessor for complex formats (PDF, DOCX, etc.)

For URL sources (Story 5.5):
- WebURLProcessor for web pages
- YouTubeProcessor for YouTube videos
- PDFURLProcessor for PDF URLs
"""

import logging
from pathlib import Path
from typing import Literal

from ragitect.services.processor.base import BaseDocumentProcessor
from ragitect.services.processor.docling_processor import DoclingProcessor
from ragitect.services.processor.pdf_url_processor import PDFURLProcessor
from ragitect.services.processor.simple import SimpleProcessor
from ragitect.services.processor.web_url_processor import WebURLProcessor
from ragitect.services.processor.youtube_processor import YouTubeProcessor

logger = logging.getLogger(__name__)


class UnsupportedFormatError(Exception):
    """
    Raised when no processor supports the uploaded file format.

    This exception provides a clear signal to the UI layer that the file
    type is not supported, allowing for appropriate user feedback.
    """

    pass


class ProcessorFactory:
    """Factory for selecting appropriate document processor based on filetype or URL source type"""

    def __init__(self):
        logger.info("Initializing ProcessorFactory")
        # File-based processors
        self.simple_processor: BaseDocumentProcessor = SimpleProcessor()
        self.docling_processor: BaseDocumentProcessor = DoclingProcessor()

        # URL-based processors (Story 5.5)
        self.web_url_processor: WebURLProcessor = WebURLProcessor()
        self.youtube_processor: YouTubeProcessor = YouTubeProcessor()
        self.pdf_url_processor: PDFURLProcessor = PDFURLProcessor()

        self.text_formats: list[str] = self.simple_processor.supported_formats()
        self.complex_formats: list[str] = self.docling_processor.supported_formats()

        logger.info(
            f"ProcessorFactory initialized - Text formats: {len(self.text_formats)}, Complex formats: {len(self.complex_formats)}"
        )

    def get_processor(
        self,
        source: str,
        source_type: Literal["file", "url", "youtube", "pdf"] = "file",
    ) -> BaseDocumentProcessor:
        """Select appropriate processor for given source

        Args:
            source: File name (for files) or URL (for URLs)
            source_type: Type of source - "file", "url", "youtube", or "pdf"
                - "file" (default): Route by file extension
                - "url": Use WebURLProcessor for web pages
                - "youtube": Use YouTubeProcessor for YouTube videos
                - "pdf": Use PDFURLProcessor for PDF URLs

        Returns:
            BaseDocumentProcessor: Appropriate processor for the source type

        Raises:
            UnsupportedFormatError: If source_type is "file" and the file extension
                is not supported by any processor
        """
        # URL-based routing (Story 5.5)
        if source_type == "url":
            logger.info(f"Selected WebURLProcessor for URL: {source[:50]}...")
            return self.web_url_processor

        if source_type == "youtube":
            logger.info(f"Selected YouTubeProcessor for YouTube URL: {source[:50]}...")
            return self.youtube_processor

        if source_type == "pdf":
            logger.info(f"Selected PDFURLProcessor for PDF URL: {source[:50]}...")
            return self.pdf_url_processor

        # Existing file-based routing (backward compatible)
        ext = Path(source).suffix.lower()

        if ext in self.text_formats:
            logger.info(f"Selected SimpleProcessor for file {source}")
            return self.simple_processor

        if ext in self.complex_formats:
            logger.info(f"Selected DoclingProcessor for file {source}")
            return self.docling_processor

        all_formats = sorted(self.text_formats + self.complex_formats)

        error_msg = (
            f"Unsupported format: {ext}\nSupported formats: {', '.join(all_formats)}"
        )

        logger.warning(f"Unsupported file format attempted: {ext} (file: {source})")
        raise UnsupportedFormatError(error_msg)
