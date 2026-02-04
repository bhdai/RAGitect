"""Document processors for RAGitect.

This module provides processors for extracting text/markdown from various sources.
"""

from ragitect.services.processor.base import BaseDocumentProcessor
from ragitect.services.processor.pdf_url_processor import (
    InvalidPDFURLError,
    PDFDownloadError,
    PDFProcessingError,
    PDFURLProcessor,
)
from ragitect.services.processor.simple import SimpleProcessor
from ragitect.services.processor.web_url_processor import (
    ContentExtractionError,
    URLFetchError,
    WebURLProcessor,
)
from ragitect.services.processor.youtube_processor import (
    InvalidYouTubeURLError,
    TranscriptUnavailableError,
    YouTubeProcessor,
)

__all__ = [
    "BaseDocumentProcessor",
    "ContentExtractionError",
    "InvalidPDFURLError",
    "InvalidYouTubeURLError",
    "PDFDownloadError",
    "PDFProcessingError",
    "PDFURLProcessor",
    "SimpleProcessor",
    "TranscriptUnavailableError",
    "URLFetchError",
    "WebURLProcessor",
    "YouTubeProcessor",
]