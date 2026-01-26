"""Document processors for RAGitect.

This module provides processors for extracting text/markdown from various sources.
"""

from ragitect.services.processor.base import BaseDocumentProcessor
from ragitect.services.processor.simple import SimpleProcessor
from ragitect.services.processor.web_url_processor import (
    ContentExtractionError,
    URLFetchError,
    WebURLProcessor,
)

__all__ = [
    "BaseDocumentProcessor",
    "ContentExtractionError",
    "SimpleProcessor",
    "URLFetchError",
    "WebURLProcessor",
]