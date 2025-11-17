import logging
from typing import override

from ragitect.services.processor.base import BaseDocumentProcessor

logger = logging.getLogger(__name__)


class SimpleProcessor(BaseDocumentProcessor):
    """Handles plain text formats (TXT, Markdown) without external libs"""

    @override
    def supported_formats(self) -> list[str]:
        """support text based format"""
        return [".txt", ".md", ".markdown"]

    @override
    def process(self, file_bytes: bytes, file_name: str) -> str:
        """simple utf-8 decode with error handling

        Args:
            file_bytes: raw file bytes
            file_name: filename

        Returns:
            decoded text string
        """
        logger.info(
            f"Processing file {file_name} with builtin processor for txt and md files"
        )
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("utf-8", errors="replace")
