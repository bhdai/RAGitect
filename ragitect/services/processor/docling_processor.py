import logging
import os
import tempfile
from pathlib import Path
from typing import override

from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from ragitect.services.processor.base import BaseDocumentProcessor

logger = logging.getLogger(__name__)


class DoclingProcessor(BaseDocumentProcessor):
    """Handles complex documents using Docling library."""

    def __init__(self):
        """Initialize Docling converter"""
        self.converter = DocumentConverter()

    @override
    def supported_formats(self) -> list[str]:
        return [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"]

    @override
    def process(self, file_bytes: bytes, file_name: str) -> str:
        """Convert document to markdown using Docling."""

        logger.info(f"Processing file {file_name} with Docling processor")
        suffix = Path(file_name).suffix

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, delete=False
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            result = self.converter.convert(tmp_path)

            content = self._extract_content(result)
            return content

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_content(self, result: ConversionResult) -> str:
        """
        Extract content with fallback methods for version compatibility.

        Tries multiple export methods in order of preference.
        """
        if not hasattr(result, "document") or not result.document:
            raise ValueError("No document object returned by Docling")

        # try different export methods
        export_methods = [
            ("export_to_markdown", lambda doc: doc.export_to_markdown()),
            ("to_markdown", lambda doc: doc.to_markdown()),
            ("text", lambda doc: doc.text),
            ("__str__", lambda doc: str(doc)),
        ]

        for method_name, extractor in export_methods:
            if hasattr(result.document, method_name):
                try:
                    content = extractor(result.document)
                    if content:
                        return content
                except Exception:
                    continue  # try next method

        raise ValueError("Could not extract content using any available method")
