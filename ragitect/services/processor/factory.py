"""
Document Processor Factory

Selects the appropriate document processor based on file type.
Uses a simple two-processor strategy:
- SimpleProcessor for text-based formats (lightweight, fast)
- DoclingProcessor for complex formats (PDF, DOCX, etc.)
"""

from ragitect.services.processor.base import BaseDocumentProcessor
from ragitect.services.processor.simple import SimpleProcessor
from ragitect.services.processor.docling_processor import DoclingProcessor
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UnsupportedFormatError(Exception):
    """
    Raised when no processor supports the uploaded file format.

    This exception provides a clear signal to the UI layer that the file
    type is not supported, allowing for appropriate user feedback.
    """

    pass


class ProcessorFactory:
    """Factory for selecting appropriate document processor based on filetype"""

    def __init__(self):
        logger.info("Initializing ProcessorFactory")
        self.simple_processor: BaseDocumentProcessor = SimpleProcessor()
        self.docling_processor: BaseDocumentProcessor = DoclingProcessor()

        self.text_formats: list[str] = self.simple_processor.supported_formats()
        self.complex_formats: list[str] = self.docling_processor.supported_formats()

        logger.info(
            f"ProcessorFactory initialized - Text formats: {len(self.text_formats)}, Complex formats: {len(self.complex_formats)}"
        )

    def get_processor(self, file_name: str) -> BaseDocumentProcessor:
        """Select appropriate processor for given file

        Args:
            file_name: name of the uploaded file

        Returns:
            BaseDocumentProcessor: either SimpleProcessor or DoclingProcessor instance

        Raises:
            UnsupportedFormatError: if the file extension is not supported by any processor
        """
        ext = Path(file_name).suffix.lower()

        if ext in self.text_formats:
            logger.info(f"Selected SimpleProcessor for file {file_name}")
            return self.simple_processor

        if ext in self.complex_formats:
            logger.info(f"Selected DoclingProcessor for file {file_name}")
            return self.docling_processor

        all_formats = sorted(self.text_formats + self.complex_formats)

        error_msg = (
            f"Unsupported format: {ext}\nSupported formats: {', '.join(all_formats)}"
        )

        logger.warning(f"Unsupported file format attempted: {ext} (file: {file_name})")
        raise UnsupportedFormatError(error_msg)
