import logging
import os
import tempfile
from pathlib import Path
from typing import override

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from ragitect.services.processor.base import BaseDocumentProcessor

logger = logging.getLogger(__name__)


class DoclingProcessor(BaseDocumentProcessor):
    """Handles complex documents using Docling library with graceful error handling."""

    def __init__(self):
        """Initialize Docling converter with version-safe configuration"""
        self._initialize_docling()

    def _initialize_docling(self):
        """Initialize Docling with version-safe configuration and error handling."""
        try:
            logger.info("🔧 Initializing Docling with version-safe configuration...")

            accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

            # create pipeline options with version-safe attribute checking
            pipeline_options = PdfPipelineOptions(
                accelerator_options=accelerator_options
            )

            if hasattr(pipeline_options, "do_ocr"):
                pipeline_options.do_ocr = False
                logger.info("OCR disabled for performance")
            else:
                logger.info("OCR attribute not available in this Docling version")

            if hasattr(pipeline_options, "do_table_structure"):
                pipeline_options.do_table_structure = True
                logger.info("Table structure detection enabled")

            pdf_format_option = PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )

            # initialize DocumentConverter
            self.converter = DocumentConverter(
                format_options={InputFormat.PDF: pdf_format_option}
            )

            logger.info("Docling initialized successfully")

        except ImportError as e:
            logger.error(f"Docling not installed: {e}")
            raise RuntimeError(f"Docling not available: {e}") from e
        except Exception as e:
            logger.error(f"Docling initialization failed: {e}")
            raise RuntimeError(f"Docling initialization failed: {e}") from e

    @override
    def supported_formats(self) -> list[str]:
        return [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"]

    @override
    def process(self, file_bytes: bytes, file_name: str) -> str:
        """Convert document to markdown using Docling with component-level error handling."""

        logger.info(
            f"Processing {file_name} with Docling processor (size: {len(file_bytes)} bytes)"
        )
        suffix = Path(file_name).suffix

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, delete=False
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            logger.info(f"Starting Docling conversion for {file_name}...")

            # convert document - Docling handles component-level failures internally
            result = self.converter.convert(tmp_path)

            if hasattr(result, "status"):
                logger.info(f"Conversion status: {result.status}")

            content = self._extract_content(result, file_name)

            if not content or not content.strip():
                raise ValueError(
                    f"No content could be extracted from {file_name}. "
                    + "The document may be empty, corrupted, or image-only."
                )

            logger.info(
                f"Docling SUCCESS - {file_name}: {len(content)} characters extracted"
            )
            return content

        except Exception as e:
            logger.error(
                f"Docling processing failed for {file_name}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Could not process {file_name}. This may be due to: "
                + "(1) corrupted file, (2) unsupported PDF features, or (3) processing error. "
                + f"Error: {e}"
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file for {file_name}")

    def _extract_content(self, result: ConversionResult, file_name: str) -> str:
        """
        Extract content with graceful fallback methods (version-safe).

        Implements component-level processing - tries multiple extraction methods
        to get as much content as possible even if some components fail.
        """
        if not hasattr(result, "document") or not result.document:
            logger.error("No document object in conversion result")
            raise ValueError("No document object returned by Docling")

        doc = result.document

        export_methods = [
            (
                "export_to_markdown",
                lambda: doc.export_to_markdown(),
                "export_to_markdown method",
            ),
            ("to_markdown", lambda: doc.to_markdown(), "to_markdown method"),
            ("text", lambda: doc.text, "text property"),
            ("__str__", lambda: str(doc), "string conversion"),
        ]

        for method_name, extractor, description in export_methods:
            if hasattr(doc, method_name):
                try:
                    logger.info(f"Attempting extraction via {description}...")
                    content = extractor()
                    if content and content.strip():
                        logger.info(f"Used {description} for {file_name}")
                        return content
                    else:
                        logger.info(f"{description} returned empty content")
                except Exception as e:
                    logger.info(f"{description} failed: {e}")
                    continue  # try next method
            else:
                logger.info(f"Method {method_name} not available")

        # Last resort: try to extract from pages or elements directly
        logger.warning(
            f"All standard extraction methods failed for {file_name}, attempting component-level extraction"
        )
        try:
            if hasattr(doc, "pages") and doc.pages:
                # Extract text from individual pages (graceful degradation)
                page_texts = []
                for i, page in enumerate(doc.pages):
                    try:
                        if hasattr(page, "text") and page.text:
                            page_texts.append(page.text)
                        elif hasattr(page, "export_to_markdown"):
                            page_texts.append(page.export_to_markdown())
                    except Exception as e:
                        logger.warning(f"Failed to extract page {i + 1}: {e}")
                        continue  # skip failed pages, continue with successful ones

                if page_texts:
                    content = "\n\n".join(filter(None, page_texts))
                    if content.strip():
                        logger.info(
                            f"Extracted content from {len(page_texts)}/{len(doc.pages)} pages"
                        )
                        return content
        except Exception as e:
            logger.error(f"Component-level extraction failed: {e}")

        raise ValueError(
            "Could not extract content using any available method. "
            + "The document may be empty, image-only, or severely corrupted."
        )
