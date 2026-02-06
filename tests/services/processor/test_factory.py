"""Tests for processor factory"""

import pytest

from ragitect.services.processor.factory import ProcessorFactory, UnsupportedFormatError
from ragitect.services.processor.simple import SimpleProcessor
from ragitect.services.processor.docling_processor import DoclingProcessor
from ragitect.services.processor.web_url_processor import WebURLProcessor
from ragitect.services.processor.youtube_processor import YouTubeProcessor
from ragitect.services.processor.pdf_url_processor import PDFURLProcessor


class TestProcessorFactory:
    """Test processor factory selection logic"""

    def setup_method(self):
        self.factory = ProcessorFactory()

    def test_initialization_creates_processors(self):
        assert isinstance(self.factory.simple_processor, SimpleProcessor)
        assert isinstance(self.factory.docling_processor, DoclingProcessor)

    def test_loads_format_lists(self):
        assert len(self.factory.text_formats) > 0
        assert len(self.factory.complex_formats) > 0

    def test_selects_simple_processor_for_txt(self):
        processor = self.factory.get_processor("document.txt")
        assert isinstance(processor, SimpleProcessor)

    def test_selects_simple_processor_for_md(self):
        processor = self.factory.get_processor("readme.md")
        assert isinstance(processor, SimpleProcessor)

    def test_selects_simple_processor_for_markdown(self):
        processor = self.factory.get_processor("doc.markdown")
        assert isinstance(processor, SimpleProcessor)

    def test_selects_docling_processor_for_pdf(self):
        processor = self.factory.get_processor("report.pdf")
        assert isinstance(processor, DoclingProcessor)

    def test_selects_docling_processor_for_docx(self):
        processor = self.factory.get_processor("document.docx")
        assert isinstance(processor, DoclingProcessor)

    def test_selects_docling_processor_for_pptx(self):
        processor = self.factory.get_processor("slides.pptx")
        assert isinstance(processor, DoclingProcessor)

    def test_handles_uppercase_extension(self):
        processor = self.factory.get_processor("file.PDF")
        assert isinstance(processor, DoclingProcessor)

    def test_handles_mixed_case_extension(self):
        processor = self.factory.get_processor("file.TxT")
        assert isinstance(processor, SimpleProcessor)

    def test_raises_error_for_unsupported_format(self):
        with pytest.raises(UnsupportedFormatError):
            self.factory.get_processor("file.xyz")

    def test_raises_error_for_no_extension(self):
        with pytest.raises(UnsupportedFormatError):
            self.factory.get_processor("file_without_ext")

    def test_error_message_includes_format(self):
        with pytest.raises(UnsupportedFormatError, match=r"\.xyz"):
            self.factory.get_processor("test.xyz")

    def test_error_message_includes_supported_formats(self):
        with pytest.raises(UnsupportedFormatError, match=r"Supported formats"):
            self.factory.get_processor("test.invalid")

    def test_handles_filename_with_dots(self):
        processor = self.factory.get_processor("my.file.name.txt")
        assert isinstance(processor, SimpleProcessor)

    def test_handles_path_like_filename(self):
        processor = self.factory.get_processor("path/to/document.pdf")
        assert isinstance(processor, DoclingProcessor)


class TestProcessorFactoryURLRouting:
    """Test processor factory URL routing via source_type parameter (Story 5.5)"""

    def setup_method(self):
        self.factory = ProcessorFactory()

    # AC1: source_type="file" returns existing behavior (backward compatible)
    def test_source_type_file_uses_file_routing(self):
        """source_type='file' should use file extension routing"""
        processor = self.factory.get_processor("document.txt", source_type="file")
        assert isinstance(processor, SimpleProcessor)

    def test_source_type_file_for_pdf_returns_docling(self):
        """source_type='file' with .pdf should return DoclingProcessor"""
        processor = self.factory.get_processor("report.pdf", source_type="file")
        assert isinstance(processor, DoclingProcessor)

    # AC1: source_type="url" returns WebURLProcessor
    def test_source_type_url_returns_web_url_processor(self):
        """source_type='url' should return WebURLProcessor"""
        processor = self.factory.get_processor(
            "https://example.com/article", source_type="url"
        )
        assert isinstance(processor, WebURLProcessor)

    def test_source_type_url_ignores_file_extension(self):
        """source_type='url' should ignore .html extension and return WebURLProcessor"""
        processor = self.factory.get_processor(
            "https://example.com/page.html", source_type="url"
        )
        assert isinstance(processor, WebURLProcessor)

    # AC1: source_type="youtube" returns YouTubeProcessor
    def test_source_type_youtube_returns_youtube_processor(self):
        """source_type='youtube' should return YouTubeProcessor"""
        processor = self.factory.get_processor(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", source_type="youtube"
        )
        assert isinstance(processor, YouTubeProcessor)

    def test_source_type_youtube_with_short_url(self):
        """source_type='youtube' should work with youtu.be short URLs"""
        processor = self.factory.get_processor(
            "https://youtu.be/dQw4w9WgXcQ", source_type="youtube"
        )
        assert isinstance(processor, YouTubeProcessor)

    # AC1: source_type="pdf" returns PDFURLProcessor
    def test_source_type_pdf_returns_pdf_url_processor(self):
        """source_type='pdf' should return PDFURLProcessor"""
        processor = self.factory.get_processor(
            "https://arxiv.org/pdf/1706.03762.pdf", source_type="pdf"
        )
        assert isinstance(processor, PDFURLProcessor)

    def test_source_type_pdf_with_non_pdf_url(self):
        """source_type='pdf' should return PDFURLProcessor even for non-.pdf URLs"""
        processor = self.factory.get_processor(
            "https://example.com/document", source_type="pdf"
        )
        assert isinstance(processor, PDFURLProcessor)

    # AC1: Default source_type is "file" (backward compatibility)
    def test_default_source_type_is_file(self):
        """Default source_type should be 'file' for backward compatibility"""
        # Calling without source_type should behave like source_type="file"
        processor = self.factory.get_processor("document.txt")
        assert isinstance(processor, SimpleProcessor)

    def test_backward_compatible_with_existing_code(self):
        """Existing code without source_type should continue to work"""
        # These calls match existing tests, ensuring backward compatibility
        processor_txt = self.factory.get_processor("file.txt")
        processor_pdf = self.factory.get_processor("file.pdf")
        processor_docx = self.factory.get_processor("file.docx")

        assert isinstance(processor_txt, SimpleProcessor)
        assert isinstance(processor_pdf, DoclingProcessor)
        assert isinstance(processor_docx, DoclingProcessor)

    def test_url_processors_are_initialized(self):
        """URL processors should be initialized in ProcessorFactory"""
        assert hasattr(self.factory, "web_url_processor")
        assert hasattr(self.factory, "youtube_processor")
        assert hasattr(self.factory, "pdf_url_processor")
        assert isinstance(self.factory.web_url_processor, WebURLProcessor)
        assert isinstance(self.factory.youtube_processor, YouTubeProcessor)
        assert isinstance(self.factory.pdf_url_processor, PDFURLProcessor)
