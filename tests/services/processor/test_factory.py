"""Tests for processor factory"""

import pytest

from ragitect.services.processor.factory import ProcessorFactory, UnsupportedFormatError
from ragitect.services.processor.simple import SimpleProcessor
from ragitect.services.processor.docling_processor import DoclingProcessor


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
