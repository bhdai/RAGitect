"""Tests for simple processor"""

from ragitect.services.processor.simple import SimpleProcessor


class TestSimpleProcessor:
    """Test simple text processor"""

    def setup_method(self):
        self.processor = SimpleProcessor()

    def test_supported_formats_includes_txt(self):
        formats = self.processor.supported_formats()
        assert ".txt" in formats

    def test_supported_formats_includes_md(self):
        formats = self.processor.supported_formats()
        assert ".md" in formats

    def test_supported_formats_includes_markdown(self):
        formats = self.processor.supported_formats()
        assert ".markdown" in formats

    def test_process_decodes_utf8(self):
        text = "Hello, World!"
        file_bytes = text.encode("utf-8")

        result = self.processor.process(file_bytes, "test.txt")

        assert result == "Hello, World!"

    def test_process_handles_unicode(self):
        text = "Hello 世界 🌍"
        file_bytes = text.encode("utf-8")

        result = self.processor.process(file_bytes, "test.txt")

        assert result == "Hello 世界 🌍"

    def test_process_handles_multiline(self):
        text = "Line 1\nLine 2\nLine 3"
        file_bytes = text.encode("utf-8")

        result = self.processor.process(file_bytes, "test.txt")

        assert result == "Line 1\nLine 2\nLine 3"
        assert result.count("\n") == 2

    def test_process_handles_empty_file(self):
        result = self.processor.process(b"", "empty.txt")
        assert result == ""

    def test_process_handles_decode_errors_gracefully(self):
        # Invalid UTF-8 sequence
        invalid_bytes = b"\x80\x81\x82"

        # Should not raise, uses 'replace' error handling
        result = self.processor.process(invalid_bytes, "test.txt")

        assert isinstance(result, str)
        # Replacement character should be present
        assert "\ufffd" in result or len(result) > 0

    def test_process_preserves_whitespace(self):
        text = "  spaces  \n\ttabs\t\n"
        file_bytes = text.encode("utf-8")

        result = self.processor.process(file_bytes, "test.txt")

        assert result == "  spaces  \n\ttabs\t\n"
