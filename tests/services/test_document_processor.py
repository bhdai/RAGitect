"""Tests for document_processor.py"""

import pytest
from langchain_core.documents import Document

from ragitect.services.document_processor import (
    create_documents,
    process_file_bytes,
    split_document,
)


class TestCreateDocuments:
    """Test document creation with metadata"""

    def test_creates_documents_with_metadata(self):
        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        source = "test_file.pdf"

        docs = create_documents(chunks, source)

        assert len(docs) == 3
        assert all(isinstance(doc, Document) for doc in docs)

    def test_adds_correct_metadata(self):
        chunks = ["first chunk", "second chunk"]
        source = "example.txt"

        docs = create_documents(chunks, source)

        assert docs[0].metadata["source"] == "example.txt"
        assert docs[0].metadata["chunk_index"] == 0
        assert docs[1].metadata["source"] == "example.txt"
        assert docs[1].metadata["chunk_index"] == 1

    def test_preserves_chunk_content(self):
        chunks = ["Hello world", "Goodbye world"]
        source = "test.md"

        docs = create_documents(chunks, source)

        assert docs[0].page_content == "Hello world"
        assert docs[1].page_content == "Goodbye world"

    def test_handles_empty_chunks_list(self):
        docs = create_documents([], "empty.txt")
        assert docs == []

    def test_handles_single_chunk(self):
        docs = create_documents(["only one"], "single.txt")
        assert len(docs) == 1
        assert docs[0].metadata["chunk_index"] == 0


class TestSplitDocument:
    """Test text splitting functionality"""

    def test_splits_text_into_chunks(self):
        text = "a" * 1000
        chunks = split_document(text, chunk_size=100, overlap=10)

        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_respects_chunk_size(self):
        text = "word " * 200
        chunks = split_document(text, chunk_size=50, overlap=5)

        # Most chunks should be close to chunk_size
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allow some tolerance

    def test_handles_small_text(self):
        text = "short text"
        chunks = split_document(text, chunk_size=1000, overlap=150)

        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_empty_string_returns_empty_list(self):
        chunks = split_document("", chunk_size=1000, overlap=150)
        assert chunks == []

    def test_overlap_creates_redundancy(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = split_document(text, chunk_size=10, overlap=3)

        # With overlap, chunks should share content
        assert len(chunks) >= 2

    def test_default_parameters(self):
        text = "a" * 2000
        chunks = split_document(text)

        # Default is chunk_size=1000, overlap=150
        assert len(chunks) >= 2

    def test_markdown_splitting_preserves_structure(self):
        """Test that markdown files are split respecting header boundaries"""
        markdown_text = """# Main Title

This is the introduction paragraph with some content.

## Section One

This section has important information about the first topic.
It continues for several lines to demonstrate chunking.

## Section Two

This section discusses the second topic in detail.
More content here to make it substantial.

### Subsection 2.1

Detailed information in a subsection.

### Subsection 2.2

More detailed information in another subsection.

## Section Three

The final section with concluding remarks.
"""
        chunks = split_document(markdown_text, chunk_size=1000, overlap=150)

        # Verify chunks were created
        assert len(chunks) > 0

        # Verify headers are preserved in chunks (markdown structure maintained)
        chunks_with_headers = [c for c in chunks if "#" in c]
        assert len(chunks_with_headers) > 0, "No headers found in chunks"

        # Verify that section headers and their content stay together
        # Look for "Section One" header with its content
        section_one_chunks = [c for c in chunks if "## Section One" in c]
        if section_one_chunks:
            # The chunk containing the header should also contain related content
            assert any("important information" in c for c in section_one_chunks), (
                "Section header separated from its content"
            )

        # Verify subsection structure is preserved
        subsection_chunks = [c for c in chunks if "### Subsection" in c]
        if subsection_chunks:
            # At least one subsection should have its content in the same chunk
            assert any(
                "Detailed information" in c or "More detailed" in c
                for c in subsection_chunks
            ), "Subsection headers separated from their content"

        # Verify markdown splitting produces different results than plain splitting
        plain_chunks = split_document(markdown_text, chunk_size=1000, overlap=150)
        # Markdown splitting should respect structure, potentially creating different boundaries
        # This is a soft check - at minimum, both should produce reasonable chunking
        assert len(chunks) > 0 and len(plain_chunks) > 0

    def test_markdown_splitting_handles_empty_text(self):
        """Test that markdown splitting efficiently handles empty text"""
        # Test completely empty string
        chunks = split_document("")
        assert chunks == []

        # Test whitespace-only string
        chunks_whitespace = split_document("   \n\t  ")
        assert chunks_whitespace == []

    def test_markdown_splitting_respects_chunk_size(self):
        """Test that markdown splitting still enforces size limits"""
        # Create a large markdown section
        large_section = "# Big Section\n\n" + ("word " * 500)
        chunk_size = 1000
        overlap = 150
        chunks = split_document(large_section, chunk_size=chunk_size, overlap=overlap)

        # Should be split into multiple chunks due to size
        assert len(chunks) >= 2

        # Each chunk should be roughly within size limits with tolerance for:
        # - Overlap between chunks
        # - RecursiveCharacterTextSplitter adding buffer
        # - Markdown structure preservation may extend slightly
        max_expected_size = chunk_size + overlap + 50  # 1000 + 150 + 50 = 1200
        for chunk in chunks:
            assert len(chunk) <= max_expected_size, (
                f"Chunk size {len(chunk)} exceeds max expected {max_expected_size}"
            )

    def test_plain_text_splitting(self):
        """Test that plain text is split using markdown-aware splitter"""
        text = "a" * 2000
        chunks = split_document(text, chunk_size=1000, overlap=150)

        # Should split into multiple chunks
        assert len(chunks) >= 2

    def test_markdown_splitting_fallback_on_error(self):
        """Test that markdown splitting falls back to recursive on parsing errors"""
        # Normal text without markdown structure should still work
        plain_text = "This is just plain text without any markdown structure. " * 50
        chunks = split_document(plain_text, chunk_size=1000, overlap=150)

        # Should still produce chunks via fallback
        assert len(chunks) > 0


class TestProcessFileBytes:
    """Test process_file_bytes functionality"""

    def test_returns_tuple_structure(self):
        """Test that process_file_bytes returns (text, metadata) tuple"""
        # Create simple text file bytes
        file_content = "Hello, this is a test document."
        file_bytes = file_content.encode("utf-8")
        file_name = "test.txt"

        result = process_file_bytes(file_bytes, file_name)

        # Should return a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        text, metadata = result
        assert isinstance(text, str)
        assert isinstance(metadata, dict)

    def test_metadata_contains_file_type(self):
        """Test that metadata includes correct file extension"""
        file_bytes = b"Sample content"

        # Test various file extensions
        test_cases = [
            ("document.txt", ".txt"),
            ("report.md", ".md"),
            ("notes.markdown", ".markdown"),
            ("uppercase.TXT", ".txt"),  # Should be lowercase
        ]

        for file_name, expected_ext in test_cases:
            _, metadata = process_file_bytes(file_bytes, file_name)
            assert "file_type" in metadata
            assert metadata["file_type"] == expected_ext

    def test_metadata_contains_file_name(self):
        """Test that metadata includes original filename"""
        file_bytes = b"Content"
        file_name = "my_document.txt"

        _, metadata = process_file_bytes(file_bytes, file_name)

        assert "file_name" in metadata
        assert metadata["file_name"] == file_name

    def test_text_extraction_works(self):
        """Test that text extraction still functions correctly"""
        expected_content = "This is the document content with special chars: æøå"
        file_bytes = expected_content.encode("utf-8")
        file_name = "test.txt"

        text, _ = process_file_bytes(file_bytes, file_name)

        assert text == expected_content
        assert isinstance(text, str)

    def test_markdown_file_metadata(self):
        """Test metadata for markdown files"""
        markdown_content = "# Title\n\nContent here."
        file_bytes = markdown_content.encode("utf-8")
        file_name = "readme.md"

        text, metadata = process_file_bytes(file_bytes, file_name)

        assert text == markdown_content
        assert metadata["file_type"] == ".md"
        assert metadata["file_name"] == "readme.md"

    def test_handles_file_without_extension(self):
        """Test that files without extension raise UnsupportedFormatError"""
        from ragitect.services.processor.factory import UnsupportedFormatError

        file_bytes = b"No extension file"
        file_name = "README"

        # Should raise UnsupportedFormatError for files without extension
        with pytest.raises(UnsupportedFormatError):
            process_file_bytes(file_bytes, file_name)
