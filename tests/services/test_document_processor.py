"""Tests for document_processor.py"""

import pytest
from langchain_core.documents import Document

from ragitect.services.document_processor import (
    create_documents,
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
        chunks = split_document(
            markdown_text, chunk_size=1000, overlap=150, file_type=".md"
        )

        # Verify chunks were created
        assert len(chunks) > 0

        # Verify at least one chunk contains a header (structure preserved)
        has_header = any("#" in chunk for chunk in chunks)
        assert has_header

    def test_markdown_splitting_respects_chunk_size(self):
        """Test that markdown splitting still enforces size limits"""
        # Create a large markdown section
        large_section = "# Big Section\n\n" + ("word " * 500)
        chunks = split_document(
            large_section, chunk_size=1000, overlap=150, file_type=".md"
        )

        # Should be split into multiple chunks due to size
        assert len(chunks) >= 2
        # Each chunk should be roughly within size limits (with some tolerance)
        for chunk in chunks:
            assert len(chunk) <= 1200  # Allow tolerance for overlap

    def test_plain_text_uses_recursive_splitter(self):
        """Test that non-markdown files use standard recursive splitting"""
        text = "a" * 2000
        chunks = split_document(text, chunk_size=1000, overlap=150, file_type=".txt")

        # Should split into multiple chunks
        assert len(chunks) >= 2

        # Test with no file_type (default behavior)
        chunks_default = split_document(text, chunk_size=1000, overlap=150)
        assert len(chunks_default) >= 2

    def test_txt_files_use_markdown_aware_splitting(self):
        """Test that .txt files with markdown content benefit from structure awareness"""
        markdown_like_text = """# Title
Content here.

## Section
More content.
"""
        chunks = split_document(markdown_like_text, file_type=".txt")
        assert len(chunks) > 0

    def test_markdown_splitting_fallback_on_error(self):
        """Test that markdown splitting falls back to recursive on parsing errors"""
        # Normal text without markdown structure should still work
        plain_text = "This is just plain text without any markdown structure. " * 50
        chunks = split_document(
            plain_text, chunk_size=1000, overlap=150, file_type=".md"
        )

        # Should still produce chunks via fallback
        assert len(chunks) > 0
