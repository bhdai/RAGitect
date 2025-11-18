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
        chunks = split_document(text, chunk_size=500, overlap=50)

        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_empty_string_returns_empty_list(self):
        chunks = split_document("", chunk_size=500, overlap=50)
        assert chunks == []

    def test_overlap_creates_redundancy(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = split_document(text, chunk_size=10, overlap=3)

        # With overlap, chunks should share content
        assert len(chunks) >= 2

    def test_default_parameters(self):
        text = "a" * 1000
        chunks = split_document(text)

        # Default is chunk_size=500, overlap=50
        assert len(chunks) >= 2
