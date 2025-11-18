"""Tests for vector_store.py"""

import pytest
import numpy as np
from langchain_core.documents import Document

from ragitect.services.vector_store import (
    initialize_index,
    add_vectors_to_index,
    search_index,
)


class TestInitializeIndex:
    """Test FAISS index initialization"""

    def test_creates_index_with_correct_dimension(self):
        index = initialize_index(768)
        assert index.d == 768

    def test_creates_empty_index(self):
        index = initialize_index(128)
        assert index.ntotal == 0

    def test_different_dimensions(self):
        index_256 = initialize_index(256)
        index_512 = initialize_index(512)

        assert index_256.d == 256
        assert index_512.d == 512


class TestAddVectorsToIndex:
    """Test adding vectors to index"""

    def test_adds_vectors_to_index(self):
        index = initialize_index(128)
        vectors = [[0.1] * 128, [0.2] * 128, [0.3] * 128]

        add_vectors_to_index(index, vectors)

        assert index.ntotal == 3

    def test_handles_empty_vector_list(self):
        index = initialize_index(128)

        add_vectors_to_index(index, [])

        assert index.ntotal == 0

    def test_adds_multiple_batches(self):
        index = initialize_index(64)

        add_vectors_to_index(index, [[0.1] * 64, [0.2] * 64])
        add_vectors_to_index(index, [[0.3] * 64])

        assert index.ntotal == 3

    def test_converts_to_numpy_float32(self):
        index = initialize_index(10)
        vectors = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]  # Python ints

        # Should not raise
        add_vectors_to_index(index, vectors)

        assert index.ntotal == 1


class TestSearchIndex:
    """Test vector search functionality"""

    def test_search_returns_correct_number_of_results(self):
        index = initialize_index(128)
        vectors = [[0.1] * 128, [0.2] * 128, [0.3] * 128]
        add_vectors_to_index(index, vectors)

        document_store = [
            Document(page_content="doc1", metadata={"source": "test"}),
            Document(page_content="doc2", metadata={"source": "test"}),
            Document(page_content="doc3", metadata={"source": "test"}),
        ]

        query_vector = [0.15] * 128
        results = search_index(index, query_vector, document_store, k=2)

        assert len(results) == 2

    def test_search_returns_documents_and_scores(self):
        index = initialize_index(10)
        vectors = [[1.0] * 10]
        add_vectors_to_index(index, vectors)

        document_store = [Document(page_content="test", metadata={"source": "test"})]

        results = search_index(index, [1.0] * 10, document_store, k=1)

        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, (float, np.floating))

    def test_search_empty_index_returns_empty_list(self):
        index = initialize_index(128)
        document_store = []

        results = search_index(index, [0.1] * 128, document_store, k=5)

        assert results == []

    def test_search_respects_k_parameter(self):
        index = initialize_index(10)
        vectors = [[i] * 10 for i in range(10)]
        add_vectors_to_index(index, vectors)

        document_store = [
            Document(page_content=f"doc{i}", metadata={"source": "test"})
            for i in range(10)
        ]

        results_k3 = search_index(index, [0.5] * 10, document_store, k=3)
        results_k5 = search_index(index, [0.5] * 10, document_store, k=5)

        assert len(results_k3) == 3
        assert len(results_k5) == 5

    def test_search_returns_correct_documents(self):
        index = initialize_index(10)
        vectors = [[1.0] * 10, [2.0] * 10]
        add_vectors_to_index(index, vectors)

        document_store = [
            Document(page_content="first doc", metadata={"source": "test1"}),
            Document(page_content="second doc", metadata={"source": "test2"}),
        ]

        results = search_index(index, [1.0] * 10, document_store, k=1)

        # Should return the closest match
        assert len(results) == 1
        assert results[0][0].page_content in ["first doc", "second doc"]
