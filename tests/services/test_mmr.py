"""Tests for Maximum Marginal Relevance (MMR) diversity selection."""

import numpy as np
import pytest


class TestMMRSelection:
    """Tests for ragitect.services.mmr module."""

    def test_mmr_empty_chunks_returns_empty(self):
        """MMR with empty input should return empty list."""
        from ragitect.services.mmr import mmr_select

        result = mmr_select(
            query_embedding=[0.1, 0.2, 0.3],
            chunk_embeddings=[],
            chunks=[],
            k=5,
        )
        assert result == []

    def test_mmr_returns_k_chunks(self):
        """MMR should return exactly k chunks when more are available."""
        from ragitect.services.mmr import mmr_select

        # Create 10 chunks with different embeddings
        chunks = [{"content": f"chunk {i}", "id": i} for i in range(10)]
        chunk_embeddings = [np.random.rand(3).tolist() for _ in range(10)]
        query_embedding = np.random.rand(3).tolist()

        result = mmr_select(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            k=5,
        )

        assert len(result) == 5

    def test_mmr_returns_all_when_fewer_than_k(self):
        """MMR should return all chunks if fewer than k available."""
        from ragitect.services.mmr import mmr_select

        chunks = [{"content": "chunk 1"}, {"content": "chunk 2"}]
        chunk_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        query_embedding = [0.2, 0.3]

        result = mmr_select(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            k=10,
        )

        assert len(result) == 2

    def test_mmr_lambda_1_pure_relevance(self):
        """With lambda=1.0, should select purely by relevance (no diversity)."""
        from ragitect.services.mmr import mmr_select

        # Query embedding
        query = [1.0, 0.0, 0.0]

        # Chunk 0: most similar to query
        # Chunk 1: less similar
        # Chunk 2: least similar
        chunk_embeddings = [
            [0.9, 0.1, 0.0],  # Very similar to query
            [0.5, 0.5, 0.0],  # Moderately similar
            [0.0, 0.0, 1.0],  # Very different
        ]
        chunks = [{"id": 0}, {"id": 1}, {"id": 2}]

        result = mmr_select(
            query_embedding=query,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            k=2,
            lambda_param=1.0,  # Pure relevance
        )

        # Should select by pure relevance: chunk 0, then chunk 1
        assert result[0]["id"] == 0
        assert result[1]["id"] == 1

    def test_mmr_lambda_0_pure_diversity(self):
        """With lambda=0.0, should maximize diversity (minimize redundancy)."""
        from ragitect.services.mmr import mmr_select

        query = [1.0, 0.0]

        # Three chunks: two very similar, one very different
        chunk_embeddings = [
            [0.9, 0.1],  # Similar to query
            [0.85, 0.15],  # Very similar to chunk 0
            [0.0, 1.0],  # Very different from chunk 0
        ]
        chunks = [{"id": 0}, {"id": 1}, {"id": 2}]

        result = mmr_select(
            query_embedding=query,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            k=2,
            lambda_param=0.0,  # Pure diversity
        )

        # First should be most relevant (chunk 0 or 1)
        # Second should be most diverse from first (chunk 2)
        selected_ids = {r["id"] for r in result}
        assert 2 in selected_ids  # Chunk 2 should be included for diversity

    def test_mmr_preserves_chunk_data(self):
        """Original chunk data should be preserved."""
        from ragitect.services.mmr import mmr_select

        chunks = [
            {
                "content": "test",
                "document_name": "doc.pdf",
                "metadata": {"key": "value"},
            }
        ]
        chunk_embeddings = [[0.5, 0.5]]
        query_embedding = [0.5, 0.5]

        result = mmr_select(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            k=1,
        )

        assert result[0]["content"] == "test"
        assert result[0]["document_name"] == "doc.pdf"
        assert result[0]["metadata"] == {"key": "value"}

    def test_mmr_default_lambda(self):
        """Default lambda should be 0.7 (70% relevance, 30% diversity)."""
        from ragitect.services.mmr import mmr_select

        # Just verify it runs with default lambda
        chunks = [{"content": f"chunk {i}"} for i in range(5)]
        chunk_embeddings = [np.random.rand(3).tolist() for _ in range(5)]
        query_embedding = np.random.rand(3).tolist()

        # Should not raise and return results
        result = mmr_select(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            k=3,
        )

        assert len(result) == 3


class TestMMREdgeCases:
    """Edge case tests for MMR selection."""

    def test_mmr_single_chunk(self):
        """Single chunk should be returned as-is."""
        from ragitect.services.mmr import mmr_select

        chunks = [{"content": "only one"}]
        chunk_embeddings = [[0.5, 0.5]]
        query_embedding = [0.5, 0.5]

        result = mmr_select(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            k=5,
        )

        assert len(result) == 1
        assert result[0]["content"] == "only one"

    def test_mmr_identical_embeddings(self):
        """Should handle identical embeddings gracefully."""
        from ragitect.services.mmr import mmr_select

        # All chunks have identical embeddings
        chunks = [{"id": i} for i in range(5)]
        chunk_embeddings = [[0.5, 0.5, 0.5]] * 5
        query_embedding = [0.5, 0.5, 0.5]

        # Should not crash
        result = mmr_select(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            k=3,
        )

        assert len(result) == 3
