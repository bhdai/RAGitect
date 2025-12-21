"""Tests for Adaptive-K selection based on score gap detection."""

import pytest


class TestAdaptiveKSelection:
    """Tests for ragitect.services.adaptive_k module."""

    def test_adaptive_k_empty_chunks_returns_empty(self):
        """Empty input should return empty list."""
        from ragitect.services.adaptive_k import select_adaptive_k

        chunks, metadata = select_adaptive_k([])

        assert chunks == []
        assert metadata["adaptive_k"] == 0

    def test_adaptive_k_returns_metadata(self):
        """Should return metadata dict with adaptive_k info."""
        from ragitect.services.adaptive_k import select_adaptive_k

        chunks = [
            {"content": "chunk1", "rerank_score": 0.9},
            {"content": "chunk2", "rerank_score": 0.8},
            {"content": "chunk3", "rerank_score": 0.7},
        ]

        result, metadata = select_adaptive_k(chunks)

        assert "adaptive_k" in metadata
        assert "gap_found" in metadata
        assert isinstance(metadata["adaptive_k"], int)

    def test_adaptive_k_respects_k_min(self):
        """Should return at least k_min chunks."""
        from ragitect.services.adaptive_k import select_adaptive_k

        chunks = [
            {"content": f"chunk{i}", "rerank_score": 1.0 - i * 0.05} for i in range(20)
        ]

        result, metadata = select_adaptive_k(chunks, k_min=6, k_max=16)

        assert len(result) >= 6

    def test_adaptive_k_respects_k_max(self):
        """Should return at most k_max chunks."""
        from ragitect.services.adaptive_k import select_adaptive_k

        # All chunks with similar scores (no clear gap)
        chunks = [
            {"content": f"chunk{i}", "rerank_score": 0.9 - i * 0.01} for i in range(30)
        ]

        result, metadata = select_adaptive_k(chunks, k_min=4, k_max=12)

        assert len(result) <= 12

    def test_adaptive_k_finds_score_gap(self):
        """Should find gap in score distribution and cut there."""
        from ragitect.services.adaptive_k import select_adaptive_k

        # Clear gap after position 5 (between 0.6 and 0.3)
        chunks = [
            {"content": "chunk1", "rerank_score": 0.95},
            {"content": "chunk2", "rerank_score": 0.90},
            {"content": "chunk3", "rerank_score": 0.85},
            {"content": "chunk4", "rerank_score": 0.80},
            {"content": "chunk5", "rerank_score": 0.75},  # Last high-relevance
            {"content": "chunk6", "rerank_score": 0.40},  # Big gap here (0.35)
            {"content": "chunk7", "rerank_score": 0.35},
            {"content": "chunk8", "rerank_score": 0.30},
        ]

        result, metadata = select_adaptive_k(
            chunks, k_min=4, k_max=12, gap_threshold=0.15
        )

        # Should cut at position 5 due to 0.35 gap
        assert len(result) == 5
        assert metadata["gap_found"] is True
        assert metadata["adaptive_k"] == 5

    def test_adaptive_k_returns_all_when_few_chunks(self):
        """When fewer chunks than k_min, return all."""
        from ragitect.services.adaptive_k import select_adaptive_k

        chunks = [
            {"content": "chunk1", "rerank_score": 0.9},
            {"content": "chunk2", "rerank_score": 0.8},
        ]

        result, metadata = select_adaptive_k(chunks, k_min=4, k_max=12)

        assert len(result) == 2
        assert metadata["gap_found"] is False

    def test_adaptive_k_default_score_key(self):
        """Default score key should be 'rerank_score'."""
        from ragitect.services.adaptive_k import select_adaptive_k

        chunks = [
            {"content": "chunk1", "rerank_score": 0.9},
            {"content": "chunk2", "rerank_score": 0.8},
        ]

        # Should not raise with default score_key
        result, metadata = select_adaptive_k(chunks)
        assert len(result) == 2

    def test_adaptive_k_custom_score_key(self):
        """Should support custom score key."""
        from ragitect.services.adaptive_k import select_adaptive_k

        chunks = [
            {"content": "chunk1", "custom_score": 0.9},
            {"content": "chunk2", "custom_score": 0.8},
            {"content": "chunk3", "custom_score": 0.7},
        ]

        result, metadata = select_adaptive_k(chunks, score_key="custom_score")
        assert len(result) == 3

    def test_adaptive_k_preserves_chunk_data(self):
        """Original chunk data should be preserved."""
        from ragitect.services.adaptive_k import select_adaptive_k

        chunks = [
            {
                "content": "test content",
                "document_name": "doc.pdf",
                "rerank_score": 0.9,
                "metadata": {"key": "value"},
            }
        ]

        result, metadata = select_adaptive_k(chunks)

        assert result[0]["content"] == "test content"
        assert result[0]["document_name"] == "doc.pdf"
        assert result[0]["metadata"] == {"key": "value"}

    def test_adaptive_k_no_gap_uses_k_max(self):
        """When no significant gap found, should use k_max."""
        from ragitect.services.adaptive_k import select_adaptive_k

        # Uniform score distribution (no gaps)
        chunks = [
            {"content": f"chunk{i}", "rerank_score": 0.9 - i * 0.02} for i in range(20)
        ]

        result, metadata = select_adaptive_k(
            chunks, k_min=4, k_max=10, gap_threshold=0.20
        )

        assert len(result) == 10
        assert metadata["gap_found"] is False


class TestAdaptiveKDefaults:
    """Tests for default parameter values."""

    def test_default_k_min_is_4(self):
        """Default k_min should be 4."""
        from ragitect.services.adaptive_k import select_adaptive_k

        # With only 4 chunks and no gap, should return all 4
        chunks = [
            {"content": f"chunk{i}", "rerank_score": 0.9 - i * 0.01} for i in range(4)
        ]

        result, metadata = select_adaptive_k(chunks)
        assert len(result) == 4

    def test_default_k_max_is_16(self):
        """Default k_max should be 16."""
        from ragitect.services.adaptive_k import select_adaptive_k

        # With many chunks and no gap, should cap at 16
        chunks = [
            {"content": f"chunk{i}", "rerank_score": 0.9 - i * 0.01} for i in range(30)
        ]

        result, metadata = select_adaptive_k(chunks)
        assert len(result) <= 16
