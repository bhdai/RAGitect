"""Tests for the cross-encoder reranker service."""

import pytest


class TestRerankerService:
    """Tests for ragitect.services.reranker module."""

    pytestmark = pytest.mark.asyncio


class TestRerankerService:
    """Tests for ragitect.services.reranker module."""

    pytestmark = pytest.mark.asyncio

    @pytest.fixture(autouse=True)
    def mock_cross_encoder(self):
        """Mock the CrossEncoder class to avoid loading real models."""
        from unittest.mock import MagicMock, patch

        import numpy as np

        with patch("ragitect.services.reranker.CrossEncoder") as mock:
            # Setup default mock behavior
            mock_instance = MagicMock()
            # Predict returns numpy array
            mock_instance.predict.side_effect = lambda pairs: np.array(
                [0.9 - (i * 0.1) for i in range(len(pairs))]
            )
            mock.return_value = mock_instance
            yield mock

    async def test_rerank_empty_chunks_returns_empty_list(self):
        """Reranking empty list should return empty list."""
        from ragitect.services.reranker import rerank_chunks

        result = await rerank_chunks("test query", [])
        assert result == []

    async def test_rerank_adds_rerank_score_to_chunks(self, mock_cross_encoder):
        """Each chunk should have rerank_score after reranking."""
        # Reset singleton to ensure mock is used
        import ragitect.services.reranker
        from ragitect.services.reranker import get_reranker, rerank_chunks

        ragitect.services.reranker._reranker_instance = None

        chunks = [
            {"content": "Python is a programming language.", "id": "1"},
            {"content": "JavaScript runs in the browser.", "id": "2"},
            {"content": "Python has excellent machine learning libraries.", "id": "3"},
        ]

        result = await rerank_chunks("What is Python?", chunks, top_k=3)

        assert len(result) == 3
        for chunk in result:
            assert "rerank_score" in chunk
            assert isinstance(chunk["rerank_score"], float)

        # Verify predict was called
        mock_instance = get_reranker()
        assert mock_instance.predict.called

    async def test_rerank_orders_by_relevance(self, mock_cross_encoder):
        """Chunks should be ordered by relevance to query (highest first)."""
        import numpy as np

        # Reset singleton
        import ragitect.services.reranker
        from ragitect.services.reranker import get_reranker, rerank_chunks

        ragitect.services.reranker._reranker_instance = None

        chunks = [
            {"content": "The weather is nice today.", "id": "1"},
            {"content": "Python is great for data science and ML.", "id": "2"},
            {"content": "Cooking recipes for beginners.", "id": "3"},
        ]

        # Setup mock to return specific scores to force ordering
        mock_instance = get_reranker()
        mock_instance.predict.side_effect = None
        mock_instance.predict.return_value = np.array([0.1, 0.9, 0.5])

        result = await rerank_chunks("Python machine learning", chunks, top_k=3)

        # Python ML chunk (index 1, score 0.9) should be ranked first
        assert result[0]["id"] == "2"
        assert result[0]["rerank_score"] == 0.9

        # Cooking (index 2, score 0.5) second
        assert result[1]["id"] == "3"
        assert result[1]["rerank_score"] == 0.5

        # Weather (index 0, score 0.1) last
        assert result[2]["id"] == "1"
        assert result[2]["rerank_score"] == 0.1

    async def test_rerank_respects_top_k(self):
        """Should return only top_k results."""
        # Reset singleton
        import ragitect.services.reranker
        from ragitect.services.reranker import rerank_chunks

        ragitect.services.reranker._reranker_instance = None

        chunks = [
            {"content": f"Document {i} content.", "id": str(i)} for i in range(10)
        ]

        result = await rerank_chunks("Document content", chunks, top_k=5)

        assert len(result) == 5

    async def test_rerank_preserves_original_chunk_data(self):
        """Original chunk data should be preserved after reranking."""
        # Reset singleton
        import ragitect.services.reranker
        from ragitect.services.reranker import rerank_chunks

        ragitect.services.reranker._reranker_instance = None

        chunks = [
            {
                "content": "Test content",
                "document_name": "test.pdf",
                "chunk_index": 3,
                "custom_field": "custom_value",
            }
        ]

        result = await rerank_chunks("test", chunks, top_k=1)

        assert result[0]["content"] == "Test content"
        assert result[0]["document_name"] == "test.pdf"
        assert result[0]["chunk_index"] == 3
        assert result[0]["custom_field"] == "custom_value"


class TestRerankerSingleton:
    """Tests for reranker model singleton pattern."""

    def test_get_reranker_returns_cross_encoder(self):
        """get_reranker should return a CrossEncoder instance."""
        from unittest.mock import patch

        # Reset singleton
        import ragitect.services.reranker
        from ragitect.services.reranker import get_reranker

        ragitect.services.reranker._reranker_instance = None

        with patch("ragitect.services.reranker.CrossEncoder") as mock_cls:
            reranker = get_reranker()
            assert reranker == mock_cls.return_value
            mock_cls.assert_called_once()

    def test_get_reranker_returns_same_instance(self):
        """Singleton pattern: same instance returned on multiple calls."""
        from unittest.mock import patch

        # Reset singleton
        import ragitect.services.reranker
        from ragitect.services.reranker import get_reranker

        ragitect.services.reranker._reranker_instance = None

        with patch("ragitect.services.reranker.CrossEncoder") as mock_cls:
            reranker1 = get_reranker()
            reranker2 = get_reranker()

            assert reranker1 is reranker2
            # Should only be initialized once
            mock_cls.assert_called_once()
