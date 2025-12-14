"""Tests for iterative fallback query processing.

Tests for query_with_iterative_fallback function that implements
the router pattern with automatic fallback to reformulation.
"""

import pytest
from unittest.mock import AsyncMock, patch

from ragitect.services.query_service import query_with_iterative_fallback

pytestmark = [pytest.mark.asyncio]


class TestIterativeFallback:
    """Tests for query_with_iterative_fallback function."""

    # ========== SIMPLE QUERY - DIRECT SEARCH SUCCESS ==========

    async def test_simple_query_good_relevance_no_reformulation(self):
        """Test simple query with good relevance skips reformulation."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Relevant document content."])

        user_query = "What is FastAPI?"
        chat_history = []  # No history = simple query

        with patch(
            "ragitect.services.query_service._grade_retrieval_relevance",
            return_value=True,
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert results == ["Relevant document content."]
        assert metadata["used_reformulation"] is False
        assert metadata["classification"] == "simple"
        assert metadata["original_query"] == user_query
        assert metadata["final_query"] == user_query

    async def test_simple_query_bad_relevance_triggers_reformulation(self):
        """Test simple query with bad relevance triggers reformulation fallback."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(
            side_effect=[
                ["Irrelevant docs first try."],
                ["Better docs after reformulation."],
            ]
        )

        user_query = "What is FastAPI?"
        chat_history = []  # No history = simple query

        with (
            patch(
                "ragitect.services.query_service._grade_retrieval_relevance",
                return_value=False,
            ),
            patch(
                "ragitect.services.query_service.reformulate_query_with_chat_history",
                return_value="FastAPI Python web framework overview",
            ),
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert results == ["Better docs after reformulation."]
        assert metadata["used_reformulation"] is True
        assert metadata["classification"] == "simple"
        assert metadata["final_query"] == "FastAPI Python web framework overview"

    # ========== AMBIGUOUS QUERY - DIRECT REFORMULATION ==========

    async def test_ambiguous_query_reformulates_directly(self):
        """Test ambiguous query (with pronouns) reformulates directly."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Retrieved documents."])

        user_query = "How do I install it?"
        chat_history = [{"role": "user", "content": "Tell me about FastAPI"}]

        with patch(
            "ragitect.services.query_service.reformulate_query_with_chat_history",
            return_value="How do I install FastAPI?",
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert metadata["used_reformulation"] is True
        assert metadata["classification"] == "ambiguous"
        assert metadata["final_query"] == "How do I install FastAPI?"

    async def test_ambiguous_query_context_reference(self):
        """Test ambiguous query with context reference reformulates."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Retrieved documents."])

        user_query = "Show me the previous example again"
        chat_history = [{"role": "user", "content": "Show me async code example"}]

        with patch(
            "ragitect.services.query_service.reformulate_query_with_chat_history",
            return_value="Show async code example in Python",
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert metadata["used_reformulation"] is True
        assert metadata["classification"] == "ambiguous"

    # ========== COMPLEX QUERY - DIRECT REFORMULATION ==========

    async def test_complex_query_reformulates_directly(self):
        """Test complex query (comparison) reformulates directly."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Comparison documents."])

        user_query = "Compare FastAPI vs Flask"
        chat_history = []

        with patch(
            "ragitect.services.query_service.reformulate_query_with_chat_history",
            return_value="FastAPI vs Flask comparison performance features",
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert metadata["used_reformulation"] is True
        assert metadata["classification"] == "complex"

    # ========== METADATA VALIDATION ==========

    async def test_metadata_includes_classification(self):
        """Test metadata includes query classification."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Documents."])

        user_query = "What is Python?"
        chat_history = []

        with patch(
            "ragitect.services.query_service._grade_retrieval_relevance",
            return_value=True,
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert "classification" in metadata
        assert metadata["classification"] in ["simple", "ambiguous", "complex"]

    async def test_metadata_includes_latency(self):
        """Test metadata includes latency_ms field."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Documents."])

        user_query = "What is Python?"
        chat_history = []

        with patch(
            "ragitect.services.query_service._grade_retrieval_relevance",
            return_value=True,
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert "latency_ms" in metadata
        assert isinstance(metadata["latency_ms"], (int, float))
        assert metadata["latency_ms"] >= 0

    async def test_metadata_includes_used_reformulation(self):
        """Test metadata includes used_reformulation boolean."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Documents."])

        user_query = "What is Python?"
        chat_history = []

        with patch(
            "ragitect.services.query_service._grade_retrieval_relevance",
            return_value=True,
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert "used_reformulation" in metadata
        assert isinstance(metadata["used_reformulation"], bool)

    async def test_metadata_includes_grade(self):
        """Test metadata includes grade for simple queries."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Documents."])

        user_query = "What is Python?"
        chat_history = []

        with patch(
            "ragitect.services.query_service._grade_retrieval_relevance",
            return_value=True,
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert "grade" in metadata

    async def test_metadata_includes_original_and_final_query(self):
        """Test metadata includes original_query and final_query."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Documents."])

        user_query = "How do I use it?"
        chat_history = [{"role": "user", "content": "Tell me about pytest"}]

        with patch(
            "ragitect.services.query_service.reformulate_query_with_chat_history",
            return_value="How do I use pytest?",
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert metadata["original_query"] == user_query
        assert metadata["final_query"] == "How do I use pytest?"

    # ========== METRICS LOGGING ==========

    async def test_metrics_logged_correctly(self):
        """Test that query metrics are logged."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=["Documents."])

        user_query = "What is Python?"
        chat_history = []

        with patch(
            "ragitect.services.query_service._grade_retrieval_relevance",
            return_value=True,
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

            # Verify metrics are captured in metadata
            assert "classification" in metadata
            assert "used_reformulation" in metadata
            assert "latency_ms" in metadata

    # ========== EDGE CASES ==========

    async def test_empty_results_still_returns_metadata(self):
        """Test empty vector search results still return proper metadata."""
        mock_llm = AsyncMock()
        mock_vector_search = AsyncMock(return_value=[])

        user_query = "What is Python?"
        chat_history = []

        with patch(
            "ragitect.services.query_service._grade_retrieval_relevance",
            return_value=True,
        ):
            results, metadata = await query_with_iterative_fallback(
                mock_llm, user_query, chat_history, mock_vector_search
            )

        assert results == []
        assert metadata["classification"] == "simple"
        assert "latency_ms" in metadata
