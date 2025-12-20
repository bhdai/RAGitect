"""Tests for LLM Judge relevance grading.

Tests for _grade_retrieval_relevance function that grades
if retrieved documents are relevant to a query.
"""

import pytest
from unittest.mock import AsyncMock, patch

# Import will fail until implementation exists - this is expected (RED phase)
from ragitect.services.query_service import _grade_retrieval_relevance

pytestmark = [pytest.mark.asyncio]


class TestLLMJudge:
    """Tests for _grade_retrieval_relevance function."""

    # ========== BASIC FUNCTIONALITY TESTS ==========

    async def test_relevant_docs_returns_true(self):
        """Test that relevant documents return True."""
        mock_llm = AsyncMock()

        query = "What is FastAPI?"
        relevant_docs = [
            "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints."
        ]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='{"score": "yes"}',
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, relevant_docs)

        assert result is True

    async def test_irrelevant_docs_returns_false(self):
        """Test that irrelevant documents return False."""
        mock_llm = AsyncMock()

        query = "What is FastAPI?"
        irrelevant_docs = [
            "Chocolate cake is a delicious dessert made with cocoa powder and butter."
        ]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='{"score": "no"}',
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, irrelevant_docs)

        assert result is False

    async def test_empty_doc_list_returns_true(self):
        """Test that empty document list returns True (fail open)."""
        mock_llm = AsyncMock()

        query = "What is FastAPI?"
        empty_docs = []

        result = await _grade_retrieval_relevance(mock_llm, query, empty_docs)
        assert result is True

    # ========== MOCK LLM RESPONSE TESTS ==========

    async def test_llm_returns_yes_returns_true(self):
        """Test LLM returning 'yes' returns True."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='{"score": "yes"}',
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True

    async def test_llm_returns_yes_with_punctuation_returns_true(self):
        """Test LLM returning 'yes.' returns True."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='{"score": "yes"}',
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True

    async def test_llm_returns_yes_with_explanation_returns_true(self):
        """Test LLM returning 'yes, the document is relevant' returns True."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='{"score": "yes"}',
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True

    async def test_llm_returns_no_returns_false(self):
        """Test LLM returning 'no' returns False."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Chocolate cake recipe."]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='{"score": "no"}',
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is False

    async def test_llm_returns_uppercase_yes_returns_true(self):
        """Test LLM returning 'YES' (uppercase) returns True."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='{"score": "YES"}',
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True

    # ========== ERROR HANDLING TESTS (FAIL OPEN) ==========

    async def test_llm_timeout_returns_true(self):
        """Test LLM timeout returns True (fail open)."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response",
            side_effect=TimeoutError("Connection timeout"),
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True  # Fail open

    async def test_llm_api_error_returns_true(self):
        """Test LLM API error returns True (fail open)."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response",
            side_effect=Exception("API Error"),
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True  # Fail open

    async def test_llm_connection_refused_returns_true(self):
        """Test LLM connection refused returns True (fail open)."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True  # Fail open

    # ========== EDGE CASES ==========

    async def test_long_document_truncated(self):
        """Test that long documents are truncated for grading."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        # Create a document longer than 500 chars
        long_doc = "Python is great. " * 100  # ~1700 chars
        docs = [long_doc]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='{"score": "yes"}',
        ) as mock_gen:
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True
        # Verify the LLM was called
        mock_gen.assert_called_once()

    async def test_multiple_docs_uses_first(self):
        """Test that multiple documents use only the first (top result)."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = [
            "Python is a programming language.",
            "Chocolate cake is delicious.",
            "The weather is nice today.",
        ]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='{"score": "yes"}',
        ) as mock_gen:
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True
        # Only called once with first doc
        mock_gen.assert_called_once()

    # ========== JSON PARSING TESTS ==========

    async def test_plain_text_yes_fallback(self):
        """Test backward compatibility with plain text 'yes' response."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response", return_value="yes"
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True

    async def test_plain_text_no_fallback(self):
        """Test backward compatibility with plain text 'no' response."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Chocolate cake recipe."]

        with patch(
            "ragitect.services.query_service.generate_response", return_value="no"
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is False

    async def test_markdown_wrapped_json(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='```json\n{"score": "yes"}\n```',
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True

    async def test_json_with_preamble(self):
        """Test parsing JSON with conversational preamble."""
        mock_llm = AsyncMock()

        query = "What is Python?"
        docs = ["Python is a programming language."]

        with patch(
            "ragitect.services.query_service.generate_response",
            return_value='Here is my assessment: {"score": "yes"}',
        ):
            result = await _grade_retrieval_relevance(mock_llm, query, docs)

        assert result is True
