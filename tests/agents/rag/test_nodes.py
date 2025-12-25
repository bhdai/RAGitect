"""Tests for RAG agent nodes.

Tests for node functions used in the LangGraph RAG pipeline.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ragitect.agents.rag.schemas import Search, SearchStrategy
from ragitect.agents.rag.state import RAGState

pytestmark = [pytest.mark.asyncio]


class TestGenerateStrategyNode:
    """Tests for generate_strategy node."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with structured output."""
        mock = MagicMock()
        mock.with_structured_output = MagicMock(return_value=MagicMock())
        return mock

    @pytest.fixture
    def base_state(self) -> RAGState:
        """Create a base RAGState for testing."""
        return {
            "messages": [HumanMessage(content="How do I install FastAPI?")],
            "original_query": "How do I install FastAPI?",
            "final_query": None,
            "strategy": None,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

    async def test_generate_strategy_simple_query_single_search(
        self, mock_llm, base_state
    ):
        """Test that a simple query generates a single search term."""
        from ragitect.agents.rag.nodes import generate_strategy

        # Configure mock to return a simple strategy
        expected_strategy = SearchStrategy(
            reasoning="Direct question about FastAPI installation",
            searches=[
                Search(term="FastAPI installation guide", reasoning="Main topic query")
            ],
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=expected_strategy
        )

        result = await generate_strategy(base_state, llm=mock_llm)

        assert "strategy" in result
        assert isinstance(result["strategy"], SearchStrategy)
        assert len(result["strategy"].searches) == 1
        assert result["strategy"].searches[0].term == "FastAPI installation guide"

    async def test_generate_strategy_complex_query_multiple_searches(
        self, mock_llm, base_state
    ):
        """Test that a complex query generates 2-5 search terms."""
        from ragitect.agents.rag.nodes import generate_strategy

        # Complex query about multiple aspects
        state = base_state.copy()
        state["original_query"] = (
            "How does FastAPI handle authentication and what are the best practices "
            "for JWT tokens and OAuth2 integration?"
        )
        state["messages"] = [HumanMessage(content=state["original_query"])]

        expected_strategy = SearchStrategy(
            reasoning="Complex query covering authentication, JWT, and OAuth2",
            searches=[
                Search(term="FastAPI authentication", reasoning="Main auth topic"),
                Search(term="FastAPI JWT tokens", reasoning="JWT implementation"),
                Search(
                    term="FastAPI OAuth2 integration", reasoning="OAuth2 integration"
                ),
            ],
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=expected_strategy
        )

        result = await generate_strategy(state, llm=mock_llm)

        assert len(result["strategy"].searches) == 3
        assert all(isinstance(s, Search) for s in result["strategy"].searches)

    async def test_generate_strategy_pronoun_resolution_via_chat_history(
        self, mock_llm, base_state
    ):
        """Test that pronouns are resolved using chat history context."""
        from ragitect.agents.rag.nodes import generate_strategy

        # Chat history with context about a specific topic
        state = base_state.copy()
        state["messages"] = [
            HumanMessage(content="Tell me about SQLAlchemy models"),
            AIMessage(
                content="SQLAlchemy models are defined using declarative base..."
            ),
            HumanMessage(content="How do I test them?"),  # 'them' = SQLAlchemy models
        ]
        state["original_query"] = "How do I test them?"

        expected_strategy = SearchStrategy(
            reasoning="User asking about testing SQLAlchemy models (resolved 'them')",
            searches=[
                Search(
                    term="testing SQLAlchemy models",
                    reasoning="Resolved pronoun from context",
                )
            ],
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=expected_strategy
        )

        result = await generate_strategy(state, llm=mock_llm)

        # The LLM should have been invoked with chat history for context
        mock_llm.with_structured_output.return_value.ainvoke.assert_called_once()
        call_args = mock_llm.with_structured_output.return_value.ainvoke.call_args
        # Verify that messages were passed (for pronoun resolution)
        assert call_args is not None

        # Result should reflect resolved pronouns in search terms
        assert "strategy" in result
        assert result["strategy"].searches[0].term == "testing SQLAlchemy models"

    async def test_generate_strategy_increments_llm_calls(self, mock_llm, base_state):
        """Test that generate_strategy increments llm_calls counter."""
        from ragitect.agents.rag.nodes import generate_strategy

        expected_strategy = SearchStrategy(
            reasoning="Test",
            searches=[Search(term="query", reasoning="reason")],
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=expected_strategy
        )

        result = await generate_strategy(base_state, llm=mock_llm)

        assert "llm_calls" in result
        assert result["llm_calls"] == 1

    async def test_generate_strategy_stores_in_state(self, mock_llm, base_state):
        """Test that strategy is stored in state for observability (AC1)."""
        from ragitect.agents.rag.nodes import generate_strategy

        expected_strategy = SearchStrategy(
            reasoning="Strategy for observability",
            searches=[Search(term="observable query", reasoning="test")],
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=expected_strategy
        )

        result = await generate_strategy(base_state, llm=mock_llm)

        # Strategy must be in result for state update
        assert "strategy" in result
        assert result["strategy"] == expected_strategy

    async def test_generate_strategy_uses_structured_output(self, mock_llm, base_state):
        """Test that LLM is called with structured output for type safety."""
        from ragitect.agents.rag.nodes import generate_strategy

        expected_strategy = SearchStrategy(
            reasoning="Test",
            searches=[Search(term="query", reasoning="reason")],
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=expected_strategy
        )

        await generate_strategy(base_state, llm=mock_llm)

        # Verify with_structured_output was called with SearchStrategy
        mock_llm.with_structured_output.assert_called_once_with(SearchStrategy)

    async def test_generate_strategy_handles_empty_chat_history(self, mock_llm):
        """Test that generate_strategy works with empty chat history."""
        from ragitect.agents.rag.nodes import generate_strategy

        state: RAGState = {
            "messages": [],
            "original_query": "What is Python?",
            "final_query": None,
            "strategy": None,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        expected_strategy = SearchStrategy(
            reasoning="No history, direct question",
            searches=[Search(term="What is Python", reasoning="Direct query")],
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=expected_strategy
        )

        result = await generate_strategy(state, llm=mock_llm)

        assert "strategy" in result
        assert len(result["strategy"].searches) >= 1
