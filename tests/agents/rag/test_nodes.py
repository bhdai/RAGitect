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
            "search_results": [],
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


class TestSearchAndRankNode:
    """Tests for search_and_rank node (AC #3).

    Verifies the node executes the retrieval pipeline:
    retriever → reranker → MMR → adaptive-K
    """

    @pytest.fixture
    def mock_retrieve_documents(self, mocker):
        """Mock the retrieve_documents tool."""
        return mocker.patch(
            "ragitect.agents.rag.nodes._retrieve_documents_impl",
            new_callable=AsyncMock,
        )

    @pytest.fixture
    def mock_rerank_chunks(self, mocker):
        """Mock the rerank_chunks service."""
        return mocker.patch(
            "ragitect.agents.rag.nodes.rerank_chunks",
            new_callable=AsyncMock,
        )

    @pytest.fixture
    def mock_mmr_select(self, mocker):
        """Mock the mmr_select service."""
        return mocker.patch(
            "ragitect.agents.rag.nodes.mmr_select",
        )

    @pytest.fixture
    def mock_select_adaptive_k(self, mocker):
        """Mock the select_adaptive_k service."""
        return mocker.patch(
            "ragitect.agents.rag.nodes.select_adaptive_k",
        )

    @pytest.fixture
    def mock_embed_fn(self):
        """Mock embedding function."""

        async def embed(text: str) -> list[float]:
            return [0.1] * 768

        return embed

    @pytest.fixture
    def mock_vector_repo(self, mocker):
        """Mock VectorRepository."""
        return mocker.MagicMock()

    @pytest.fixture
    def sample_chunks(self) -> list[dict]:
        """Sample chunks for testing."""
        return [
            {
                "chunk_id": "chunk-1",
                "content": "FastAPI is a modern Python framework.",
                "score": 0.95,
                "document_id": "doc-1",
                "title": "fastapi-guide.md",
            },
            {
                "chunk_id": "chunk-2",
                "content": "Python async/await syntax is powerful.",
                "score": 0.88,
                "document_id": "doc-2",
                "title": "python-async.md",
            },
            {
                "chunk_id": "chunk-3",
                "content": "Uvicorn is an ASGI server.",
                "score": 0.82,
                "document_id": "doc-3",
                "title": "servers.md",
            },
        ]

    @pytest.fixture
    def base_search_state(self) -> dict:
        """Create a state for search_and_rank (sub-state from Send())."""
        return {
            "search_term": "FastAPI installation",
            "workspace_id": "ws-123",
            "query_embedding": [0.1] * 768,
        }

    async def test_search_and_rank_calls_retriever(
        self,
        mock_retrieve_documents,
        mock_rerank_chunks,
        mock_mmr_select,
        mock_select_adaptive_k,
        mock_embed_fn,
        mock_vector_repo,
        sample_chunks,
        base_search_state,
    ):
        """Test that search_and_rank calls the retriever with correct args."""
        from ragitect.agents.rag.nodes import search_and_rank

        # Setup mocks
        mock_retrieve_documents.return_value = sample_chunks
        mock_rerank_chunks.return_value = sample_chunks
        mock_mmr_select.return_value = sample_chunks
        mock_select_adaptive_k.return_value = (sample_chunks, {"adaptive_k": 3})

        # Add dependencies to state
        state = {
            **base_search_state,
            "vector_repo": mock_vector_repo,
            "embed_fn": mock_embed_fn,
        }

        await search_and_rank(state)

        # Verify retriever was called
        mock_retrieve_documents.assert_called_once()
        call_args = mock_retrieve_documents.call_args
        assert call_args.kwargs["query"] == "FastAPI installation"
        assert call_args.kwargs["workspace_id"] == "ws-123"

    async def test_search_and_rank_applies_reranking(
        self,
        mock_retrieve_documents,
        mock_rerank_chunks,
        mock_mmr_select,
        mock_select_adaptive_k,
        mock_embed_fn,
        mock_vector_repo,
        sample_chunks,
        base_search_state,
    ):
        """Test that search_and_rank applies cross-encoder reranking."""
        from ragitect.agents.rag.nodes import search_and_rank

        # Setup mocks
        mock_retrieve_documents.return_value = sample_chunks
        reranked_chunks = [
            {**chunk, "rerank_score": 0.9 - i * 0.1}
            for i, chunk in enumerate(sample_chunks)
        ]
        mock_rerank_chunks.return_value = reranked_chunks
        mock_mmr_select.return_value = reranked_chunks
        mock_select_adaptive_k.return_value = (reranked_chunks, {"adaptive_k": 3})

        # Add dependencies to state
        state = {
            **base_search_state,
            "vector_repo": mock_vector_repo,
            "embed_fn": mock_embed_fn,
        }

        await search_and_rank(state)

        # Verify reranker was called with retrieved chunks
        mock_rerank_chunks.assert_called_once()
        call_args = mock_rerank_chunks.call_args
        assert call_args.args[0] == "FastAPI installation"  # query
        assert call_args.args[1] == sample_chunks  # chunks

    async def test_search_and_rank_applies_mmr_diversity(
        self,
        mock_retrieve_documents,
        mock_rerank_chunks,
        mock_mmr_select,
        mock_select_adaptive_k,
        mock_embed_fn,
        mock_vector_repo,
        sample_chunks,
        base_search_state,
    ):
        """Test that search_and_rank applies MMR diversity selection."""
        from ragitect.agents.rag.nodes import search_and_rank

        # Setup mocks
        mock_retrieve_documents.return_value = sample_chunks
        reranked_chunks = [
            {**chunk, "rerank_score": 0.9 - i * 0.1}
            for i, chunk in enumerate(sample_chunks)
        ]
        mock_rerank_chunks.return_value = reranked_chunks
        mock_mmr_select.return_value = reranked_chunks[:2]  # MMR reduces set
        mock_select_adaptive_k.return_value = (reranked_chunks[:2], {"adaptive_k": 2})

        # Add dependencies to state
        state = {
            **base_search_state,
            "vector_repo": mock_vector_repo,
            "embed_fn": mock_embed_fn,
        }

        await search_and_rank(state)

        # Verify MMR was called with reranked chunks
        mock_mmr_select.assert_called_once()

    async def test_search_and_rank_applies_adaptive_k(
        self,
        mock_retrieve_documents,
        mock_rerank_chunks,
        mock_mmr_select,
        mock_select_adaptive_k,
        mock_embed_fn,
        mock_vector_repo,
        sample_chunks,
        base_search_state,
    ):
        """Test that search_and_rank applies adaptive-K selection."""
        from ragitect.agents.rag.nodes import search_and_rank

        # Setup mocks
        mock_retrieve_documents.return_value = sample_chunks
        reranked_chunks = [
            {**chunk, "rerank_score": 0.9 - i * 0.1}
            for i, chunk in enumerate(sample_chunks)
        ]
        mock_rerank_chunks.return_value = reranked_chunks
        mock_mmr_select.return_value = reranked_chunks
        mock_select_adaptive_k.return_value = (
            reranked_chunks[:2],
            {"adaptive_k": 2, "gap_found": True, "gap_size": 0.2},
        )

        # Add dependencies to state
        state = {
            **base_search_state,
            "vector_repo": mock_vector_repo,
            "embed_fn": mock_embed_fn,
        }

        await search_and_rank(state)

        # Verify adaptive-K was called
        mock_select_adaptive_k.assert_called_once()

    async def test_search_and_rank_returns_context_chunks(
        self,
        mock_retrieve_documents,
        mock_rerank_chunks,
        mock_mmr_select,
        mock_select_adaptive_k,
        mock_embed_fn,
        mock_vector_repo,
        sample_chunks,
        base_search_state,
    ):
        """Test that search_and_rank returns context_chunks for reducer."""
        from ragitect.agents.rag.nodes import search_and_rank

        # Setup mocks
        mock_retrieve_documents.return_value = sample_chunks
        mock_rerank_chunks.return_value = sample_chunks
        mock_mmr_select.return_value = sample_chunks
        final_chunks = sample_chunks[:2]
        mock_select_adaptive_k.return_value = (final_chunks, {"adaptive_k": 2})

        # Add dependencies to state
        state = {
            **base_search_state,
            "vector_repo": mock_vector_repo,
            "embed_fn": mock_embed_fn,
        }

        result = await search_and_rank(state)

        # Result should have search_results for state reducer
        assert "search_results" in result
        assert result["search_results"] == final_chunks
        assert len(result["search_results"]) == 2

    async def test_search_and_rank_handles_empty_results(
        self,
        mock_retrieve_documents,
        mock_rerank_chunks,
        mock_mmr_select,
        mock_select_adaptive_k,
        mock_embed_fn,
        mock_vector_repo,
        base_search_state,
    ):
        """Test that search_and_rank handles no retrieval results gracefully."""
        from ragitect.agents.rag.nodes import search_and_rank

        # No chunks retrieved
        mock_retrieve_documents.return_value = []
        mock_rerank_chunks.return_value = []
        mock_mmr_select.return_value = []
        mock_select_adaptive_k.return_value = ([], {"adaptive_k": 0})

        # Add dependencies to state
        state = {
            **base_search_state,
            "vector_repo": mock_vector_repo,
            "embed_fn": mock_embed_fn,
        }

        result = await search_and_rank(state)

        assert "search_results" in result
        assert result["search_results"] == []

    async def test_search_and_rank_uses_config_constants(
        self,
        mock_retrieve_documents,
        mock_rerank_chunks,
        mock_mmr_select,
        mock_select_adaptive_k,
        mock_embed_fn,
        mock_vector_repo,
        sample_chunks,
        base_search_state,
    ):
        """Test that search_and_rank uses config constants for pipeline params."""
        from ragitect.agents.rag.nodes import search_and_rank
        from ragitect.services.config import (
            RETRIEVAL_INITIAL_K,
            RETRIEVAL_RERANKER_TOP_K,
        )

        # Setup mocks
        mock_retrieve_documents.return_value = sample_chunks
        mock_rerank_chunks.return_value = sample_chunks
        mock_mmr_select.return_value = sample_chunks
        mock_select_adaptive_k.return_value = (sample_chunks, {"adaptive_k": 3})

        # Add dependencies to state
        state = {
            **base_search_state,
            "vector_repo": mock_vector_repo,
            "embed_fn": mock_embed_fn,
        }

        await search_and_rank(state)

        # Verify initial retrieval uses RETRIEVAL_INITIAL_K
        retrieve_call = mock_retrieve_documents.call_args
        assert retrieve_call.kwargs["top_k"] == RETRIEVAL_INITIAL_K

        # Verify reranker uses RETRIEVAL_RERANKER_TOP_K
        rerank_call = mock_rerank_chunks.call_args
        assert rerank_call.kwargs["top_k"] == RETRIEVAL_RERANKER_TOP_K


class TestMergeContextNode:
    """Tests for merge_context node (AC #4).

    Verifies deduplication by chunk_id, re-ranking by score, and top-N limiting.
    """

    @pytest.fixture
    def sample_chunks_with_duplicates(self) -> list[dict]:
        """Sample chunks with duplicates for testing merge_context."""
        return [
            # From search 1
            {
                "chunk_id": "chunk-1",
                "content": "FastAPI is a modern Python framework.",
                "score": 0.95,
                "rerank_score": 0.92,
                "document_id": "doc-1",
                "title": "fastapi-guide.md",
            },
            {
                "chunk_id": "chunk-2",
                "content": "Python async/await syntax is powerful.",
                "score": 0.88,
                "rerank_score": 0.85,
                "document_id": "doc-2",
                "title": "python-async.md",
            },
            # From search 2 (duplicate chunk-1 with different score)
            {
                "chunk_id": "chunk-1",
                "content": "FastAPI is a modern Python framework.",
                "score": 0.90,
                "rerank_score": 0.88,  # Lower score for same chunk
                "document_id": "doc-1",
                "title": "fastapi-guide.md",
            },
            {
                "chunk_id": "chunk-3",
                "content": "Uvicorn is an ASGI server.",
                "score": 0.82,
                "rerank_score": 0.80,
                "document_id": "doc-3",
                "title": "servers.md",
            },
        ]

    @pytest.fixture
    def base_merge_state(self, sample_chunks_with_duplicates) -> RAGState:
        """Create a RAGState with chunks from parallel searches."""
        return {
            "messages": [HumanMessage(content="How do I use FastAPI?")],
            "original_query": "How do I use FastAPI?",
            "final_query": None,
            "final_query": None,
            "strategy": None,
            "search_results": sample_chunks_with_duplicates,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

    async def test_merge_context_deduplicates_by_chunk_id(
        self, base_merge_state, sample_chunks_with_duplicates
    ):
        """Test that merge_context removes duplicate chunks by chunk_id."""
        from ragitect.agents.rag.nodes import merge_context

        result = await merge_context(base_merge_state)

        # Should have 3 unique chunks (chunk-1, chunk-2, chunk-3)
        assert "context_chunks" in result
        chunk_ids = [c["chunk_id"] for c in result["context_chunks"]]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids found"
        assert len(result["context_chunks"]) == 3

    async def test_merge_context_keeps_highest_score_on_duplicate(
        self, base_merge_state
    ):
        """Test that merge_context keeps the highest score when deduplicating."""
        from ragitect.agents.rag.nodes import merge_context

        result = await merge_context(base_merge_state)

        # Find chunk-1 in results
        chunk_1 = next(
            (c for c in result["context_chunks"] if c["chunk_id"] == "chunk-1"), None
        )
        assert chunk_1 is not None
        # Should keep the higher rerank_score (0.92, not 0.88)
        assert chunk_1["rerank_score"] == 0.92

    async def test_merge_context_sorts_by_score_descending(self, base_merge_state):
        """Test that merge_context sorts chunks by score in descending order."""
        from ragitect.agents.rag.nodes import merge_context

        result = await merge_context(base_merge_state)

        scores = [c["rerank_score"] for c in result["context_chunks"]]
        assert scores == sorted(scores, reverse=True), "Chunks not sorted by score"

    async def test_merge_context_limits_to_top_n(self, base_merge_state):
        """Test that merge_context limits results to RETRIEVAL_ADAPTIVE_K_MAX."""
        from ragitect.agents.rag.nodes import merge_context
        from ragitect.services.config import RETRIEVAL_ADAPTIVE_K_MAX

        # Create state with many chunks (more than max)
        many_chunks = [
            {
                "chunk_id": f"chunk-{i}",
                "content": f"Content {i}",
                "score": 0.95 - i * 0.01,
                "rerank_score": 0.95 - i * 0.01,
                "document_id": f"doc-{i}",
                "title": f"doc-{i}.md",
            }
            for i in range(50)  # More than RETRIEVAL_ADAPTIVE_K_MAX (16)
        ]
        state = base_merge_state.copy()
        state["search_results"] = many_chunks

        result = await merge_context(state)

        assert len(result["context_chunks"]) <= RETRIEVAL_ADAPTIVE_K_MAX

    async def test_merge_context_handles_empty_chunks(self):
        """Test that merge_context handles empty chunk list gracefully."""
        from ragitect.agents.rag.nodes import merge_context

        state: RAGState = {
            "messages": [],
            "original_query": "test",
            "final_query": None,
            "strategy": None,
            "search_results": [],
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        result = await merge_context(state)

        assert "context_chunks" in result
        assert result["context_chunks"] == []

    async def test_merge_context_stores_in_state(self, base_merge_state):
        """Test that merged context is stored in state for downstream nodes."""
        from ragitect.agents.rag.nodes import merge_context

        result = await merge_context(base_merge_state)

        # Result should have context_chunks key for state update
        assert "context_chunks" in result
        assert isinstance(result["context_chunks"], list)
        # All chunks should have required fields
        for chunk in result["context_chunks"]:
            assert "chunk_id" in chunk
            assert "content" in chunk
            assert "score" in chunk or "rerank_score" in chunk


class TestGenerateAnswerNode:
    """Tests for generate_answer node (AC #5).

    Verifies LLM response generation with proper citations.
    """

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock()
        return mock

    @pytest.fixture
    def sample_context_chunks(self) -> list[dict]:
        """Sample context chunks for answer generation."""
        return [
            {
                "chunk_id": "chunk-1",
                "content": "FastAPI is a modern, fast web framework for Python.",
                "score": 0.95,
                "rerank_score": 0.92,
                "document_id": "doc-1",
                "title": "fastapi-guide.md",
            },
            {
                "chunk_id": "chunk-2",
                "content": "FastAPI uses Pydantic for data validation.",
                "score": 0.88,
                "rerank_score": 0.85,
                "document_id": "doc-2",
                "title": "pydantic-docs.md",
            },
        ]

    @pytest.fixture
    def base_answer_state(self, sample_context_chunks) -> RAGState:
        """Create a RAGState for generate_answer testing."""
        return {
            "messages": [HumanMessage(content="What is FastAPI?")],
            "original_query": "What is FastAPI?",
            "final_query": None,
            "strategy": None,
            "context_chunks": sample_context_chunks,
            "citations": [],
            "llm_calls": 0,
        }

    async def test_generate_answer_calls_llm(self, mock_llm, base_answer_state):
        """Test that generate_answer invokes the LLM."""
        from ragitect.agents.rag.nodes import generate_answer

        mock_llm.ainvoke.return_value = AIMessage(
            content="FastAPI is a modern Python web framework. [cite: 1]"
        )

        await generate_answer(base_answer_state, llm=mock_llm)

        mock_llm.ainvoke.assert_called_once()

    async def test_generate_answer_uses_rag_prompt(self, mock_llm, base_answer_state):
        """Test that generate_answer uses build_rag_system_prompt."""
        from ragitect.agents.rag.nodes import generate_answer

        mock_llm.ainvoke.return_value = AIMessage(content="FastAPI is great. [cite: 1]")

        await generate_answer(base_answer_state, llm=mock_llm)

        # Verify the prompt includes context from chunks
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        # Should have system message with RAG prompt containing context
        assert any("FastAPI" in str(msg) for msg in messages)

    async def test_generate_answer_returns_ai_message(
        self, mock_llm, base_answer_state
    ):
        """Test that generate_answer returns AIMessage in messages list."""
        from ragitect.agents.rag.nodes import generate_answer

        expected_response = AIMessage(
            content="FastAPI is a modern Python web framework. [cite: 1] [cite: 2]"
        )
        mock_llm.ainvoke.return_value = expected_response

        result = await generate_answer(base_answer_state, llm=mock_llm)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "[cite: 1]" in result["messages"][0].content

    async def test_generate_answer_increments_llm_calls(
        self, mock_llm, base_answer_state
    ):
        """Test that generate_answer increments llm_calls counter."""
        from ragitect.agents.rag.nodes import generate_answer

        mock_llm.ainvoke.return_value = AIMessage(content="Answer. [cite: 1]")

        result = await generate_answer(base_answer_state, llm=mock_llm)

        assert "llm_calls" in result
        assert result["llm_calls"] == 1

    async def test_generate_answer_handles_empty_context(self, mock_llm):
        """Test that generate_answer handles empty context gracefully."""
        from ragitect.agents.rag.nodes import generate_answer

        state: RAGState = {
            "messages": [HumanMessage(content="What is FastAPI?")],
            "original_query": "What is FastAPI?",
            "final_query": None,
            "strategy": None,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        mock_llm.ainvoke.return_value = AIMessage(
            content="I cannot find information about FastAPI in your documents."
        )

        result = await generate_answer(state, llm=mock_llm)

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)

    async def test_generate_answer_includes_chat_history(
        self, mock_llm, base_answer_state
    ):
        """Test that generate_answer includes chat history for context."""
        from ragitect.agents.rag.nodes import generate_answer

        # Add conversation history
        state = base_answer_state.copy()
        state["messages"] = [
            HumanMessage(content="Tell me about Python web frameworks"),
            AIMessage(content="There are several Python web frameworks..."),
            HumanMessage(content="What about FastAPI specifically?"),
        ]

        mock_llm.ainvoke.return_value = AIMessage(
            content="FastAPI is a modern framework. [cite: 1]"
        )

        await generate_answer(state, llm=mock_llm)

        # Verify LLM was called with messages
        call_args = mock_llm.ainvoke.call_args
        messages_passed = call_args[0][0]
        # Should include previous messages for context
        assert len(messages_passed) >= 2  # At least system + user message

    async def test_generate_answer_formats_chunks_for_prompt(
        self, mock_llm, base_answer_state
    ):
        """Test that context chunks are properly formatted in the prompt."""
        from ragitect.agents.rag.nodes import generate_answer

        mock_llm.ainvoke.return_value = AIMessage(content="Answer. [cite: 1]")

        await generate_answer(base_answer_state, llm=mock_llm)

        # Check that chunks were formatted into the prompt
        call_args = mock_llm.ainvoke.call_args
        messages_passed = call_args[0][0]
        prompt_content = str(messages_passed)
        # Should contain chunk content
        assert "FastAPI is a modern" in prompt_content or "Pydantic" in prompt_content
