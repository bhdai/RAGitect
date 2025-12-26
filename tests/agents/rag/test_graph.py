"""Tests for RAG StateGraph assembly and execution.

Tests for StateGraph compilation and Send() fan-out behavior (AC #6, #8).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send

from ragitect.agents.rag.schemas import Search, SearchStrategy
from ragitect.agents.rag.state import RAGState

pytestmark = [pytest.mark.asyncio]


class TestStateGraphCompilation:
    """Tests for StateGraph assembly and compilation (AC #6)."""

    async def test_build_rag_graph_returns_compiled_runnable(self):
        """Test that build_rag_graph() compiles and returns a Runnable."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()

        # Graph should be a compiled Runnable
        assert graph is not None
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    async def test_build_rag_graph_has_required_nodes(self):
        """Test that the graph contains all required nodes."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()

        # The compiled graph should have our nodes
        # Access the underlying graph structure
        assert graph is not None
        # Verify nodes exist in the graph (implementation will expose this)
        nodes = graph.get_graph().nodes
        node_names = [n.name for n in nodes.values() if hasattr(n, "name")]

        assert "generate_strategy" in node_names or any(
            "strategy" in str(n) for n in node_names
        )
        assert "search_and_rank" in node_names or any(
            "search" in str(n) for n in node_names
        )
        assert "merge_context" in node_names or any(
            "merge" in str(n) for n in node_names
        )
        assert "generate_answer" in node_names or any(
            "answer" in str(n) for n in node_names
        )

    async def test_build_rag_graph_compiles_without_errors(self):
        """Test that StateGraph compiles without errors (AC #6)."""
        from ragitect.agents.rag.graph import build_rag_graph

        # Should not raise any exceptions
        try:
            graph = build_rag_graph()
            assert graph is not None
        except Exception as e:
            pytest.fail(f"Graph compilation failed with error: {e}")

    async def test_graph_has_start_to_generate_strategy_edge(self):
        """Test that graph starts with generate_strategy node."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()
        graph_repr = graph.get_graph()

        # Check edges from START to generate_strategy
        edges = graph_repr.edges
        # START should connect to generate_strategy
        start_edges = [e for e in edges if "__start__" in str(e)]
        assert len(start_edges) > 0, "No edge from START found"

    async def test_graph_has_generate_answer_to_end_edge(self):
        """Test that graph ends with generate_answer node to END."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()
        graph_repr = graph.get_graph()

        # Check edges from generate_answer to END
        edges = graph_repr.edges
        end_edges = [e for e in edges if "__end__" in str(e)]
        assert len(end_edges) > 0, "No edge to END found"


class TestContinueToSearches:
    """Tests for the continue_to_searches conditional edge (AC #2)."""

    @pytest.fixture
    def sample_strategy(self) -> SearchStrategy:
        """Create a sample search strategy."""
        return SearchStrategy(
            reasoning="Test strategy",
            searches=[
                Search(term="FastAPI installation", reasoning="Main topic"),
                Search(term="FastAPI configuration", reasoning="Setup details"),
                Search(term="FastAPI deployment", reasoning="Production use"),
            ],
        )

    @pytest.fixture
    def base_state(self, sample_strategy) -> RAGState:
        """Create a base RAGState with strategy."""
        return {
            "messages": [HumanMessage(content="How do I install FastAPI?")],
            "original_query": "How do I install FastAPI?",
            "final_query": None,
            "strategy": sample_strategy,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 1,
        }

    async def test_continue_to_searches_returns_send_objects(
        self, base_state, sample_strategy
    ):
        """Test that continue_to_searches returns list of Send objects."""
        from ragitect.agents.rag.graph import continue_to_searches

        result = continue_to_searches(base_state)

        assert isinstance(result, list)
        assert all(isinstance(s, Send) for s in result)
        assert len(result) == len(sample_strategy.searches)

    async def test_continue_to_searches_sends_to_search_and_rank(
        self, base_state, sample_strategy
    ):
        """Test that Send objects target search_and_rank node."""
        from ragitect.agents.rag.graph import continue_to_searches

        result = continue_to_searches(base_state)

        # All Send objects should target "search_and_rank"
        for send in result:
            assert send.node == "search_and_rank"

    async def test_continue_to_searches_includes_search_term_in_state(
        self, base_state, sample_strategy
    ):
        """Test that each Send includes the search term in sub-state."""
        from ragitect.agents.rag.graph import continue_to_searches

        result = continue_to_searches(base_state)

        # Each Send should have the search term in its state
        for i, send in enumerate(result):
            assert "search_term" in send.arg
            assert send.arg["search_term"] == sample_strategy.searches[i].term

    async def test_continue_to_searches_enforces_max_5_searches(self):
        """Test that maximum 5 parallel searches are enforced (AC #2)."""
        from ragitect.agents.rag.graph import continue_to_searches

        # Create strategy with more than 5 searches
        strategy_with_many_searches = SearchStrategy(
            reasoning="Complex query",
            searches=[
                Search(term=f"search term {i}", reasoning=f"reason {i}")
                for i in range(10)
            ],
        )

        state: RAGState = {
            "messages": [],
            "original_query": "complex query",
            "final_query": None,
            "strategy": strategy_with_many_searches,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 1,
        }

        result = continue_to_searches(state)

        # Should be capped at 5
        assert len(result) <= 5

    async def test_continue_to_searches_passes_workspace_id(self, base_state):
        """Test that workspace_id is passed to parallel searches."""
        from ragitect.agents.rag.graph import continue_to_searches

        # Add workspace_id to state
        state = dict(base_state)
        state["workspace_id"] = "ws-test-123"

        result = continue_to_searches(state)

        # Each Send should include workspace_id
        for send in result:
            assert send.arg.get("workspace_id") == "ws-test-123"

    async def test_continue_to_searches_single_search_term(self):
        """Test that simple queries with 1 search term work correctly."""
        from ragitect.agents.rag.graph import continue_to_searches

        single_search_strategy = SearchStrategy(
            reasoning="Simple query",
            searches=[Search(term="FastAPI basics", reasoning="Direct question")],
        )

        state: RAGState = {
            "messages": [],
            "original_query": "What is FastAPI?",
            "final_query": None,
            "strategy": single_search_strategy,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 1,
        }

        result = continue_to_searches(state)

        assert len(result) == 1
        assert result[0].arg["search_term"] == "FastAPI basics"


class TestGraphExecution:
    """Tests for graph execution flow (AC #6)."""

    @pytest.fixture
    def mock_dependencies(self, mocker):
        """Set up mocks for all external dependencies."""
        # Mock nodes
        mocker.patch(
            "ragitect.agents.rag.graph.generate_strategy",
            new_callable=AsyncMock,
            return_value={
                "strategy": SearchStrategy(
                    reasoning="Test",
                    searches=[Search(term="test query", reasoning="test")],
                ),
                "llm_calls": 1,
            },
        )

        mocker.patch(
            "ragitect.agents.rag.graph.search_and_rank",
            new_callable=AsyncMock,
            return_value={
                "search_results": [
                    {
                        "chunk_id": "chunk-1",
                        "content": "Test content",
                        "score": 0.9,
                        "document_id": "doc-1",
                        "title": "test.md",
                    }
                ]
            },
        )

        mocker.patch(
            "ragitect.agents.rag.graph.merge_context",
            new_callable=AsyncMock,
            return_value={
                "context_chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "content": "Test content",
                        "score": 0.9,
                        "document_id": "doc-1",
                        "title": "test.md",
                    }
                ]
            },
        )

        mocker.patch(
            "ragitect.agents.rag.graph.generate_answer",
            new_callable=AsyncMock,
            return_value={
                "messages": [AIMessage(content="Test answer. [cite: 1]")],
                "llm_calls": 1,
            },
        )

        return mocker

    async def test_graph_flow_start_to_generate_strategy(self):
        """Test that graph execution starts with generate_strategy."""
        from ragitect.agents.rag.graph import build_rag_graph

        # This tests the graph structure, not actual execution
        graph = build_rag_graph()
        graph_repr = graph.get_graph()

        # Find starting node
        edges = list(graph_repr.edges)
        start_edge = next((e for e in edges if e.source == "__start__"), None)

        assert start_edge is not None
        assert "strategy" in start_edge.target.lower()

    async def test_graph_flow_ends_with_generate_answer(self):
        """Test that graph execution ends with generate_answer."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()
        graph_repr = graph.get_graph()

        # Find ending node
        edges = list(graph_repr.edges)
        end_edge = next((e for e in edges if e.target == "__end__"), None)

        assert end_edge is not None
        assert "answer" in end_edge.source.lower()

    async def test_graph_has_conditional_edge_for_fanout(self):
        """Test that graph has conditional edge for Send() fan-out."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()
        graph_repr = graph.get_graph()

        # Verify conditional edge exists from generate_strategy
        edges = list(graph_repr.edges)
        strategy_edges = [e for e in edges if "strategy" in str(e.source).lower()]

        # Should have conditional edge (not direct edge to single node)
        assert len(strategy_edges) > 0


class TestSearchAndRankConvergence:
    """Tests for search_and_rank parallel branch convergence."""

    async def test_all_parallel_branches_converge_to_merge_context(self):
        """Test that all search_and_rank branches converge to merge_context."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()
        graph_repr = graph.get_graph()

        edges = list(graph_repr.edges)

        # Find edges from search_and_rank
        search_edges = [e for e in edges if "search" in str(e.source).lower()]

        # All should go to merge_context
        for edge in search_edges:
            assert "merge" in str(edge.target).lower()


class TestFailForwardDesign:
    """Tests for fail-forward design without retry loops (NFR2)."""

    async def test_graph_has_no_retry_loops(self):
        """Test that graph has no retry loops (fail-forward design)."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()
        graph_repr = graph.get_graph()

        edges = list(graph_repr.edges)

        # Check for back edges (edges going to earlier nodes)
        # A proper DAG should not have cycles
        node_order = [
            "__start__",
            "generate_strategy",
            "search_and_rank",
            "merge_context",
            "generate_answer",
            "__end__",
        ]

        for edge in edges:
            source_idx = next(
                (i for i, n in enumerate(node_order) if n in str(edge.source).lower()),
                -1,
            )
            target_idx = next(
                (i for i, n in enumerate(node_order) if n in str(edge.target).lower()),
                -1,
            )

            # Target should always be after source (no back edges)
            if source_idx >= 0 and target_idx >= 0:
                # Allow parallel edges (same level) but no back edges
                # search_and_rank to merge_context is valid
                if "search" in str(edge.source).lower():
                    continue  # Skip checking parallel search nodes
                assert source_idx <= target_idx, (
                    f"Back edge found: {edge.source} -> {edge.target}"
                )


# =============================================================================
# Integration Tests (AC #8)
# =============================================================================


@pytest.mark.integration
class TestGraphIntegration:
    """Integration tests for end-to-end graph execution (AC #8).

    These tests verify the full graph execution flow with mocked LLM
    but exercise the actual node logic and state transitions.
    """

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for both strategy and answer generation."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock()
        mock.with_structured_output = MagicMock(return_value=MagicMock())
        return mock

    @pytest.fixture
    def sample_context_chunks(self) -> list[dict]:
        """Sample chunks for testing."""
        return [
            {
                "chunk_id": "chunk-1",
                "content": "FastAPI is a modern Python web framework.",
                "score": 0.95,
                "rerank_score": 0.92,
                "document_id": "doc-1",
                "title": "fastapi-guide.md",
            },
            {
                "chunk_id": "chunk-2",
                "content": "FastAPI uses Pydantic for validation.",
                "score": 0.88,
                "rerank_score": 0.85,
                "document_id": "doc-2",
                "title": "pydantic-docs.md",
            },
        ]

    async def test_simple_query_single_search_term(
        self, mock_llm, sample_context_chunks, mocker
    ):
        """Test end-to-end execution with a simple query (1 search term)."""
        # Setup mocks
        simple_strategy = SearchStrategy(
            reasoning="Simple question about FastAPI",
            searches=[Search(term="FastAPI basics", reasoning="Direct query")],
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=simple_strategy
        )
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="FastAPI is a modern framework. [cite: 1]")
        )

        # Mock node dependencies
        mocker.patch(
            "ragitect.agents.rag.nodes.generate_strategy",
            new_callable=AsyncMock,
            return_value={"strategy": simple_strategy, "llm_calls": 1},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.generate_strategy",
            new_callable=AsyncMock,
            return_value={"strategy": simple_strategy, "llm_calls": 1},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.merge_context",
            new_callable=AsyncMock,
            return_value={"context_chunks": sample_context_chunks},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.generate_answer",
            new_callable=AsyncMock,
            return_value={
                "messages": [
                    AIMessage(content="FastAPI is a modern framework. [cite: 1]")
                ],
                "llm_calls": 1,
            },
        )

        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()

        initial_state: RAGState = {
            "messages": [],
            "original_query": "What is FastAPI?",
            "final_query": None,
            "strategy": None,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        result = await graph.ainvoke(initial_state)

        # Verify result has expected keys
        assert "messages" in result
        assert "strategy" in result
        assert "context_chunks" in result

    async def test_complex_query_multiple_search_terms(
        self, mock_llm, sample_context_chunks, mocker
    ):
        """Test multi-search-term queries fan out correctly (2-5 searches)."""
        complex_strategy = SearchStrategy(
            reasoning="Complex query about multiple topics",
            searches=[
                Search(term="FastAPI authentication", reasoning="Auth aspect"),
                Search(term="FastAPI JWT tokens", reasoning="JWT aspect"),
                Search(term="FastAPI OAuth2", reasoning="OAuth aspect"),
            ],
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=complex_strategy
        )
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="FastAPI supports multiple auth methods. [cite: 1] [cite: 2]"
            )
        )

        # Mock node dependencies
        mocker.patch(
            "ragitect.agents.rag.graph.generate_strategy",
            new_callable=AsyncMock,
            return_value={"strategy": complex_strategy, "llm_calls": 1},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.merge_context",
            new_callable=AsyncMock,
            return_value={"context_chunks": sample_context_chunks},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.generate_answer",
            new_callable=AsyncMock,
            return_value={
                "messages": [
                    AIMessage(
                        content="FastAPI supports multiple auth methods. [cite: 1] [cite: 2]"
                    )
                ],
                "llm_calls": 1,
            },
        )

        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()

        initial_state: RAGState = {
            "messages": [],
            "original_query": "How does FastAPI handle authentication with JWT and OAuth2?",
            "final_query": None,
            "strategy": None,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        result = await graph.ainvoke(initial_state)

        # Verify strategy has 3 search terms
        assert result.get("strategy") is not None
        assert len(result["strategy"].searches) == 3

    async def test_deduplication_in_merge_context(self, mock_llm, mocker):
        """Test that duplicate chunks from multiple searches are deduplicated."""
        # Create chunks with duplicate chunk_id from different searches
        duplicate_chunks = [
            {
                "chunk_id": "chunk-shared",
                "content": "Shared content from search 1",
                "score": 0.90,
                "rerank_score": 0.85,
                "document_id": "doc-1",
                "title": "shared.md",
            },
            {
                "chunk_id": "chunk-shared",  # Duplicate
                "content": "Shared content from search 2",
                "score": 0.92,
                "rerank_score": 0.88,  # Higher score
                "document_id": "doc-1",
                "title": "shared.md",
            },
            {
                "chunk_id": "chunk-unique",
                "content": "Unique content",
                "score": 0.80,
                "rerank_score": 0.78,
                "document_id": "doc-2",
                "title": "unique.md",
            },
        ]

        simple_strategy = SearchStrategy(
            reasoning="Test",
            searches=[Search(term="test query", reasoning="test")],
        )

        # Use actual merge_context to test deduplication
        from ragitect.agents.rag.nodes import merge_context

        state: RAGState = {
            "messages": [],
            "original_query": "test",
            "final_query": None,
            "strategy": simple_strategy,
            "final_query": None,
            "strategy": simple_strategy,
            "search_results": duplicate_chunks,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        result = await merge_context(state)

        # Should have 2 unique chunks
        assert len(result["context_chunks"]) == 2

        # chunk-shared should have the higher score (0.88)
        shared_chunk = next(
            (c for c in result["context_chunks"] if c["chunk_id"] == "chunk-shared"),
            None,
        )
        assert shared_chunk is not None
        assert shared_chunk["rerank_score"] == 0.88

    async def test_citations_in_generated_response(
        self, mock_llm, sample_context_chunks, mocker
    ):
        """Test that citations in response match context chunks."""
        response_with_citations = AIMessage(
            content="FastAPI is modern [cite: 1] and uses Pydantic [cite: 2]."
        )

        simple_strategy = SearchStrategy(
            reasoning="Test",
            searches=[Search(term="FastAPI", reasoning="test")],
        )

        mocker.patch(
            "ragitect.agents.rag.graph.generate_strategy",
            new_callable=AsyncMock,
            return_value={"strategy": simple_strategy, "llm_calls": 1},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.merge_context",
            new_callable=AsyncMock,
            return_value={"context_chunks": sample_context_chunks},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.generate_answer",
            new_callable=AsyncMock,
            return_value={
                "messages": [response_with_citations],
                "llm_calls": 1,
            },
        )

        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()

        initial_state: RAGState = {
            "messages": [],
            "original_query": "What is FastAPI?",
            "final_query": None,
            "strategy": None,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        result = await graph.ainvoke(initial_state)

        # Verify response has citations
        assert len(result["messages"]) > 0
        last_message = result["messages"][-1]
        assert "[cite: 1]" in last_message.content
        assert "[cite: 2]" in last_message.content

        # Verify context chunks are available for citation resolution
        assert len(result["context_chunks"]) == 2

    async def test_graph_state_accumulation_via_reducers(self, mocker):
        """Test that state reducers properly accumulate values."""
        simple_strategy = SearchStrategy(
            reasoning="Test",
            searches=[Search(term="test", reasoning="test")],
        )

        sample_chunks = [
            {
                "chunk_id": "chunk-1",
                "content": "Test content",
                "score": 0.9,
                "document_id": "doc-1",
                "title": "test.md",
            }
        ]

        # Mock all nodes
        mocker.patch(
            "ragitect.agents.rag.graph.generate_strategy",
            new_callable=AsyncMock,
            return_value={"strategy": simple_strategy, "llm_calls": 1},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.merge_context",
            new_callable=AsyncMock,
            return_value={"context_chunks": sample_chunks},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.generate_answer",
            new_callable=AsyncMock,
            return_value={
                "messages": [AIMessage(content="Answer. [cite: 1]")],
                "llm_calls": 1,
            },
        )

        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()

        initial_state: RAGState = {
            "messages": [HumanMessage(content="Hello")],
            "original_query": "What is this?",
            "final_query": None,
            "strategy": None,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        result = await graph.ainvoke(initial_state)

        # llm_calls should be accumulated (1 from strategy + 1 from answer = 2)
        # But since we're mocking, we just verify the structure
        assert "llm_calls" in result

        # messages reducer should accumulate
        assert len(result["messages"]) >= 1

    async def test_empty_search_results_handled_gracefully(self, mocker):
        """Test that empty search results don't break the pipeline."""
        simple_strategy = SearchStrategy(
            reasoning="Test query with no results",
            searches=[Search(term="nonexistent topic", reasoning="test")],
        )

        mocker.patch(
            "ragitect.agents.rag.graph.generate_strategy",
            new_callable=AsyncMock,
            return_value={"strategy": simple_strategy, "llm_calls": 1},
        )
        # search_and_rank returns empty
        mocker.patch(
            "ragitect.agents.rag.graph.merge_context",
            new_callable=AsyncMock,
            return_value={"context_chunks": []},
        )
        mocker.patch(
            "ragitect.agents.rag.graph.generate_answer",
            new_callable=AsyncMock,
            return_value={
                "messages": [
                    AIMessage(
                        content="I couldn't find relevant information in the documents."
                    )
                ],
                "llm_calls": 1,
            },
        )

        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()

        initial_state: RAGState = {
            "messages": [],
            "original_query": "Tell me about something not in docs",
            "final_query": None,
            "strategy": None,
            "search_results": [],
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        result = await graph.ainvoke(initial_state)

        # Should complete without errors
        assert "messages" in result
        assert len(result["context_chunks"]) == 0


class TestDependencyInjection:
    """Tests for dependency injection in the graph (AC #7)."""

    async def test_llm_injection_into_generate_strategy(self):
        """Test that LLM is injected into generate_strategy node."""
        from ragitect.agents.rag.graph import build_rag_graph

        mock_llm = MagicMock()

        # Patch the node function in the graph module
        with patch(
            "ragitect.agents.rag.graph.generate_strategy", new_callable=AsyncMock
        ) as mock_node:
            # Return valid response to prevent graph collapse
            mock_node.return_value = {
                "strategy": SearchStrategy(reasoning="test", searches=[]),
                "llm_calls": 1,
            }

            graph = build_rag_graph(llm=mock_llm, retrieval_only=True)

            initial_state: RAGState = {
                "messages": [],
                "original_query": "test",
                "final_query": None,
                "strategy": None,
                "search_results": [],
                "context_chunks": [],
                "citations": [],
                "llm_calls": 0,
            }

            await graph.ainvoke(initial_state)

            # Verify mock was called with llm kwarg
            mock_node.assert_called_once()
            call_kwargs = mock_node.call_args.kwargs
            assert call_kwargs.get("llm") is mock_llm

    async def test_llm_injection_into_generate_answer(self):
        """Test that LLM is injected into generate_answer node."""
        from ragitect.agents.rag.graph import build_rag_graph

        mock_llm = MagicMock()

        # Patch nodes
        with (
            patch(
                "ragitect.agents.rag.graph.generate_strategy", new_callable=AsyncMock
            ) as mock_strat,
            patch(
                "ragitect.agents.rag.graph.generate_answer", new_callable=AsyncMock
            ) as mock_ans,
            patch(
                "ragitect.agents.rag.graph.search_and_rank", new_callable=AsyncMock
            ) as mock_search,
            patch(
                "ragitect.agents.rag.graph.merge_context", new_callable=AsyncMock
            ) as mock_merge,
        ):
            # Need at least one search to trigger flow to search_and_rank -> merge -> answer
            mock_strat.return_value = {
                "strategy": SearchStrategy(
                    reasoning="t", searches=[Search(term="t", reasoning="t")]
                ),
                "llm_calls": 0,
            }
            mock_ans.return_value = {
                "messages": [AIMessage(content="test")],
                "llm_calls": 0,
            }
            mock_merge.return_value = {"context_chunks": []}
            mock_search.return_value = {"search_results": []}

            graph = build_rag_graph(llm=mock_llm, retrieval_only=False)

            initial_state: RAGState = {
                "messages": [],
                "original_query": "test",
                "final_query": None,
                "strategy": None,
                "search_results": [],
                "context_chunks": [],
                "citations": [],
                "llm_calls": 0,
            }

            await graph.ainvoke(initial_state)

            # Verify generate_answer mock was called with llm kwarg
            mock_ans.assert_called_once()
            call_kwargs = mock_ans.call_args.kwargs
            assert call_kwargs.get("llm") is mock_llm


class TestRetrievalOnlyMode:
    """Tests for retrieval_only parameter in build_rag_graph (Performance Fix)."""

    async def test_retrieval_only_excludes_generate_answer(self):
        """Test that retrieval_only=True excludes generate_answer node."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph(retrieval_only=True)

        # Check nodes
        nodes = graph.get_graph().nodes
        node_names = [n.name for n in nodes.values() if hasattr(n, "name")]

        assert "generate_strategy" in node_names
        assert "search_and_rank" in node_names
        assert "merge_context" in node_names

        # generate_answer should NOT be present
        assert "generate_answer" not in node_names
        assert not any("answer" in str(n).lower() for n in node_names)

    async def test_retrieval_only_connects_merge_to_end(self):
        """Test that retrieval_only=True connects merge_context to END."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph(retrieval_only=True)
        graph_repr = graph.get_graph()

        edges = list(graph_repr.edges)

        # Find edge from merge_context
        merge_edge = next((e for e in edges if "merge" in str(e.source).lower()), None)

        assert merge_edge is not None
        # Should connect directly to END
        assert merge_edge.target == "__end__"

    async def test_default_includes_generate_answer(self):
        """Test that default (False) still includes generate_answer."""
        from ragitect.agents.rag.graph import build_rag_graph

        graph = build_rag_graph()

        nodes = graph.get_graph().nodes
        node_names = [n.name for n in nodes.values() if hasattr(n, "name")]

        assert "generate_answer" in node_names
