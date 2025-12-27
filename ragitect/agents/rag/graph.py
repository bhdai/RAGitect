"""LangGraph StateGraph assembly for RAG pipeline.

This module assembles the RAG pipeline as a LangGraph StateGraph with:
- generate_strategy: Decomposes queries into search terms
- search_and_rank: Parallel retrieval branches via Send()
- merge_context: Aggregates and deduplicates results
- generate_answer: Generates response with citations
"""

from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ragitect.agents.rag.nodes import (
    generate_answer,
    generate_strategy,
    merge_context,
    search_and_rank,
)
from ragitect.agents.rag.state import RAGState


def continue_to_searches(state: RAGState) -> list[Send]:
    """Conditional edge: fan-out to parallel search_and_rank nodes.

    Returns a list of Send objects, each containing a search term from
    the generated strategy. LangGraph executes these in parallel.

    Maximum 5 parallel searches are enforced per AC #2.

    Args:
        state: RAGState containing the search strategy

    Returns:
        List of Send objects targeting search_and_rank with isolated sub-states
    """
    strategy = state.get("strategy")
    if strategy is None:
        return []

    # Enforce max 5 parallel searches (AC #2)
    searches = strategy.searches[:5]

    # Build sub-state for each parallel branch
    # Include workspace_id and dependencies from parent state
    workspace_id = state.get("workspace_id", "")
    vector_repo = state.get("vector_repo")
    embed_fn = state.get("embed_fn")

    return [
        Send(
            "search_and_rank",
            {
                "search_term": search.term,
                "workspace_id": workspace_id,
                "vector_repo": vector_repo,
                "embed_fn": embed_fn,
            },
        )
        for search in searches
    ]


def build_rag_graph(
    *,
    retrieval_only: bool = False,
    llm: Any = None,
):
    """Assemble and compile the RAG StateGraph.

    Creates a directed graph with the following flow:
    START → generate_strategy → [Send() fan-out] → search_and_rank (parallel)
          → merge_context → [generate_answer] → END

    Dependencies (vector_repo, embed_fn) are passed via state at runtime,
    allowing the graph to be compiled once and reused across requests.

    Args:
        retrieval_only: If True, graph ends after merge_context (skips answer gen)
        llm: Optional LLM instance to inject into nodes (for provider overrides)

    Returns:
        Compiled LangGraph Runnable
    """
    # Create StateGraph with RAGState schema
    builder = StateGraph(RAGState)

    # Add nodes
    # generate_strategy dependency injection
    # generate_strategy dependency injection (now runtime via state)
    builder.add_node("generate_strategy", generate_strategy)

    # search_and_rank now extracts dependencies from state at runtime
    builder.add_node("search_and_rank", search_and_rank)
    builder.add_node("merge_context", merge_context)

    # Add edges
    # START → generate_strategy
    builder.add_edge(START, "generate_strategy")

    # generate_strategy → [conditional fan-out] → search_and_rank
    builder.add_conditional_edges(
        "generate_strategy",
        continue_to_searches,
        ["search_and_rank"],
    )

    # search_and_rank → merge_context (all parallel branches converge)
    builder.add_edge("search_and_rank", "merge_context")

    if retrieval_only:
        # Stop after context merging (for API endpoints that do their own generation)
        builder.add_edge("merge_context", END)
    else:
        # Continue to answer generation (for autonomous agents)
        # Continue to answer generation (for autonomous agents)
        builder.add_node("generate_answer", generate_answer)
        builder.add_edge("merge_context", "generate_answer")
        builder.add_edge("generate_answer", END)

    # Compile and return
    return builder.compile()
