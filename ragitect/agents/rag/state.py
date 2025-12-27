"""LangGraph state schema for RAG pipeline.

This module defines the TypedDicts used as state in the LangGraph agent-based
RAG pipeline. Annotated types with operator.add enable LangGraph reducers for
parallel aggregation.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, Awaitable, Callable, TypedDict

from langchain_core.messages import AnyMessage

from ragitect.agents.rag.schemas import SearchStrategy
from ragitect.api.schemas.chat import Citation

if TYPE_CHECKING:
    from ragitect.services.database.repositories.vector_repo import VectorRepository


class ContextChunk(TypedDict):
    """A document chunk retrieved during RAG search.

    Attributes:
        chunk_id: Unique identifier for the chunk (used for deduplication)
        content: The text content of the chunk
        score: Similarity score from vector search
        document_id: ID of the parent document
        title: Title of the parent document
        embedding: 768-dimensional embedding vector from database (preserved to
            avoid redundant API calls during MMR selection)
    """

    chunk_id: str
    content: str
    score: float
    document_id: str
    title: str
    embedding: list[float]


class SearchSubState(TypedDict):
    """Sub-state for parallel search_and_rank branches.

    Each parallel search receives an isolated sub-state via Send().

    Attributes:
        search_term: The search term assigned to this branch
        workspace_id: Workspace ID for search scope
    """

    search_term: str
    workspace_id: str


class RAGState(TypedDict):
    """State schema for LangGraph RAG pipeline.

    Annotated fields with operator.add are LangGraph reducers that enable
    accumulation from parallel node executions.

    Runtime dependency injection:
        vector_repo: Injected at runtime for request-scoped DB access
        embed_fn: Injected at runtime for request-scoped embedding model

    Attributes:
        messages: Conversation history (reducer for multi-turn)
        original_query: The user's original query text
        final_query: Optionally refined/rewritten query for retrieval
        strategy: Generated search strategy (from generate_strategy node)
        search_results: Accumulator for parallel search results (reducer)
        context_chunks: Final merged and deduplicated context chunks
        citations: Extracted citations for response
        llm_calls: Counter for telemetry (reducer for tracking)
        workspace_id: Workspace ID for search scope
        vector_repo: VectorRepository instance (runtime injection)
        embed_fn: Async embedding function (runtime injection)
    """

    messages: Annotated[list[AnyMessage], operator.add]
    original_query: str
    final_query: str | None
    strategy: SearchStrategy | None
    search_results: Annotated[list[ContextChunk], operator.add]
    context_chunks: list[ContextChunk]
    citations: list[Citation]
    llm_calls: Annotated[int, operator.add]
    workspace_id: str
    # Runtime dependency injection
    vector_repo: Any  # VectorRepository - use Any to avoid circular import
    embed_fn: Callable[[str], Awaitable[list[float]]]
    llm: Any  # ChatLiteLLM or similar (runtime injection)
