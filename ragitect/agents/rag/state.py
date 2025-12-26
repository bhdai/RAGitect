"""LangGraph state schema for RAG pipeline.

This module defines the TypedDicts used as state in the LangGraph agent-based
RAG pipeline. Annotated types with operator.add enable LangGraph reducers for
parallel aggregation.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage

from ragitect.agents.rag.schemas import SearchStrategy
from ragitect.api.schemas.chat import Citation


class ContextChunk(TypedDict):
    """A document chunk retrieved during RAG search.

    Attributes:
        chunk_id: Unique identifier for the chunk (used for deduplication)
        content: The text content of the chunk
        score: Similarity score from vector search
        document_id: ID of the parent document
        title: Title of the parent document
    """

    chunk_id: str
    content: str
    score: float
    document_id: str
    title: str


class RAGState(TypedDict):
    """State schema for LangGraph RAG pipeline.

    Annotated fields with operator.add are LangGraph reducers that enable
    accumulation from parallel node executions.

    Attributes:
        messages: Conversation history (reducer for multi-turn)
        original_query: The user's original query text
        final_query: Optionally refined/rewritten query for retrieval
        strategy: Generated search strategy (from generate_strategy node)
        context_chunks: Retrieved chunks (reducer for parallel search aggregation)
        citations: Extracted citations for response
        llm_calls: Counter for telemetry (reducer for tracking)
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
