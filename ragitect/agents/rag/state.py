"""LangGraph state schema for RAG pipeline.

This module defines the TypedDicts used as state in the LangGraph agent-based
RAG pipeline. Annotated types with operator.add enable LangGraph reducers for
parallel aggregation.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage

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
        context_chunks: Retrieved chunks (reducer for parallel search aggregation)
        citations: Extracted citations for response
        llm_calls: Counter for telemetry (reducer for tracking)
    """

    messages: Annotated[list[AnyMessage], operator.add]
    original_query: str
    final_query: str | None
    context_chunks: Annotated[list[ContextChunk], operator.add]
    citations: list[Citation]
    llm_calls: Annotated[int, operator.add]
