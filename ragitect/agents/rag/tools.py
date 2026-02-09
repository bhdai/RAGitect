"""LangGraph tools for RAG pipeline.

This module provides LangGraph @tool decorated functions for document retrieval.
Uses dependency injection patterns compatible with LangGraph's tool calling.
"""

from typing import Callable, Awaitable
from uuid import UUID

from langchain_core.tools import tool

from ragitect.services.config import RETRIEVAL_RRF_K
from ragitect.services.database.repositories.vector_repo import VectorRepository
from ragitect.agents.rag.state import ContextChunk


async def _retrieve_documents_impl(
    query: str,
    workspace_id: str,
    vector_repo: VectorRepository,
    embed_fn: Callable[[str], Awaitable[list[float]]],
    top_k: int = 50,
    query_embedding: list[float] | None = None,
) -> list[ContextChunk]:
    """Core retrieval logic - retrieve relevant document chunks for the given query.

    This is the underlying implementation used by the LangGraph tool.
    Prefixed with underscore to indicate it's internal - use retrieve_documents for LangGraph.

    Args:
        query: The search query string
        workspace_id: ID of the workspace to search within
        vector_repo: VectorRepository instance for database access
        embed_fn: Async function to generate query embeddings
        top_k: Maximum number of chunks to retrieve (default: 50)
        query_embedding: Optional pre-computed embedding to avoid regeneration

    Returns:
        List of ContextChunk dicts containing retrieved chunks with
        chunk_id, content, score, document_id, and title.
    """
    # Generate query embedding if not provided
    if query_embedding is None:
        query_embedding = await embed_fn(query)

    # Search using hybrid RRF fusion (vector + full-text search)
    chunks_with_scores = await vector_repo.hybrid_search(
        workspace_id=UUID(workspace_id),
        query_vector=query_embedding,
        query_text=query,
        k=top_k,
        rrf_k=RETRIEVAL_RRF_K,
    )

    # Convert to ContextChunk format
    # Note: hybrid_search returns (chunk, rrf_score) tuples
    # RRF score is already "higher = better", no conversion needed
    # Embedding preserved from DB to avoid redundant API calls during MMR selection
    return [
        ContextChunk(
            chunk_id=str(chunk.id),
            content=chunk.content,
            score=rrf_score,
            document_id=str(chunk.document_id),
            title="",  # Populated by graph node after document lookup
            embedding=list(chunk.embedding),
        )
        for chunk, rrf_score in chunks_with_scores
    ]


@tool
async def retrieve_documents(
    query: str,
    workspace_id: str,
    vector_repo: VectorRepository,
    embed_fn: Callable[[str], Awaitable[list[float]]],
    top_k: int = 50,
) -> list[ContextChunk]:
    """Retrieve relevant document chunks for the given query.

    This tool performs vector similarity search to find document chunks
    relevant to the query. It is designed for use in LangGraph pipelines
    where dependencies are injected at runtime.

    Args:
        query: The search query string
        workspace_id: ID of the workspace to search within
        vector_repo: VectorRepository instance for database access
        embed_fn: Async function to generate query embeddings
        top_k: Maximum number of chunks to retrieve (default: 50)

    Returns:
        List of ContextChunk dicts containing retrieved chunks with
        chunk_id, content, score, document_id, and title.
    """
    return await _retrieve_documents_impl(
        query=query,
        workspace_id=workspace_id,
        vector_repo=vector_repo,
        embed_fn=embed_fn,
        top_k=top_k,
    )
