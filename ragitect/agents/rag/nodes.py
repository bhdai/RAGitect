"""LangGraph nodes for RAG pipeline.

This module implements the node functions used in the RAG StateGraph.
Each node takes state and returns partial state updates.
"""

from typing import TYPE_CHECKING, Any, Awaitable, Callable
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM

from ragitect.agents.rag.schemas import SearchStrategy
from ragitect.agents.rag.state import ContextChunk
from ragitect.agents.rag.tools import _retrieve_documents_impl
from ragitect.prompts.rag_prompts import build_rag_system_prompt
from ragitect.prompts.strategy_prompts import (
    build_strategy_prompt,
    format_chat_history_for_strategy,
)
from ragitect.services.adaptive_k import select_adaptive_k
from ragitect.services.config import (
    DEFAULT_LLM_MODEL,
    RETRIEVAL_ADAPTIVE_K_GAP_THRESHOLD,
    RETRIEVAL_ADAPTIVE_K_MAX,
    RETRIEVAL_ADAPTIVE_K_MIN,
    RETRIEVAL_INITIAL_K,
    RETRIEVAL_MMR_K,
    RETRIEVAL_MMR_LAMBDA,
    RETRIEVAL_RERANKER_TOP_K,
)
from ragitect.services.mmr import mmr_select
from ragitect.services.reranker import rerank_chunks

if TYPE_CHECKING:
    from ragitect.agents.rag.state import RAGState
    from ragitect.services.database.repositories.vector_repo import VectorRepository


# =============================================================================
# Node Functions
# =============================================================================


async def generate_strategy(
    state: "RAGState",
    *,
    llm: Any = None,
) -> dict:
    """Generate search strategy from user query.

    Analyzes the user's query along with chat history context to produce
    a search strategy with 1-5 targeted search terms for parallel retrieval.

    The LLM resolves pronouns and ambiguous references using conversation
    history, then decomposes complex queries into distinct search aspects.

    Args:
        state: Current RAGState containing messages and original_query
        llm: LLM instance (injected for testing). If None, creates default.

    Returns:
        Dict with 'strategy' (SearchStrategy) and 'llm_calls' (int)
    """
    if llm is None:
        # Use centralized config for default LLM model
        llm = ChatLiteLLM(model=DEFAULT_LLM_MODEL)

    # Build prompt with chat history for pronoun resolution
    query = state["original_query"]
    messages = state.get("messages", [])

    formatted_history = format_chat_history_for_strategy(messages)
    prompt = build_strategy_prompt(query, formatted_history)

    # Use structured output for type-safe strategy generation
    strategy_llm = llm.with_structured_output(SearchStrategy)

    # Generate strategy
    strategy = await strategy_llm.ainvoke([HumanMessage(content=prompt)])

    return {
        "strategy": strategy,
        "llm_calls": 1,
    }


async def search_and_rank(
    state: dict,
    *,
    vector_repo: "VectorRepository",
    embed_fn: Callable[[str], Awaitable[list[float]]],
) -> dict:
    """Execute retrieval pipeline for a single search term.

    Performs the full retrieval pipeline:
    1. Retrieve initial candidates (top_k=RETRIEVAL_INITIAL_K)
    2. Rerank with cross-encoder (top_k=RETRIEVAL_RERANKER_TOP_K)
    3. Apply MMR diversity selection (k=RETRIEVAL_MMR_K)
    4. Apply adaptive-K selection based on score gaps

    This node receives isolated sub-state from Send() containing a single
    search term. Results are aggregated via the context_chunks reducer.

    Args:
        state: Sub-state containing search_term, workspace_id
        vector_repo: VectorRepository instance for database access
        embed_fn: Async function to generate embeddings

    Returns:
        Dict with 'search_results' list for state reducer aggregation
    """
    search_term = state["search_term"]
    workspace_id = state["workspace_id"]

    # Step 1: Pre-compute query embedding for reuse (optimization)
    query_embedding = await embed_fn(search_term)

    # Step 2: Retrieve initial candidates
    chunks = await _retrieve_documents_impl(
        query=search_term,
        workspace_id=workspace_id,
        vector_repo=vector_repo,
        embed_fn=embed_fn,
        top_k=RETRIEVAL_INITIAL_K,
        query_embedding=query_embedding,
    )

    if not chunks:
        return {"search_results": []}

    # Convert ContextChunk TypedDicts to regular dicts for service functions
    chunks_as_dicts = [dict(chunk) for chunk in chunks]

    # Step 3: Rerank with cross-encoder
    reranked = await rerank_chunks(
        search_term,
        chunks_as_dicts,
        top_k=RETRIEVAL_RERANKER_TOP_K,
    )

    if not reranked:
        return {"search_results": []}

    # Step 4: Apply MMR diversity selection
    # MMR requires embeddings - generate them in parallel for performance
    # Using asyncio.gather to parallelize API calls (avoids 30 x latency)
    # NOTE: Concurrency limited by max_concurrency in search strategy (5 searches)
    # query_embedding is already computed above
    chunk_embedding_tasks = [embed_fn(chunk["content"]) for chunk in reranked]
    chunk_embeddings = await asyncio.gather(*chunk_embedding_tasks)

    mmr_selected = mmr_select(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        chunks=reranked,
        k=RETRIEVAL_MMR_K,
        lambda_param=RETRIEVAL_MMR_LAMBDA,
    )

    if not mmr_selected:
        return {"search_results": []}

    # Step 4: Apply adaptive-K selection
    final_chunks, _metadata = select_adaptive_k(
        mmr_selected,
        score_key="rerank_score",
        k_min=RETRIEVAL_ADAPTIVE_K_MIN,
        k_max=RETRIEVAL_ADAPTIVE_K_MAX,
        gap_threshold=RETRIEVAL_ADAPTIVE_K_GAP_THRESHOLD,
    )

    return {"search_results": final_chunks}


async def merge_context(state: "RAGState") -> dict:
    """Deduplicate and re-rank aggregated chunks from parallel searches.

    After parallel search_and_rank nodes converge, this node:
    1. Deduplicates chunks by chunk_id (keeps highest score)
    2. Re-ranks combined set by relevance score (descending)
    3. Limits final context to RETRIEVAL_ADAPTIVE_K_MAX chunks

    Args:
        state: RAGState containing search_results from parallel searches

    Returns:
        Dict with deduplicated, sorted, limited 'context_chunks'
    """
    all_chunks = state.get("search_results", [])

    if not all_chunks:
        return {"context_chunks": []}

    # Deduplicate by chunk_id, keeping the highest score
    # Using dict[str, ContextChunk] but ContextChunk can have extra keys from processing
    seen: dict[str, ContextChunk | dict] = {}
    for chunk in all_chunks:
        chunk_id = chunk["chunk_id"]
        # Use rerank_score if available, otherwise fall back to score
        current_score = chunk.get("rerank_score", chunk.get("score", 0))

        if chunk_id not in seen:
            seen[chunk_id] = chunk
        else:
            existing_score = seen[chunk_id].get(
                "rerank_score", seen[chunk_id].get("score", 0)
            )
            if current_score > existing_score:
                seen[chunk_id] = chunk

    # Sort by score descending
    merged = sorted(
        seen.values(),
        key=lambda c: c.get("rerank_score", c.get("score", 0)),
        reverse=True,
    )

    # Limit to top N (RETRIEVAL_ADAPTIVE_K_MAX)
    final_chunks = merged[:RETRIEVAL_ADAPTIVE_K_MAX]

    return {"context_chunks": final_chunks}


async def generate_answer(
    state: "RAGState",
    *,
    llm: Any = None,
) -> dict:
    """Generate LLM response with citations from context chunks.

    Builds a RAG prompt using the merged context chunks and generates
    a response with proper [cite: N] citations (1-based indexing).

    Args:
        state: RAGState containing context_chunks and messages
        llm: LLM instance (injected for testing). If None, creates default.

    Returns:
        Dict with 'messages' (list containing AIMessage) and 'llm_calls' (int)
    """
    if llm is None:
        # Use centralized config for default LLM model
        llm = ChatLiteLLM(model=DEFAULT_LLM_MODEL)

    context_chunks = state.get("context_chunks", [])
    original_query = state["original_query"]
    chat_history = state.get("messages", [])

    # Convert context_chunks to format expected by build_rag_system_prompt
    # The function expects chunks with 'document_name', 'similarity', 'content'
    formatted_chunks = []
    for chunk in context_chunks:
        formatted_chunks.append(
            {
                "document_name": chunk.get("title", "Unknown"),
                "similarity": chunk.get("rerank_score", chunk.get("score", 0)),
                "content": chunk.get("content", ""),
            }
        )

    # Build the RAG system prompt with context
    system_prompt = build_rag_system_prompt(
        context_chunks=formatted_chunks,
        include_citations=True,
        include_examples=True,
    )

    # Construct messages for LLM
    # Include system prompt, chat history, and the current query
    messages_for_llm = [
        SystemMessage(content=system_prompt),
        *chat_history,
        HumanMessage(content=original_query),
    ]

    # Generate response
    response = await llm.ainvoke(messages_for_llm)

    # Ensure response is AIMessage (it should be from LLM)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))

    return {
        "messages": [response],
        "llm_calls": 1,
    }
