"""Chat streaming endpoint using Server-Sent Events (SSE) with RAG integration.

This module provides the SSE streaming endpoint for chat functionality with
full Retrieval-Augmented Generation (RAG) integration.

Story 3.0: Streaming Infrastructure (Prep)
Story 3.1: Natural Language Querying
Story 3.1.2: Multi-Stage Retrieval Pipeline
Story 3.2.B: Streaming LLM Responses with Citations
"""

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.api.schemas.chat import Citation

from ragitect.prompts.rag_prompts import build_rag_system_prompt
from ragitect.services.adaptive_k import select_adaptive_k
from ragitect.services.config import (
    DEFAULT_RETRIEVAL_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    RETRIEVAL_ADAPTIVE_K_GAP_THRESHOLD,
    RETRIEVAL_ADAPTIVE_K_MAX,
    RETRIEVAL_ADAPTIVE_K_MIN,
    RETRIEVAL_INITIAL_K,
    RETRIEVAL_MMR_K,
    RETRIEVAL_MMR_LAMBDA,
    RETRIEVAL_RERANKER_TOP_K,
    RETRIEVAL_USE_ADAPTIVE_K,
    RETRIEVAL_USE_MMR,
    RETRIEVAL_USE_RERANKER,
    EmbeddingConfig,
)
from ragitect.services.database.connection import get_async_session
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.repositories.vector_repo import VectorRepository
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.embedding import create_embeddings_model, embed_text
from ragitect.services.llm import generate_response_stream
from ragitect.services.llm_config_service import get_active_embedding_config
from ragitect.services.llm_factory import create_llm_with_provider
from ragitect.services.mmr import mmr_select
from ragitect.services.query_service import query_with_iterative_fallback
from ragitect.services.reranker import rerank_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces/{workspace_id}/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="User message to process")
    chat_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous messages for context. Each dict has 'role' and 'content' keys.",
    )
    provider: str | None = Field(
        None,
        description="Optional provider override. Uses that provider's configured model.",
    )


async def format_sse_stream(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Format LLM chunks as AI SDK Data Stream Protocol events.

    Implements the Vercel AI SDK UI Message Stream Protocol v1 for
    compatibility with @ai-sdk/react useChat hook.

    Protocol format:
    - data: {"type": "start", "messageId": "..."}
    - data: {"type": "text-start", "id": "..."}
    - data: {"type": "text-delta", "id": "...", "delta": "..."}
    - data: {"type": "text-end", "id": "..."}
    - data: {"type": "finish", "finishReason": "stop"}

    Args:
        chunks: Async generator yielding response chunks

    Yields:
        SSE formatted strings following Data Stream Protocol
    """
    import uuid

    message_id = str(uuid.uuid4())
    text_id = str(uuid.uuid4())

    # Message start
    yield f"data: {json.dumps({'type': 'start', 'messageId': message_id})}\n\n"

    # Text block start
    yield f"data: {json.dumps({'type': 'text-start', 'id': text_id})}\n\n"

    # Stream text deltas
    async for chunk in chunks:
        yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': chunk})}\n\n"

    # Text block end
    yield f"data: {json.dumps({'type': 'text-end', 'id': text_id})}\n\n"

    # Finish message
    yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'stop'})}\n\n"


def build_citation_metadata(context_chunks: list[dict]) -> list[Citation]:
    """Build citation metadata from context chunks.

    Story 3.2.B: Streaming LLM Responses with Citations - AC1, AC2
    Story 3.3.A: Added document_id for frontend navigation - AC1, AC2

    NOTE: Prompt engineering for [N] format was done in Story 3.1.1.
    This function just prepares metadata for frontend consumption.

    Args:
        context_chunks: Chunks returned from retrieve_context()

    Returns:
        List of Citation objects for streaming
    """
    citations = []
    for i, chunk in enumerate(context_chunks):
        citations.append(
            Citation.from_context_chunk(
                index=i,
                document_id=str(chunk.get("document_id", "")),
                document_name=chunk.get("document_name", "Unknown"),
                chunk_index=chunk.get("chunk_index", 0),
                similarity=chunk.get("rerank_score") or chunk.get("similarity", 0.0),
                content=chunk.get("content", ""),
            )
        )
    return citations


class CitationStreamParser:
    """Stateful parser for detecting citations across chunk boundaries.

    Story 3.2.B: Streaming LLM Responses with Citations - AC1, AC2

    ADR Decision: Real-time regex streaming with cross-chunk buffering.
    Handles edge case where LLM outputs '[' in one chunk and '1]' in next.

    The LLM is prompted (Story 3.1.1) to cite using [N] format matching
    [Chunk N] labels in the context. This parser detects those markers
    in the streaming output and triggers citation metadata emission.
    """

    def __init__(self, citations: list[Citation]):
        """Initialize parser with available citations.

        Args:
            citations: Pre-built citation metadata from context chunks
        """
        self.citations = citations
        self.buffer = ""  # Buffer for partial citation markers
        self.emitted_ids: set[str] = set()  # Track which citations already emitted
        self.pattern = re.compile(r"\[(\d+)\]")

    def parse_chunk(self, chunk: str) -> tuple[str, list[Citation]]:
        """Parse chunk and detect citation markers.

        Args:
            chunk: New text chunk from LLM stream

        Returns:
            Tuple of (text_to_emit, new_citations_found)
        """
        # Add chunk to buffer
        self.buffer += chunk

        # Find all complete citation markers in buffer
        new_citations = []
        for match in self.pattern.finditer(self.buffer):
            cite_idx = int(match.group(1))
            cite_id = f"cite-{cite_idx}"

            # Validate citation index (ADR: Hallucination Handling)
            if cite_idx >= len(self.citations):
                logger.warning(
                    "LLM cited non-existent source [%d] (only %d chunks available)",
                    cite_idx,
                    len(self.citations),
                )
                continue  # Graceful degradation - skip invalid citation

            # Emit each citation only once
            if cite_id not in self.emitted_ids:
                new_citations.append(self.citations[cite_idx])
                self.emitted_ids.add(cite_id)

        # Emit text, but keep last 10 chars in buffer for partial markers
        # Max citation marker length: "[9999]" = 6 chars, buffer 10 for safety
        if len(self.buffer) > 10:
            text_to_emit = self.buffer[:-10]
            self.buffer = self.buffer[-10:]
        else:
            text_to_emit = ""

        return text_to_emit, new_citations

    def flush(self) -> str:
        """Flush remaining buffer at end of stream.

        Returns:
            Any remaining text in the buffer
        """
        remaining = self.buffer
        self.buffer = ""
        return remaining


async def format_sse_stream_with_citations(
    chunks: AsyncGenerator[str, None],
    citations: list[Citation],
) -> AsyncGenerator[str, None]:
    """Format LLM chunks with AI SDK UI Message Stream Protocol v1 + citations.

    Story 3.2.B: Streaming LLM Responses with Citations - AC1, AC2

    ADR: Real-time regex streaming with cross-chunk buffering.
    Emits citations as 'source-document' parts for AI SDK useChat.

    Args:
        chunks: LLM token stream
        citations: Pre-built citation metadata

    Yields:
        SSE formatted messages (UI Message Stream Protocol v1):
        - data: {"type": "start", "messageId": "..."} - Message start
        - data: {"type": "text-start", "id": "..."} - Text block start
        - data: {"type": "text-delta", "id": "...", "delta": "..."} - Text chunks
        - data: {"type": "source-document", "sourceId": "...", ...} - Citations
        - data: {"type": "text-end", "id": "..."} - Text block end
        - data: {"type": "finish", "finishReason": "stop"} - Stream end
    """
    message_id = str(uuid.uuid4())
    text_id = str(uuid.uuid4())
    parser = CitationStreamParser(citations)

    # Message start (protocol requirement)
    yield f"data: {json.dumps({'type': 'start', 'messageId': message_id})}\n\n"

    # Text block start
    yield f"data: {json.dumps({'type': 'text-start', 'id': text_id})}\n\n"

    async for chunk in chunks:
        # Parse chunk for citations
        text_to_emit, new_citations = parser.parse_chunk(chunk)

        # Emit text delta if we have text
        if text_to_emit:
            yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': text_to_emit})}\n\n"

        # Emit source-document parts for detected citations
        for citation in new_citations:
            source_doc = citation.to_sse_dict()
            yield f"data: {json.dumps(source_doc)}\n\n"

    # Flush remaining buffer
    remaining = parser.flush()
    if remaining:
        yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': remaining})}\n\n"

    # Log citation usage for monitoring (ADR: Zero Citations case)
    if citations and not parser.emitted_ids:
        logger.info(
            "LLM response had no citations despite %d available chunks",
            len(citations),
        )

    # Text block end
    yield f"data: {json.dumps({'type': 'text-end', 'id': text_id})}\n\n"

    # Finish message
    yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'stop'})}\n\n"


async def empty_workspace_response() -> AsyncGenerator[str, None]:
    """Return SSE stream for empty workspace message using AI SDK protocol.

    Returns helpful message when user queries a workspace with no documents.
    Story 3.1: Natural Language Querying - AC6

    Yields:
        SSE formatted message following Data Stream Protocol
    """
    import uuid

    message = (
        "I don't have any documents to search in this workspace. "
        "Please upload some documents first to start asking questions about them."
    )
    message_id = str(uuid.uuid4())
    text_id = str(uuid.uuid4())

    # Message start
    yield f"data: {json.dumps({'type': 'start', 'messageId': message_id})}\n\n"

    # Text block with full message
    yield f"data: {json.dumps({'type': 'text-start', 'id': text_id})}\n\n"
    yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': message})}\n\n"
    yield f"data: {json.dumps({'type': 'text-end', 'id': text_id})}\n\n"

    # Finish message
    yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'stop'})}\n\n"


async def retrieve_context(
    session: AsyncSession,
    workspace_id: UUID,
    query: str,
    chat_history: list[dict[str, str]],
    provider: str | None = None,
    initial_k: int = RETRIEVAL_INITIAL_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    use_reranker: bool = RETRIEVAL_USE_RERANKER,
    use_mmr: bool = RETRIEVAL_USE_MMR,
    use_adaptive_k: bool = RETRIEVAL_USE_ADAPTIVE_K,
    mmr_lambda: float = RETRIEVAL_MMR_LAMBDA,
) -> list[dict]:
    """Retrieve relevant context chunks using multi-stage retrieval pipeline.

    Story 3.1.2: Multi-Stage Retrieval Pipeline

    Pipeline stages:
    1. Over-retrieve: Get top-50 candidates (AC1)
    2. Rerank: Use cross-encoder for accurate relevance scoring (AC2)
    3. MMR: Apply diversity selection to reduce redundancy (AC3)
    4. Adaptive-K: Select K based on score distribution gaps (AC4)

    Uses query_with_iterative_fallback for intelligent query processing:
    - Classifies query complexity (simple/ambiguous/complex)
    - For simple queries: tries direct search, falls back to reformulation if low relevance
    - For ambiguous/complex: reformulates directly with chat history context

    Args:
        session: Database session
        workspace_id: Workspace to search
        query: User query
        chat_history: Previous conversation for context
        provider: Optional provider override for query processing LLM
        initial_k: Number of candidates for over-retrieval (default 50)
        similarity_threshold: Minimum similarity for initial retrieval
        use_reranker: Whether to apply cross-encoder reranking
        use_mmr: Whether to apply MMR diversity selection
        use_adaptive_k: Whether to use adaptive K selection
        mmr_lambda: Balance between relevance and diversity (0-1)

    Returns:
        List of chunks with content and metadata
    """
    # Get LLM for query optimization (uses provider override if specified)
    llm = await create_llm_with_provider(session, provider=provider)

    # Get embedding configuration and create model
    embedding_config = await get_active_embedding_config(session)

    # Build EmbeddingConfig from database config
    if embedding_config:
        config = EmbeddingConfig(
            provider=embedding_config.provider_name,
            model=embedding_config.model_name or "nomic-embed-text",
            base_url=embedding_config.config_data.get("base_url"),
            api_key=embedding_config.config_data.get("api_key"),
            dimension=embedding_config.config_data.get("dimension", 768),
        )
    else:
        config = EmbeddingConfig()  # Use defaults (Ollama)

    embedding_model = create_embeddings_model(config)

    # Store search results and embeddings for pipeline stages
    search_results_cache: dict[str, list[tuple]] = {}
    query_embedding_cache: dict[str, list[float]] = {}

    # Create vector search function for iterative fallback
    async def vector_search_fn(search_query: str) -> list[str]:
        """Perform vector search and return chunk contents (caches full results)."""
        query_embedding = await embed_text(embedding_model, search_query)
        query_embedding_cache[search_query] = query_embedding
        repo = VectorRepository(session)
        # Stage 1: Over-retrieve (AC1) - get more candidates for reranking
        chunks_with_scores = await repo.search_similar_chunks(
            workspace_id,
            query_embedding,
            k=initial_k,
            similarity_threshold=similarity_threshold,
        )
        # Cache full results for later use
        search_results_cache[search_query] = chunks_with_scores
        return [chunk.content for chunk, _distance in chunks_with_scores]

    # Use iterative fallback for intelligent query processing and retrieval
    retrieved_contents, metadata = await query_with_iterative_fallback(
        llm, query, chat_history, vector_search_fn
    )

    final_query = metadata.get("final_query", query)
    logger.info(
        "Query processed: '%s' -> '%s' (classification=%s, used_reformulation=%s)",
        query,
        final_query,
        metadata.get("classification"),
        metadata.get("used_reformulation"),
    )

    # Use cached search results to avoid duplicate retrieval
    chunks_with_scores = search_results_cache.get(final_query, [])
    query_embedding = query_embedding_cache.get(final_query, [])

    # Log initial retrieval stats (AC6)
    if chunks_with_scores:
        similarities = [1.0 - dist for _, dist in chunks_with_scores]
        logger.info(
            "Initial retrieval: %d chunks, similarity range [%.3f, %.3f], mean: %.3f",
            len(chunks_with_scores),
            min(similarities),
            max(similarities),
            sum(similarities) / len(similarities),
        )

    # Format chunks for processing pipeline
    doc_repo = DocumentRepository(session)
    chunks = []
    for chunk, distance in chunks_with_scores:
        # Load the parent document to get filename
        document = await doc_repo.get_by_id(chunk.document_id)

        chunk_dict = {
            "content": chunk.content,
            "document_name": document.file_name if document else "Unknown",
            "document_id": str(chunk.document_id),
            "chunk_index": chunk.chunk_index,
            "similarity": 1.0 - distance,  # Convert distance to similarity
            "embedding": list(chunk.embedding) if chunk.embedding is not None else [],
        }
        chunks.append(chunk_dict)

    # Stage 2: Rerank with cross-encoder (AC2)
    if use_reranker and chunks:
        rerank_start = time.time()
        chunks = await rerank_chunks(
            final_query, chunks, top_k=RETRIEVAL_RERANKER_TOP_K
        )
        rerank_latency = (time.time() - rerank_start) * 1000
        logger.info(
            "Reranker latency: %.1fms for %d chunks", rerank_latency, len(chunks)
        )

    # Stage 3: MMR diversity selection (AC3)
    if use_mmr and chunks and query_embedding:
        chunk_embeddings = [c.get("embedding", []) for c in chunks]
        # Filter out chunks without embeddings
        valid_chunks = [(c, e) for c, e in zip(chunks, chunk_embeddings) if len(e) > 0]
        if valid_chunks:
            valid_chunk_list = [c for c, _ in valid_chunks]
            valid_embeddings = [e for _, e in valid_chunks]
            chunks = mmr_select(
                query_embedding=query_embedding,
                chunk_embeddings=valid_embeddings,
                chunks=valid_chunk_list,
                k=RETRIEVAL_MMR_K,
                lambda_param=mmr_lambda,
            )
            logger.info(
                "MMR selected %d diverse chunks (lambda=%.2f)", len(chunks), mmr_lambda
            )

    # Stage 4: Adaptive-K selection (AC4)
    if use_adaptive_k and chunks:
        chunks, k_metadata = select_adaptive_k(
            chunks,
            score_key="rerank_score" if use_reranker else "similarity",
            k_min=RETRIEVAL_ADAPTIVE_K_MIN,
            k_max=RETRIEVAL_ADAPTIVE_K_MAX,
            gap_threshold=RETRIEVAL_ADAPTIVE_K_GAP_THRESHOLD,
        )
        logger.info(
            "Adaptive-K: selected %d chunks (gap_found=%s)",
            k_metadata["adaptive_k"],
            k_metadata["gap_found"],
        )
    elif not use_adaptive_k:
        chunks = chunks[:DEFAULT_RETRIEVAL_K]  # Fallback to fixed K

    # Clean up: remove embedding from final results (not needed for prompt)
    results = []
    for i, chunk in enumerate(chunks):
        chunk_copy = {k: v for k, v in chunk.items() if k != "embedding"}
        chunk_copy["chunk_label"] = f"Chunk {i}"  # Zero-based for citation binding
        results.append(chunk_copy)

    logger.info("Retrieved %d context chunks after full pipeline", len(results))
    return results


def build_rag_prompt(
    user_query: str,
    context_chunks: list[dict],
    chat_history: list[dict[str, str]],
) -> list[BaseMessage]:
    """Build prompt with RAG context for LLM using modular prompt system.

    Story 3.1: Natural Language Querying - AC3
    Story 3.1.1: Retrieval Tuning & Prompt Enhancement - AC3, AC4
    Story 3.2.A: Modular Prompt System - ADR-3.2.9

    Args:
        user_query: Current user question
        context_chunks: Retrieved document chunks
        chat_history: Previous conversation

    Returns:
        List of messages for LangChain chat model
    """
    # Use modular prompt builder (Story 3.2.A)
    system_content = build_rag_system_prompt(
        context_chunks=context_chunks,
        include_citations=True,
        include_examples=True,
    )

    messages: list[BaseMessage] = [SystemMessage(content=system_content)]

    # Add chat history
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Add current query
    messages.append(HumanMessage(content=user_query))

    return messages


@router.post("/stream")
async def chat_stream(
    workspace_id: UUID,
    request: ChatRequest,
    session: AsyncSession = Depends(get_async_session),
) -> StreamingResponse:
    """Stream RAG-enhanced chat response using Server-Sent Events.

    Story 3.1: Natural Language Querying - AC1, AC2, AC3, AC4, AC6

    Args:
        workspace_id: UUID of the workspace
        request: Chat request with user message, optional chat history, and optional provider
        session: Database session

    Returns:
        StreamingResponse with SSE content type

    Raises:
        HTTPException: 404 if workspace not found, 400 if provider invalid
    """
    # Validate workspace exists
    ws_repo = WorkspaceRepository(session)
    workspace = await ws_repo.get_by_id(workspace_id)

    if not workspace:
        raise HTTPException(
            status_code=404,
            detail=f"Workspace {workspace_id} not found",
        )

    logger.info(
        f"Chat stream requested for workspace {workspace_id}, provider={request.provider}"
    )

    # Check if workspace has documents (AC6)
    doc_repo = DocumentRepository(session)
    doc_count = await doc_repo.get_by_workspace_count(workspace_id)

    if doc_count == 0:
        logger.info(f"Empty workspace {workspace_id} - returning helpful message")
        return StreamingResponse(
            empty_workspace_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "x-vercel-ai-ui-message-stream": "v1",
            },
        )

    # Retrieve context from documents (AC2)
    # Pass provider override to use consistent LLM for query processing
    context_chunks = await retrieve_context(
        session,
        workspace_id,
        request.message,
        request.chat_history,
        provider=request.provider,
    )

    # Build citation metadata from context chunks (Story 3.2.B - AC1, AC2)
    citations = build_citation_metadata(context_chunks)

    # Build RAG prompt with context (AC3)
    messages = build_rag_prompt(
        request.message,
        context_chunks,
        request.chat_history,
    )

    # Get LLM with optional provider override
    try:
        llm = await create_llm_with_provider(session, provider=request.provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate streaming response WITH citation detection (AC4, Story 3.2.B)
    async def generate():
        """Generate SSE formatted stream with citations."""
        chunks = generate_response_stream(llm, messages)
        async for sse_chunk in format_sse_stream_with_citations(chunks, citations):
            yield sse_chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )
