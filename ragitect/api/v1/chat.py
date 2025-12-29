"""Chat streaming endpoint using Server-Sent Events (SSE) with RAG integration.

This module provides the SSE streaming endpoint for chat functionality with
full Retrieval-Augmented Generation (RAG) integration using LangGraph agent-based pipeline.

The RAG pipeline uses intelligent query decomposition with parallel search execution:
- Strategy generation: LLM decomposes queries into 1-5 targeted search terms
- Parallel search: Each term searches independently with reranking, MMR, and adaptive-K
- Context merging: Deduplicate and re-rank aggregated results for final context
"""

import json
import logging
import re
import uuid
from collections.abc import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.agents.rag import build_rag_graph
from ragitect.agents.rag.state import RAGState
from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter
from ragitect.api.schemas.chat import Citation
from ragitect.prompts.rag_prompts import build_rag_system_prompt
from ragitect.services.config import EmbeddingConfig
from ragitect.services.database.connection import get_async_session
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.repositories.vector_repo import VectorRepository
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.embedding import create_embeddings_model, embed_text
from ragitect.services.llm_config_service import get_active_embedding_config
from ragitect.services.llm_factory import create_llm_with_provider


# Compile graphs once at module level (performance optimization)
# Graph compilation is expensive - do it once, reuse across requests
# Dependencies (vector_repo, embed_fn, llm) injected via state at runtime

# Retrieval-only graph: For current non-streaming chat endpoint
# Executes: generate_strategy → search_and_rank (parallel) → merge_context → END
_RAG_GRAPH_RETRIEVAL_ONLY = build_rag_graph(retrieval_only=True)

# Full graph: For streaming with LangGraphToAISDKAdapter
# Executes: generate_strategy → search_and_rank (parallel) → merge_context → generate_answer → END
_RAG_GRAPH_FULL = build_rag_graph(retrieval_only=False)

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
    # import uuid (removed redundant import)

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

    NOTE: Prompt engineering for [N] format
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
                index=i + 1,  # 1-based index
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

    ADR Decision: Real-time regex streaming with cross-chunk buffering.
    Handles edge case where LLM outputs '[cite:' in one chunk and '0]' in next.

    ADR-3.4.1: Citation format changed from [N] to [cite: N] to avoid
    false positives with markdown lists and array indices.
    """

    def __init__(self, citations: list[Citation]):
        """Initialize parser with available citations.

        Args:
            citations: Pre-built citation metadata from context chunks
        """
        self.citations = citations
        self.buffer = ""  # Buffer for partial citation markers
        self.emitted_ids: set[str] = set()  # Track which citations already emitted
        # ADR-3.4.1: Updated pattern from [N] to [cite: N] format
        self.pattern = re.compile(r"\[cite:\s*(\d+)\]")

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
            # 1-based index check
            if cite_idx < 1 or cite_idx > len(self.citations):
                logger.warning(
                    "LLM cited non-existent source [%d] (only %d chunks available)",
                    cite_idx,
                    len(self.citations),
                )
                continue  # Graceful degradation - skip invalid citation

            # Emit each citation only once
            if cite_id not in self.emitted_ids:
                # Map 1-based index to 0-based list
                new_citations.append(self.citations[cite_idx - 1])
                self.emitted_ids.add(cite_id)

        # Emit text, but keep last 20 chars in buffer for partial markers
        # Max citation marker length: "[cite: 9999]" = 12 chars, buffer 20 for safety
        if len(self.buffer) > 20:
            text_to_emit = self.buffer[:-20]
            self.buffer = self.buffer[-20:]
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

    Yields:
        SSE formatted message following Data Stream Protocol
    """
    # import uuid (removed redundant import)

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


async def retrieve_context_with_graph(
    session: AsyncSession,
    workspace_id: UUID,
    query: str,
    chat_history: list[dict[str, str]],
    provider: str | None = None,
) -> list[dict]:
    """Retrieve relevant context chunks using LangGraph-based pipeline

    This function uses the pre-compiled LangGraph StateGraph for intelligent query
    decomposition and parallel search execution:
    1. generate_strategy: Decompose query into 1-5 search terms
    2. search_and_rank: Parallel retrieval with reranking, MMR, adaptive-K
    3. merge_context: Deduplicate and re-rank aggregated results

    Performance: Graph is compiled once at module level. Request-scoped
    dependencies (vector_repo, embed_fn) are injected via state at runtime.

    Args:
        session: Database session
        workspace_id: Workspace to search
        query: User query
        chat_history: Previous conversation for context
        provider: Optional provider override for LLM

    Returns:
        List of chunks with content and metadata
    """
    # Get embedding configuration and create model
    embedding_config = await get_active_embedding_config(session)

    if embedding_config:
        config = EmbeddingConfig(
            provider=embedding_config.provider_name,
            model=embedding_config.model_name or "nomic-embed-text",
            base_url=embedding_config.config_data.get("base_url"),
            api_key=embedding_config.config_data.get("api_key"),
            dimension=embedding_config.config_data.get("dimension", 768),
        )
    else:
        config = EmbeddingConfig()

    embedding_model = create_embeddings_model(config)

    # Create async embedding function for graph
    async def embed_fn(text: str) -> list[float]:
        return await embed_text(embedding_model, text)

    # Create vector repository
    vector_repo = VectorRepository(session)

    # Create LLM with provider for graph nodes
    # This ensures generate_strategy uses the requested provider (e.g. Ollama)
    llm = await create_llm_with_provider(session, provider=provider)

    # Convert chat history to LangChain messages
    messages = []
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Build initial state with dependencies injected
    initial_state: RAGState = {
        "messages": messages,
        "original_query": query,
        "final_query": None,
        "strategy": None,
        "search_results": [],
        "context_chunks": [],
        "citations": [],
        "llm_calls": 0,
        "workspace_id": str(workspace_id),
        # Runtime dependency injection (avoiding per-request graph compilation)
        "vector_repo": vector_repo,
        "embed_fn": embed_fn,
        "llm": llm,
    }

    # Execute pre-compiled graph (strategy → search → merge, stops before generate_answer)
    # Uses module-level _RAG_GRAPH_RETRIEVAL_ONLY compiled once at import time
    result = await _RAG_GRAPH_RETRIEVAL_ONLY.ainvoke(initial_state)

    # Extract context chunks from result
    context_chunks = result.get("context_chunks", [])
    strategy = result.get("strategy")

    if strategy:
        search_terms = [s.term for s in strategy.searches]
        logger.info(
            "LangGraph retrieval: %d search terms generated: %s",
            len(strategy.searches),
            search_terms,
        )
        logger.info(
            "LangGraph retrieval: %d chunks retrieved after merge",
            len(context_chunks),
        )
    else:
        logger.warning("LangGraph retrieval: No strategy generated")

    # Format chunks for prompt (match existing retrieve_context format)
    doc_repo = DocumentRepository(session)
    results = []
    for i, chunk in enumerate(context_chunks):
        # Try to get document name from chunk or load from DB
        document_name = chunk.get("title", "Unknown")
        if document_name == "Unknown" and chunk.get("document_id"):
            try:
                doc = await doc_repo.get_by_id(UUID(chunk["document_id"]))
                if doc:
                    document_name = doc.file_name
            except (ValueError, TypeError):
                pass

        chunk_copy = {
            "content": chunk.get("content", ""),
            "document_name": document_name,
            "document_id": chunk.get("document_id", ""),
            "chunk_index": chunk.get("chunk_index", i),
            "similarity": chunk.get("score", 0.0),
            "rerank_score": chunk.get("rerank_score"),
            "chunk_label": f"Chunk {i + 1}",
        }
        results.append(chunk_copy)

    logger.info(
        "LangGraph pipeline complete: %d context chunks, %d LLM calls",
        len(results),
        result.get("llm_calls", 0),
    )
    return results


def build_rag_prompt(
    user_query: str,
    context_chunks: list[dict],
    chat_history: list[dict[str, str]],
) -> list[BaseMessage]:
    """Build prompt with RAG context for LLM using modular prompt system.

    Args:
        user_query: Current user question
        context_chunks: Retrieved document chunks
        chat_history: Previous conversation

    Returns:
        List of messages for LangChain chat model
    """
    # Use modular prompt builder
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

    # Retrieve context from documents using LangGraph-based pipeline
    logger.info("Using LangGraph full pipeline with streaming adapter")

    # Get LLM with optional provider override
    try:
        llm = await create_llm_with_provider(session, provider=request.provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Get active embedding configuration and model
    embed_config_dto = await get_active_embedding_config(session)
    if embed_config_dto is None:
        # Use default config if no active config found
        embed_config = EmbeddingConfig()
    else:
        # Convert DTO to EmbeddingConfig
        embed_config = EmbeddingConfig(
            provider=embed_config_dto.provider_name,
            model=embed_config_dto.model_name or "all-MiniLM-L6-v2",
            api_key=embed_config_dto.api_key,
            base_url=embed_config_dto.base_url,
            dimension=embed_config_dto.dimension,
        )

    embed_model = create_embeddings_model(embed_config)

    async def embed_fn(text: str) -> list[float]:
        """Embedding function for runtime dependency injection."""
        return await embed_text(embed_model, text)

    # Initialize vector repository
    vector_repo = VectorRepository(session)

    # Build chat history messages for LangGraph state
    messages = []
    for msg in request.chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Build initial state with all dependencies
    initial_state: RAGState = {
        "messages": messages,
        "original_query": request.message,
        "final_query": None,
        "strategy": None,
        "search_results": [],
        "context_chunks": [],
        "citations": [],
        "llm_calls": 0,
        "workspace_id": str(workspace_id),
        # Runtime dependency injection
        "vector_repo": vector_repo,
        "embed_fn": embed_fn,
        "llm": llm,
    }

    # Generate streaming response using LangGraphToAISDKAdapter
    async def generate():
        """Generate SSE stream from full LangGraph execution."""
        adapter = LangGraphToAISDKAdapter()
        async for sse_event in adapter.transform_stream(
            _RAG_GRAPH_FULL, dict(initial_state)
        ):
            yield sse_event

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
