"""Chat streaming endpoint using Server-Sent Events (SSE) with RAG integration.

This module provides the SSE streaming endpoint for chat functionality with
full Retrieval-Augmented Generation (RAG) integration.

Story 3.0: Streaming Infrastructure (Prep)
Story 3.1: Natural Language Querying
"""

import json
import logging
from collections.abc import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.services.config import (
    DEFAULT_RETRIEVAL_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    EmbeddingConfig,
)
from ragitect.services.database.connection import get_async_session
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.repositories.vector_repo import VectorRepository
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.embedding import create_embeddings_model, embed_text
from ragitect.services.llm import generate_response_stream
from ragitect.services.llm_config_service import get_active_embedding_config
from ragitect.services.llm_factory import create_llm_from_db
from ragitect.services.query_service import query_with_iterative_fallback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces/{workspace_id}/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="User message to process")
    chat_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous messages for context. Each dict has 'role' and 'content' keys.",
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
    k: int = DEFAULT_RETRIEVAL_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Retrieve relevant context chunks for RAG using iterative fallback.

    Uses query_with_iterative_fallback for intelligent query processing:
    - Classifies query complexity (simple/ambiguous/complex)
    - For simple queries: tries direct search, falls back to reformulation if low relevance
    - For ambiguous/complex: reformulates directly with chat history context

    Story 3.1: Natural Language Querying - AC2

    Args:
        session: Database session
        workspace_id: Workspace to search
        query: User query
        chat_history: Previous conversation for context
        k: Number of chunks to retrieve

    Returns:
        List of chunks with content and metadata
    """
    # Get LLM for query optimization
    llm = await create_llm_from_db(session)

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

    # Store search results to avoid duplicate retrieval
    search_results_cache: dict[str, list[tuple]] = {}

    # Create vector search function for iterative fallback
    async def vector_search_fn(search_query: str) -> list[str]:
        """Perform vector search and return chunk contents (caches full results)."""
        query_embedding = await embed_text(embedding_model, search_query)
        repo = VectorRepository(session)
        chunks_with_scores = await repo.search_similar_chunks(
            workspace_id,
            query_embedding,
            k=k,
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

    # Log similarity score distribution (AC5)
    if chunks_with_scores:
        similarities = [1.0 - dist for _, dist in chunks_with_scores]
        logger.info(
            "Retrieval stats: %d chunks, similarity range [%.3f, %.3f], mean: %.3f",
            len(chunks_with_scores),
            min(similarities),
            max(similarities),
            sum(similarities) / len(similarities),
        )

    # Format results with document info
    results = []
    doc_repo = DocumentRepository(session)
    for chunk, distance in chunks_with_scores:
        # Load the parent document to get filename
        document = await doc_repo.get_by_id(chunk.document_id)

        results.append(
            {
                "content": chunk.content,
                "document_name": document.file_name if document else "Unknown",
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "similarity": 1.0 - distance,  # Convert distance to similarity
                "chunk_label": f"Chunk {len(results) + 1}",  # For citation binding
            }
        )

    logger.info("Retrieved %d context chunks for query", len(results))
    return results


def build_rag_prompt(
    user_query: str,
    context_chunks: list[dict],
    chat_history: list[dict[str, str]],
) -> list[BaseMessage]:
    """Build prompt with RAG context for LLM using research-backed patterns.

    Story 3.1: Natural Language Querying - AC3
    Story 3.1.1: Retrieval Tuning & Prompt Enhancement - AC3, AC4

    Args:
        user_query: Current user question
        context_chunks: Retrieved document chunks
        chat_history: Previous conversation

    Returns:
        List of messages for LangChain chat model
    """
    # Format context with indexed chunks for citation binding
    if context_chunks:
        context_text = "\n\n".join(
            [
                f"[Chunk {i + 1}] (From: {chunk['document_name']}, Similarity: {chunk['similarity']:.2f})\n{chunk['content']}"
                for i, chunk in enumerate(context_chunks)
            ]
        )
    else:
        context_text = "No relevant context found in documents."

    # System prompt with research-backed RAG instructions
    system_content = f"""<system_instructions>
IDENTITY:
You are a research librarian specializing in technical documentation. Your role is to locate, organize, and accurately cite information from the user's document collection.

ABSOLUTE CONSTRAINTS:
1. USE ONLY the information within <context>. Your training data does NOT exist for this task.
2. DO NOT fabricate, infer, or extrapolate beyond what is explicitly stated.
3. If the answer cannot be found, respond: "I cannot find information about [topic] in your documents."
4. If documents contain conflicting information, present BOTH positions with their citations.

CITATION RULES:
- Cite every factual claim using [N] where N matches the chunk number from [Chunk N] labels.
- Place citations immediately after the sentence, no space: "sentence.[1]"
- Maximum 3 citations per sentence.
- ONLY cite chunks that directly support the claim.

RESPONSE FORMAT:
1. First, internally assess if <context> contains sufficient information.
2. If sufficient, provide a comprehensive answer with inline citations.
3. If partial, answer what you can and explicitly state what information is missing.
4. If insufficient, refuse politely and suggest what documents might help.

OUTPUT STYLE:
- Use markdown formatting (headers, bullets, code blocks) for readability.
- Be thorough but objective. Do not editorialize.
- Maintain a professional, journalistic tone.
</system_instructions>

<context>
{context_text}
</context>
"""

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
        request: Chat request with user message and optional chat history
        session: Database session

    Returns:
        StreamingResponse with SSE content type

    Raises:
        HTTPException: 404 if workspace not found
    """
    # Validate workspace exists
    ws_repo = WorkspaceRepository(session)
    workspace = await ws_repo.get_by_id(workspace_id)

    if not workspace:
        raise HTTPException(
            status_code=404,
            detail=f"Workspace {workspace_id} not found",
        )

    logger.info(f"Chat stream requested for workspace {workspace_id}")

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
    context_chunks = await retrieve_context(
        session,
        workspace_id,
        request.message,
        request.chat_history,
    )

    # Build RAG prompt with context (AC3)
    messages = build_rag_prompt(
        request.message,
        context_chunks,
        request.chat_history,
    )

    # Get LLM from database config
    llm = await create_llm_from_db(session)

    # Generate streaming response (AC4)
    async def generate():
        """Generate SSE formatted stream."""
        chunks = generate_response_stream(llm, messages)
        async for sse_chunk in format_sse_stream(chunks):
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
