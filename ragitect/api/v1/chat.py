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

from ragitect.services.database.connection import get_async_session
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.repositories.vector_repo import VectorRepository
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.embedding import create_embeddings_model, embed_text
from ragitect.services.llm import generate_response_stream
from ragitect.services.llm_config_service import get_active_embedding_config
from ragitect.services.llm_factory import create_llm_from_db
from ragitect.services.query_service import adaptive_query_processing

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
    """Format LLM chunks as SSE events for Vercel AI SDK.

    Args:
        chunks: Async generator yielding response chunks

    Yields:
        SSE formatted strings: "data: {...}\\n\\n"
    """
    async for chunk in chunks:
        yield f"data: {json.dumps({'text': chunk})}\n\n"
    yield "data: [DONE]\n\n"


async def empty_workspace_response() -> AsyncGenerator[str, None]:
    """Return SSE stream for empty workspace message.

    Returns helpful message when user queries a workspace with no documents.
    Story 3.1: Natural Language Querying - AC6

    Yields:
        SSE formatted message about uploading documents
    """
    message = (
        "I don't have any documents to search in this workspace. "
        "Please upload some documents first to start asking questions about them."
    )
    yield f"data: {json.dumps({'text': message})}\n\n"
    yield "data: [DONE]\n\n"


async def retrieve_context(
    session: AsyncSession,
    workspace_id: UUID,
    query: str,
    chat_history: list[dict[str, str]],
    k: int = 5,
) -> list[dict]:
    """Retrieve relevant context chunks for RAG.

    Uses query optimization and vector search to find relevant document chunks.

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

    # Process query with adaptive optimization (handles pronouns, complex queries)
    processed_query = await adaptive_query_processing(llm, query, chat_history)
    logger.info(f"Query processed: '{query}' -> '{processed_query}'")

    # Get embedding configuration and generate query embedding
    embedding_config = await get_active_embedding_config(session)

    # Build EmbeddingConfig from database config
    from ragitect.services.config import EmbeddingConfig

    if embedding_config:
        config = EmbeddingConfig(
            provider=embedding_config.provider_name,
            model=embedding_config.model_name,
            base_url=embedding_config.config_data.get("base_url"),
            api_key=embedding_config.config_data.get("api_key"),
            dimension=embedding_config.config_data.get("dimension", 768),
        )
    else:
        config = EmbeddingConfig()  # Use defaults (Ollama)

    embedding_model = create_embeddings_model(config)
    query_embedding = await embed_text(embedding_model, processed_query)

    # Search similar chunks
    repo = VectorRepository(session)
    chunks_with_scores = await repo.search_similar_chunks(
        workspace_id, query_embedding, k=k
    )

    # Format results with document info
    results = []
    for chunk, distance in chunks_with_scores:
        # Load the parent document to get filename
        doc_repo = DocumentRepository(session)
        document = await doc_repo.get_by_id(chunk.document_id)

        results.append(
            {
                "content": chunk.content,
                "document_name": document.file_name if document else "Unknown",
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "similarity": 1.0 - distance,  # Convert distance to similarity
            }
        )

    logger.info(f"Retrieved {len(results)} context chunks for query")
    return results


def build_rag_prompt(
    user_query: str,
    context_chunks: list[dict],
    chat_history: list[dict[str, str]],
) -> list[BaseMessage]:
    """Build prompt with RAG context for LLM.

    Story 3.1: Natural Language Querying - AC3

    Args:
        user_query: Current user question
        context_chunks: Retrieved document chunks
        chat_history: Previous conversation

    Returns:
        List of messages for LangChain chat model
    """
    # Format context chunks with sources
    if context_chunks:
        context_text = "\n\n".join(
            [
                f"[Source: {chunk['document_name']}]\n{chunk['content']}"
                for chunk in context_chunks
            ]
        )
    else:
        context_text = "No relevant context found in documents."

    # System prompt with RAG instructions - research-backed expert persona
    system_content = f"""You are a knowledgeable documentation expert who helps users understand their documents thoroughly.

Your response guidelines:
- Provide comprehensive, well-structured explanations that fully address the question
- Use markdown formatting (headers, bullet points, code blocks) for readability when appropriate
- Include relevant code examples or configuration snippets from the documents when helpful
- Reference source documents to support your answers (e.g., "According to [Source: filename]...")
- Explain concepts in context - don't just list bare facts
- Be thorough and helpful - users prefer detailed, complete answers over terse responses
- If multiple aspects of a topic are covered in the documents, explain them all

If the context doesn't contain enough information to fully answer the question, clearly state what information is available and what is missing.

Context from user's documents:
{context_text}
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
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
