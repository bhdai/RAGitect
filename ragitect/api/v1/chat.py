"""Chat streaming endpoint using Server-Sent Events (SSE).

This module provides the SSE streaming endpoint for chat functionality.
Infrastructure only for Story 3.0 - full RAG integration in Story 3.1.

Story 3.0: Streaming Infrastructure (Prep)
"""

import json
import logging
from collections.abc import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.services.database.connection import get_async_session
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.llm import generate_response_stream
from ragitect.services.llm_factory import create_llm_from_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces/{workspace_id}/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="User message to process")


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


@router.post("/stream")
async def chat_stream(
    workspace_id: UUID,
    request: ChatRequest,
    session: AsyncSession = Depends(get_async_session),
) -> StreamingResponse:
    """Stream chat response using Server-Sent Events.

    Args:
        workspace_id: UUID of the workspace
        request: Chat request with user message
        session: Database session

    Returns:
        StreamingResponse with SSE content type

    Raises:
        HTTPException: 404 if workspace not found
    """
    # Validate workspace exists
    repo = WorkspaceRepository(session)
    workspace = await repo.get_by_id(workspace_id)

    if not workspace:
        raise HTTPException(
            status_code=404,
            detail=f"Workspace {workspace_id} not found",
        )

    logger.info(f"Chat stream requested for workspace {workspace_id}")

    # Get LLM from database config
    llm = await create_llm_from_db(session)

    # Create message for LLM
    messages = [HumanMessage(content=request.message)]

    # Generate streaming response
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
