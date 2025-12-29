"""LangGraph streaming adapter for Vercel AI SDK.

Transforms LangGraph node-level streaming events to Vercel AI SDK UI Message Stream Protocol v1.
Uses stream_mode="updates" for node-level streaming (emits complete node outputs).

Protocol Reference:
- https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol

Event Sequence:
1. start - Stream begins
2. text-start - Text content starts
3. source-document (multiple) - Citations from merge_context node
4. text-delta - Full answer from generate_answer node
5. text-end - Text content ends
6. finish - Stream complete
"""

import json
import uuid
from typing import Any, AsyncIterator

from ragitect.api.schemas.chat import Citation


class LangGraphToAISDKAdapter:
    """Transforms LangGraph streaming events to Vercel AI SDK SSE format.

    Current Implementation: Node-Level Streaming
    ============================================
    Uses stream_mode="updates" for node-level streaming. This emits complete
    node outputs rather than individual tokens. Production RAG systems (Vercel AI
    examples, OpenAI Assistant API) successfully use node-level streaming for:
    - Simplified citation handling (no cross-chunk buffer needed)
    - Cleaner event ordering (citations always before text)
    - Reduced complexity (single chunk per node vs hundreds of tokens)

    User Experience:
    - Citations appear instantly when context is ready
    - Full answer appears when LLM generation completes
    - No character-by-character typing effect

    Future Enhancement: Token-Level Streaming
    =========================================
    If smoother UX is desired, implement token-level streaming:

    1. Change stream mode:
       ```python
       async for chunk in graph.astream(inputs, stream_mode="messages"):
       ```

    2. Parse message chunks:
       ```python
       for node_name, update in chunk.items():
           if node_name == "generate_answer":
               messages = update.get("messages", [])
               for msg in messages:
                   # msg is AIMessageChunk with incremental content
                   if hasattr(msg, "content"):
                       yield self._format_sse({
                           "type": "text-delta",
                           "id": self.text_id,
                           "delta": msg.content,  # Single token
                       })
       ```

    3. Add citation parser:
       Reintroduce CitationStreamParser from chat.py to buffer and detect
       [cite: N] markers split across token chunks (20-char lookahead).

    Reference Implementation:
    - See git history for CitationStreamParser (pre-Story 4.3)
    - Production examples: docs/research/2025-12-28-ENG-4.3-langgraph-streaming-adapter.md

    Example:
        >>> adapter = LangGraphToAISDKAdapter()
        >>> async for sse_event in adapter.transform_stream(graph, inputs):
        ...     yield sse_event
    """

    def __init__(self):
        """Initialize adapter with unique message and text IDs."""
        self.message_id = str(uuid.uuid4())
        self.text_id = str(uuid.uuid4())

    def _format_sse(self, event: dict[str, Any]) -> str:
        """Format event as SSE (Server-Sent Events).

        Args:
            event: Event dictionary to serialize

        Returns:
            SSE-formatted string with "data: " prefix and double newline
        """
        return f"data: {json.dumps(event)}\n\n"

    def _build_citations_from_context(
        self, context_chunks: list[dict[str, Any]]
    ) -> list[Citation]:
        """Build Citation objects from context chunks.

        Args:
            context_chunks: List of context chunk dicts from merge_context node

        Returns:
            List of Citation objects ready for streaming
        """
        citations = []
        for idx, chunk in enumerate(context_chunks):
            # Note: chunk comes from ContextChunk TypedDict (state.py)
            # Use "title" for document_name and "content" for chunk text
            citation = Citation.from_context_chunk(
                index=idx,
                document_id=chunk.get("document_id", ""),
                document_name=chunk.get("title", "Unknown"),
                chunk_index=chunk.get("chunk_index", idx),
                similarity=chunk.get("rerank_score", chunk.get("score", 0.0)),
                content=chunk.get("content", ""),
            )
            citations.append(citation)
        return citations

    async def transform_stream(
        self, graph: Any, inputs: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Transform LangGraph stream to AI SDK SSE events.

        Event mapping:
        - Graph start -> start, text-start (emitted after first node update)
        - merge_context node -> source-document (for each citation)
        - generate_answer node -> text-delta (full response)
        - Graph end -> text-end, finish

        Args:
            graph: Compiled LangGraph with node-level streaming
            inputs: Initial state dictionary for graph execution

        Yields:
            SSE-formatted event strings
        """
        # Track if we've emitted start events yet
        started = False

        # Stream graph node updates
        async for chunk in graph.astream(inputs, stream_mode="updates"):
            for node_name, update in chunk.items():
                # Emit start events on first node update (not before pipeline starts)
                # This keeps frontend in 'submitted' status during the whole retrieval pipeline
                if not started:
                    yield self._format_sse(
                        {"type": "start", "messageId": self.message_id}
                    )
                    yield self._format_sse({"type": "text-start", "id": self.text_id})
                    started = True

                # Handle merge_context node - emit citations
                if node_name == "merge_context":
                    context_chunks = update.get("context_chunks", [])
                    citations = self._build_citations_from_context(context_chunks)

                    # Emit each citation as source-document event
                    for citation in citations:
                        yield self._format_sse(citation.to_sse_dict())

                # Handle generate_answer node - emit full text
                elif node_name == "generate_answer":
                    messages = update.get("messages", [])
                    if messages:
                        ai_message = messages[-1]
                        full_text = ai_message.content

                        # Emit full response as single delta
                        # (Node-level streaming = complete output)
                        yield self._format_sse(
                            {
                                "type": "text-delta",
                                "id": self.text_id,
                                "delta": full_text,
                            }
                        )

        # Emit protocol end events
        yield self._format_sse({"type": "text-end", "id": self.text_id})
        yield self._format_sse({"type": "finish", "finishReason": "stop"})
