"""Tests for LangGraph streaming adapter.

Tests the LangGraphToAISDKAdapter that transforms LangGraph node-level
streaming events into Vercel AI SDK UI Message Stream Protocol v1 format.
"""

import json
import logging
from typing import Any, AsyncIterator

import pytest


# Module will be created in Task 2
# from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter


pytestmark = [pytest.mark.asyncio]


async def async_iter(items: list[Any]) -> AsyncIterator[Any]:
    """Helper to create async iterator from list."""
    for item in items:
        yield item


class TestSSEFormatting:
    """Test SSE event formatting."""

    async def test_format_sse_returns_correct_format(self):
        """Test _format_sse returns data: prefix and double newline."""
        # Import will fail - this is RED phase
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        adapter = LangGraphToAISDKAdapter()
        event = {"type": "start", "messageId": "test-123"}

        result = adapter._format_sse(event)

        assert result == 'data: {"type": "start", "messageId": "test-123"}\n\n'
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    async def test_format_sse_with_complex_event(self):
        """Test _format_sse handles nested objects correctly."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        adapter = LangGraphToAISDKAdapter()
        event = {
            "type": "source-document",
            "sourceId": "cite-0",
            "providerMetadata": {"ragitect": {"chunkIndex": 0}},
        }

        result = adapter._format_sse(event)

        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        # Parse JSON to verify it's valid
        json_str = result[6:-2]  # Strip "data: " and "\n\n"
        parsed = json.loads(json_str)
        assert parsed["type"] == "source-document"
        assert parsed["sourceId"] == "cite-0"


class TestStartFinishEvents:
    """Test start and finish event emission."""

    async def test_emits_start_and_text_start_at_beginning(self):
        """Test stream starts with start and text-start events after first node update."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        # Mock graph that yields at least one node update
        class MockGraph:
            def astream(self, inputs, stream_mode):
                # Emit a dummy node update to trigger start events
                return async_iter([{"some_node": {}}])

        adapter = LangGraphToAISDKAdapter()
        graph = MockGraph()

        events = [e async for e in adapter.transform_stream(graph, {})]

        # First two events should be start and text-start (emitted after first node update)
        assert len(events) >= 2
        start_event = json.loads(events[0][6:-2])  # Strip "data: " and "\n\n"
        text_start_event = json.loads(events[1][6:-2])

        assert start_event["type"] == "start"
        assert "messageId" in start_event
        assert text_start_event["type"] == "text-start"
        assert "id" in text_start_event

    async def test_emits_text_end_and_finish_at_end(self):
        """Test stream ends with text-end and finish events."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        class MockGraph:
            def astream(self, inputs, stream_mode):
                return async_iter([])

        adapter = LangGraphToAISDKAdapter()
        graph = MockGraph()

        events = [e async for e in adapter.transform_stream(graph, {})]

        # Last two events should be text-end and finish
        text_end_event = json.loads(events[-2][6:-2])
        finish_event = json.loads(events[-1][6:-2])

        assert text_end_event["type"] == "text-end"
        assert "id" in text_end_event
        assert finish_event["type"] == "finish"
        assert finish_event["finishReason"] == "stop"


class TestSourceDocumentExtraction:
    """Test source-document event extraction from merge_context node."""

    async def test_emits_source_documents_from_merge_context_node(self):
        """Test adapter emits source-document events from merge_context output."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        # Mock merge_context node output with context chunks
        class MockGraph:
            def astream(self, inputs, stream_mode):
                return async_iter(
                    [
                        {
                            "merge_context": {
                                "context_chunks": [
                                    {
                                        "document_id": "doc-1",
                                        "title": "test.pdf",
                                        "chunk_index": 0,
                                        "score": 0.95,
                                        "content": "This is chunk 0",
                                    },
                                    {
                                        "document_id": "doc-1",
                                        "title": "test.pdf",
                                        "chunk_index": 1,
                                        "score": 0.87,
                                        "content": "This is chunk 1",
                                    },
                                ]
                            }
                        }
                    ]
                )

        adapter = LangGraphToAISDKAdapter()
        graph = MockGraph()

        events = [e async for e in adapter.transform_stream(graph, {})]

        # Find source-document events
        source_docs = [
            json.loads(e[6:-2])
            for e in events
            if json.loads(e[6:-2])["type"] == "source-document"
        ]

        assert len(source_docs) == 2
        assert source_docs[0]["sourceId"] == "cite-0"
        assert source_docs[0]["title"] == "test.pdf"
        assert source_docs[1]["sourceId"] == "cite-1"


class TestTextDeltaExtraction:
    """Test text-delta event extraction from generate_answer node."""

    async def test_emits_text_delta_from_generate_answer_node(self):
        """Test adapter emits text-delta event from generate_answer output."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        # Mock generate_answer node output with AIMessage
        class AIMessage:
            def __init__(self, content):
                self.content = content

        class MockGraph:
            def astream(self, inputs, stream_mode):
                return async_iter(
                    [
                        {
                            "generate_answer": {
                                "messages": [AIMessage("This is the LLM response")]
                            }
                        }
                    ]
                )

        adapter = LangGraphToAISDKAdapter()
        graph = MockGraph()

        events = [e async for e in adapter.transform_stream(graph, {})]

        # Find text-delta event
        text_deltas = [
            json.loads(e[6:-2])
            for e in events
            if json.loads(e[6:-2])["type"] == "text-delta"
        ]

        assert len(text_deltas) == 1
        assert text_deltas[0]["delta"] == "This is the LLM response"
        assert "id" in text_deltas[0]


class TestEventOrdering:
    """Test event sequence and ordering."""

    async def test_correct_event_sequence(self):
        """Test events are emitted in correct order."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        class AIMessage:
            def __init__(self, content):
                self.content = content

        class MockGraph:
            def astream(self, inputs, stream_mode):
                # Yield nodes in graph execution order
                return async_iter(
                    [
                        {
                            "merge_context": {
                                "context_chunks": [
                                    {
                                        "document_id": "doc-1",
                                        "title": "test.pdf",
                                        "chunk_index": 0,
                                        "score": 0.95,
                                        "content": "Context",
                                    }
                                ]
                            }
                        },
                        {
                            "generate_answer": {
                                "messages": [AIMessage("Answer with [cite: 1]")]
                            }
                        },
                    ]
                )

        adapter = LangGraphToAISDKAdapter()
        graph = MockGraph()

        events = [e async for e in adapter.transform_stream(graph, {})]

        # Parse all events
        parsed_events = [json.loads(e[6:-2]) for e in events]
        event_types = [e["type"] for e in parsed_events]

        # Verify sequence
        assert event_types[0] == "start"
        assert event_types[1] == "text-start"
        assert event_types[2] == "source-document"  # Citations before text
        assert event_types[3] == "text-delta"
        assert event_types[4] == "text-end"
        assert event_types[5] == "finish"

    async def test_citations_emitted_before_text_delta(self):
        """Test citations are always emitted before text-delta (UX requirement)."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        class AIMessage:
            def __init__(self, content):
                self.content = content

        class MockGraph:
            def astream(self, inputs, stream_mode):
                return async_iter(
                    [
                        {
                            "merge_context": {
                                "context_chunks": [
                                    {
                                        "document_id": "doc-1",
                                        "title": "test.pdf",
                                        "chunk_index": 0,
                                        "score": 0.95,
                                        "content": "Context",
                                    }
                                ]
                            }
                        },
                        {"generate_answer": {"messages": [AIMessage("Answer")]}},
                    ]
                )

        adapter = LangGraphToAISDKAdapter()
        graph = MockGraph()

        events = [e async for e in adapter.transform_stream(graph, {})]
        parsed_events = [json.loads(e[6:-2]) for e in events]

        # Find indices
        source_doc_idx = next(
            i for i, e in enumerate(parsed_events) if e["type"] == "source-document"
        )
        text_delta_idx = next(
            i for i, e in enumerate(parsed_events) if e["type"] == "text-delta"
        )

        assert source_doc_idx < text_delta_idx, "Citations must appear before text"


class TestCitationValidation:
    """Test citation index validation for invalid references.

    AC #4: Invalid citation indices should be logged as warnings and not crash.
    """

    async def test_build_citations_handles_empty_chunks(self):
        """Test _build_citations_from_context handles empty list gracefully."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        adapter = LangGraphToAISDKAdapter()

        citations = adapter._build_citations_from_context([])

        assert citations == []

    async def test_build_citations_handles_missing_fields(self):
        """Test _build_citations_from_context handles chunks with missing optional fields."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        adapter = LangGraphToAISDKAdapter()

        # Minimal chunk with only required fields missing
        chunks = [
            {
                "content": "Some content",
                # Missing: document_id, title, chunk_index, score
            }
        ]

        citations = adapter._build_citations_from_context(chunks)

        assert len(citations) == 1
        assert citations[0].source_id == "cite-0"
        # Should use defaults for missing fields
        assert citations[0].title == "Unknown"

    async def test_citation_indices_are_zero_based(self):
        """Test that citation indices are 0-based (cite-0, cite-1, etc.)."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        adapter = LangGraphToAISDKAdapter()

        chunks = [
            {
                "content": "First",
                "title": "doc1.pdf",
                "document_id": "d1",
                "chunk_index": 0,
                "score": 0.9,
            },
            {
                "content": "Second",
                "title": "doc2.pdf",
                "document_id": "d2",
                "chunk_index": 1,
                "score": 0.8,
            },
            {
                "content": "Third",
                "title": "doc3.pdf",
                "document_id": "d3",
                "chunk_index": 2,
                "score": 0.7,
            },
        ]

        citations = adapter._build_citations_from_context(chunks)

        assert len(citations) == 3
        assert citations[0].source_id == "cite-0"
        assert citations[1].source_id == "cite-1"
        assert citations[2].source_id == "cite-2"

    async def test_citation_sse_format_matches_ai_sdk_protocol(self):
        """Test that Citation.to_sse_dict() produces valid AI SDK source-document format."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        adapter = LangGraphToAISDKAdapter()

        chunks = [
            {
                "content": "Test content",
                "title": "research.pdf",
                "document_id": "doc-abc-123",
                "chunk_index": 5,
                "score": 0.95,
            }
        ]

        citations = adapter._build_citations_from_context(chunks)
        sse_dict = citations[0].to_sse_dict()

        # Verify AI SDK source-document format
        assert sse_dict["type"] == "source-document"
        assert sse_dict["sourceId"] == "cite-0"
        assert sse_dict["mediaType"] == "text/plain"
        assert sse_dict["title"] == "research.pdf"

        # Verify providerMetadata.ragitect structure
        ragitect_meta = sse_dict["providerMetadata"]["ragitect"]
        assert ragitect_meta["chunkIndex"] == 5
        assert ragitect_meta["similarity"] == 0.95
        assert ragitect_meta["preview"] == "Test content"
        assert ragitect_meta["documentId"] == "doc-abc-123"

    async def test_citation_uses_rerank_score_over_score(self):
        """Test that rerank_score takes precedence over score when available."""
        from ragitect.agents.rag.streaming import LangGraphToAISDKAdapter

        adapter = LangGraphToAISDKAdapter()

        chunks = [
            {
                "content": "Test",
                "title": "doc.pdf",
                "document_id": "d1",
                "chunk_index": 0,
                "score": 0.5,  # Original vector similarity
                "rerank_score": 0.95,  # Reranker score should take precedence
            }
        ]

        citations = adapter._build_citations_from_context(chunks)
        sse_dict = citations[0].to_sse_dict()

        # Should use rerank_score (0.95) not score (0.5)
        assert sse_dict["providerMetadata"]["ragitect"]["similarity"] == 0.95
