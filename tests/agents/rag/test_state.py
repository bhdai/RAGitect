"""Tests for RAGState TypedDict and ContextChunk schema."""

import operator

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ragitect.agents.rag.state import ContextChunk, RAGState


class TestContextChunk:
    """Tests for ContextChunk TypedDict."""

    def test_context_chunk_has_required_keys(self):
        """ContextChunk should have chunk_id, content, score, document_id, title."""
        chunk: ContextChunk = {
            "chunk_id": "chunk-001",
            "content": "Sample text content",
            "score": 0.95,
            "document_id": "doc-001",
            "title": "Document Title",
        }

        assert chunk["chunk_id"] == "chunk-001"
        assert chunk["content"] == "Sample text content"
        assert chunk["score"] == 0.95
        assert chunk["document_id"] == "doc-001"
        assert chunk["title"] == "Document Title"

    def test_context_chunk_score_is_float(self):
        """Score field should support float values."""
        chunk: ContextChunk = {
            "chunk_id": "c1",
            "content": "text",
            "score": 0.123456789,
            "document_id": "d1",
            "title": "t1",
        }
        assert isinstance(chunk["score"], float)


class TestRAGState:
    """Tests for RAGState TypedDict."""

    def test_rag_state_initialization(self):
        """RAGState should initialize with all required fields."""
        state: RAGState = {
            "messages": [],
            "original_query": "What is LangGraph?",
            "final_query": None,
            "context_chunks": [],
            "citations": [],
            "llm_calls": 0,
        }

        assert state["messages"] == []
        assert state["original_query"] == "What is LangGraph?"
        assert state["final_query"] is None
        assert state["context_chunks"] == []
        assert state["citations"] == []
        assert state["llm_calls"] == 0

    def test_rag_state_with_messages(self):
        """RAGState should accept LangChain message types."""
        state: RAGState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ],
            "original_query": "test",
            "final_query": "refined test",
            "context_chunks": [],
            "citations": [],
            "llm_calls": 1,
        }

        assert len(state["messages"]) == 2
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)

    def test_rag_state_with_context_chunks(self):
        """RAGState should accept list of ContextChunk."""
        chunks: list[ContextChunk] = [
            {
                "chunk_id": "c1",
                "content": "first chunk",
                "score": 0.9,
                "document_id": "d1",
                "title": "Doc 1",
            },
            {
                "chunk_id": "c2",
                "content": "second chunk",
                "score": 0.8,
                "document_id": "d2",
                "title": "Doc 2",
            },
        ]

        state: RAGState = {
            "messages": [],
            "original_query": "query",
            "final_query": None,
            "context_chunks": chunks,
            "citations": [],
            "llm_calls": 0,
        }

        assert len(state["context_chunks"]) == 2
        assert state["context_chunks"][0]["chunk_id"] == "c1"


class TestReducerBehavior:
    """Tests for LangGraph reducer behavior with Annotated types."""

    def test_messages_reducer_with_operator_add(self):
        """Messages list should support operator.add for reducer pattern."""
        messages1 = [HumanMessage(content="Hello")]
        messages2 = [AIMessage(content="Hi")]

        combined = operator.add(messages1, messages2)

        assert len(combined) == 2
        assert combined[0].content == "Hello"
        assert combined[1].content == "Hi"

    def test_context_chunks_reducer_with_operator_add(self):
        """Context chunks should support operator.add for parallel aggregation."""
        chunks1: list[ContextChunk] = [
            {
                "chunk_id": "c1",
                "content": "chunk 1",
                "score": 0.9,
                "document_id": "d1",
                "title": "Doc 1",
            }
        ]
        chunks2: list[ContextChunk] = [
            {
                "chunk_id": "c2",
                "content": "chunk 2",
                "score": 0.8,
                "document_id": "d2",
                "title": "Doc 2",
            }
        ]

        combined = operator.add(chunks1, chunks2)

        assert len(combined) == 2
        assert combined[0]["chunk_id"] == "c1"
        assert combined[1]["chunk_id"] == "c2"

    def test_llm_calls_reducer_with_operator_add(self):
        """llm_calls counter should support operator.add for telemetry."""
        calls1 = 1
        calls2 = 2

        combined = operator.add(calls1, calls2)

        assert combined == 3
