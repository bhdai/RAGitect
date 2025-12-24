"""Tests for LangGraph retriever tool."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.tools import BaseTool

from ragitect.agents.rag.state import ContextChunk
from ragitect.agents.rag.tools import retrieve_documents, _retrieve_documents_impl


class TestRetrieveDocumentsToolDecorator:
    """Tests for @tool decorated retrieve_documents (sync tests)."""

    def test_retrieve_documents_is_langchain_tool(self):
        """retrieve_documents should be a LangChain BaseTool instance."""
        assert isinstance(retrieve_documents, BaseTool)

    def test_retrieve_documents_has_name(self):
        """retrieve_documents should have the correct name."""
        assert retrieve_documents.name == "retrieve_documents"

    def test_retrieve_documents_has_description(self):
        """retrieve_documents should have a description."""
        assert retrieve_documents.description
        assert "Retrieve relevant document chunks" in retrieve_documents.description


@pytest.mark.asyncio
class TestRetrieveDocumentsImpl:
    """Tests for retrieve_documents implementation logic."""

    async def test_retrieve_documents_returns_list_of_context_chunks(self):
        """Tool should return a list of ContextChunk dicts."""
        # Setup mocks
        workspace_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_embeddings_model = MagicMock()

        # Mock chunk returned from vector repo
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-001"
        mock_chunk.content = "Sample document content"
        mock_chunk.document_id = str(uuid4())

        # search_similar_chunks returns (chunk, distance) tuples
        mock_vector_repo.search_similar_chunks.return_value = [
            (mock_chunk, 0.1),  # distance 0.1 = similarity 0.9
        ]

        # Mock embedding function
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        # Execute tool (use impl directly for testing)
        result = await _retrieve_documents_impl(
            query="test query",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=10,
        )

        # Verify
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "chunk-001"
        assert result[0]["content"] == "Sample document content"

    async def test_retrieve_documents_returns_empty_list_when_no_results(self):
        """Tool should return empty list when no chunks found."""
        workspace_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_vector_repo.search_similar_chunks.return_value = []
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        result = await _retrieve_documents_impl(
            query="nonexistent query",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=10,
        )

        assert result == []

    async def test_retrieve_documents_converts_distance_to_similarity(self):
        """Tool should convert cosine distance to similarity score."""
        workspace_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-002"
        mock_chunk.content = "Content"
        mock_chunk.document_id = str(uuid4())

        # Distance 0.15 should become similarity 0.85
        mock_vector_repo.search_similar_chunks.return_value = [
            (mock_chunk, 0.15),
        ]

        result = await _retrieve_documents_impl(
            query="query",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=10,
        )

        # Score should be 1.0 - distance = 0.85
        assert result[0]["score"] == pytest.approx(0.85, abs=0.001)

    async def test_retrieve_documents_respects_top_k_parameter(self):
        """Tool should pass top_k to vector repo."""
        workspace_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_vector_repo.search_similar_chunks.return_value = []
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        await _retrieve_documents_impl(
            query="test",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=50,
        )

        # Verify search was called with k=50
        mock_vector_repo.search_similar_chunks.assert_awaited_once()
        call_kwargs = mock_vector_repo.search_similar_chunks.call_args.kwargs
        assert call_kwargs.get("k") == 50

    async def test_retrieve_documents_calls_embed_fn(self):
        """Tool should call embed_fn with the query."""
        workspace_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_vector_repo.search_similar_chunks.return_value = []
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        await _retrieve_documents_impl(
            query="my search query",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=10,
        )

        mock_embed_fn.assert_awaited_once_with("my search query")

    async def test_retrieve_documents_structure_matches_context_chunk(self):
        """Returned chunks should match ContextChunk TypedDict structure."""
        workspace_id = str(uuid4())
        doc_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-abc"
        mock_chunk.content = "The content of the chunk"
        mock_chunk.document_id = doc_id

        mock_vector_repo.search_similar_chunks.return_value = [
            (mock_chunk, 0.05),
        ]

        result = await _retrieve_documents_impl(
            query="test",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=10,
        )

        chunk = result[0]
        # Verify all ContextChunk keys present
        assert "chunk_id" in chunk
        assert "content" in chunk
        assert "score" in chunk
        assert "document_id" in chunk
        assert "title" in chunk

        # Verify values
        assert chunk["chunk_id"] == "chunk-abc"
        assert chunk["content"] == "The content of the chunk"
        assert chunk["document_id"] == doc_id
        assert chunk["title"] == ""  # Title populated later by graph node
        assert chunk["score"] == pytest.approx(0.95, abs=0.001)
