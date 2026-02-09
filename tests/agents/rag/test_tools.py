"""Tests for LangGraph retriever tool."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.tools import BaseTool

from ragitect.agents.rag.tools import retrieve_documents, _retrieve_documents_impl
from ragitect.services.config import RETRIEVAL_RRF_K


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

        # Mock chunk returned from vector repo
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-001"
        mock_chunk.content = "Sample document content"
        mock_chunk.document_id = str(uuid4())
        mock_chunk.embedding = [0.1] * 768  # Add embedding

        # hybrid_search returns (chunk, rrf_score) tuples (higher = better)
        mock_vector_repo.hybrid_search.return_value = [
            (mock_chunk, 0.032),  # RRF score
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
        mock_vector_repo.hybrid_search.return_value = []
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        result = await _retrieve_documents_impl(
            query="nonexistent query",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=10,
        )

        assert result == []

    async def test_retrieve_documents_uses_rrf_score_directly(self):
        """Tool should use RRF score directly (higher = better)."""
        workspace_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-002"
        mock_chunk.content = "Content"
        mock_chunk.document_id = str(uuid4())
        mock_chunk.embedding = [0.2] * 768  # Add embedding

        # RRF score is used directly (no distance-to-similarity conversion)
        rrf_score = 0.032
        mock_vector_repo.hybrid_search.return_value = [
            (mock_chunk, rrf_score),
        ]

        result = await _retrieve_documents_impl(
            query="query",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=10,
        )

        # Score should be the RRF score directly
        assert result[0]["score"] == pytest.approx(rrf_score, abs=0.001)

    async def test_retrieve_documents_respects_top_k_parameter(self):
        """Tool should pass top_k to hybrid_search."""
        workspace_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_vector_repo.hybrid_search.return_value = []
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        await _retrieve_documents_impl(
            query="test",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=50,
        )

        # Verify hybrid_search was called with k=50 and rrf_k from config
        mock_vector_repo.hybrid_search.assert_awaited_once()
        call_kwargs = mock_vector_repo.hybrid_search.call_args.kwargs
        assert call_kwargs.get("k") == 50
        assert call_kwargs.get("rrf_k") == RETRIEVAL_RRF_K
        assert call_kwargs.get("query_text") == "test"

    async def test_retrieve_documents_calls_embed_fn(self):
        """Tool should call embed_fn with the query."""
        workspace_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_vector_repo.hybrid_search.return_value = []
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
        mock_chunk.embedding = [0.3] * 768  # Add embedding

        rrf_score = 0.032
        mock_vector_repo.hybrid_search.return_value = [
            (mock_chunk, rrf_score),
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
        assert "embedding" in chunk  # New field for embedding preservation

        # Verify values
        assert chunk["chunk_id"] == "chunk-abc"
        assert chunk["content"] == "The content of the chunk"
        assert chunk["document_id"] == doc_id
        assert chunk["title"] == ""  # Title populated later by graph node
        assert chunk["score"] == pytest.approx(rrf_score, abs=0.001)
        assert len(chunk["embedding"]) == 768  # Verify embedding preserved

    async def test_context_chunk_preserves_embeddings(self):
        """Embeddings from DB should be preserved in ContextChunk (AC2).

        This test verifies that embeddings retrieved from the vector database
        are properly preserved in the ContextChunk format, eliminating the need
        for redundant embedding API calls during MMR selection.
        """
        workspace_id = str(uuid4())
        mock_vector_repo = AsyncMock()
        mock_embed_fn = AsyncMock(return_value=[0.1] * 768)

        # Create mock chunks with specific embeddings to verify preservation
        expected_embedding_1 = [0.5 + i * 0.001 for i in range(768)]
        expected_embedding_2 = [0.3 + i * 0.002 for i in range(768)]

        mock_chunk_1 = MagicMock()
        mock_chunk_1.id = "chunk-embed-1"
        mock_chunk_1.content = "First chunk content"
        mock_chunk_1.document_id = str(uuid4())
        mock_chunk_1.embedding = expected_embedding_1

        mock_chunk_2 = MagicMock()
        mock_chunk_2.id = "chunk-embed-2"
        mock_chunk_2.content = "Second chunk content"
        mock_chunk_2.document_id = str(uuid4())
        mock_chunk_2.embedding = expected_embedding_2

        mock_vector_repo.hybrid_search.return_value = [
            (mock_chunk_1, 0.032),
            (mock_chunk_2, 0.016),
        ]

        result = await _retrieve_documents_impl(
            query="test embeddings",
            workspace_id=workspace_id,
            vector_repo=mock_vector_repo,
            embed_fn=mock_embed_fn,
            top_k=10,
        )

        # Verify embeddings are preserved exactly from DB
        assert len(result) == 2
        assert result[0]["embedding"] == expected_embedding_1
        assert result[1]["embedding"] == expected_embedding_2
        # Verify dimensionality
        assert len(result[0]["embedding"]) == 768
        assert len(result[1]["embedding"]) == 768
        # Verify specific values to ensure no transformation occurred
        assert result[0]["embedding"][0] == pytest.approx(0.5, abs=0.0001)
        assert result[1]["embedding"][0] == pytest.approx(0.3, abs=0.0001)
