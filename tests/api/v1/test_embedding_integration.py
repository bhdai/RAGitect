"""Integration tests for embedding generation in document processing.

Requirements:
    - Running PostgreSQL database (ragitect_test)
    - DATABASE_URL environment variable set
    - Ollama running with nomic-embed-text model

Tests the complete flow:
    - Upload → Parse → Embed → Ready
    - Chunk storage verification in database
    - Similarity search returns embedded chunks
"""

import asyncio
from uuid import UUID

import pytest
from httpx import AsyncClient

from ragitect.services.database.connection import get_session_factory
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.repositories.vector_repo import VectorRepository

# Module-level markers - REQUIRED for integration tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestEmbeddingIntegration:
    """Integration tests for embedding generation flow."""

    async def test_document_embedding_complete_flow(
        self,
        shared_integration_client: AsyncClient,
        test_workspace,
        clean_database,
    ):
        """Test full flow: upload → parse → embed → ready with chunks stored.

        This test verifies:
        1. Document uploads successfully
        2. Status transitions through processing → embedding → ready
        3. Chunks are stored in database with embeddings
        """
        # Arrange - create a simple text file
        file_content = b"This is test content for embedding generation. " * 20
        file_name = "embedding_test.txt"

        # Act 1 - Upload document
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents",
            files={"files": (file_name, file_content, "text/plain")},
        )

        # Assert upload successful
        assert response.status_code == 201
        data = response.json()
        assert data["total"] == 1
        document_id = data["documents"][0]["id"]

        # Act 2 - Poll until ready or timeout
        max_attempts = 30  # 30 seconds max
        final_status = None
        observed_statuses = []

        for _ in range(max_attempts):
            await asyncio.sleep(1.0)

            status_response = await shared_integration_client.get(
                f"/api/v1/workspaces/documents/{document_id}/status"
            )
            assert status_response.status_code == 200

            current_status = status_response.json()["status"]
            if current_status not in observed_statuses:
                observed_statuses.append(current_status)

            final_status = current_status

            if current_status == "ready":
                break
            elif current_status == "error":
                pytest.fail(
                    f"Document processing failed. Observed: {observed_statuses}"
                )

        # Assert - document reached ready status
        assert final_status == "ready", (
            f"Expected 'ready', got '{final_status}'. Observed: {observed_statuses}"
        )

        # Assert - embedding status was observed (if implementation includes it)
        # This will fail until implementation adds "embedding" status
        assert "embedding" in observed_statuses, (
            f"Expected 'embedding' status in flow. Observed: {observed_statuses}"
        )

        # Act 3 - Verify chunks exist in database
        session_factory = get_session_factory()
        async with session_factory() as session:
            doc_repo = DocumentRepository(session)
            chunks = await doc_repo.get_chunks(UUID(document_id))

            # Assert chunks were created
            assert len(chunks) > 0, "Expected chunks to be created"

            # Assert each chunk has embedding
            for chunk in chunks:
                assert chunk.embedding is not None, "Chunk should have embedding"
                assert len(chunk.embedding) == 768, (
                    f"Expected 768-dim embedding, got {len(chunk.embedding)}"
                )
                assert chunk.content is not None, "Chunk should have content"

    async def test_similarity_search_returns_embedded_chunks(
        self,
        shared_integration_client: AsyncClient,
        test_workspace,
        clean_database,
    ):
        """Test that similarity search can retrieve embedded chunks."""
        # Arrange - upload and process a document
        file_content = (
            b"""
        RAGitect is a Retrieval Augmented Generation system.
        It uses vector embeddings for semantic search.
        Documents are chunked and embedded locally.
        Privacy is maintained by using local embedding models.
        """
            * 5
        )  # Repeat to ensure chunking
        file_name = "rag_test.txt"

        # Upload document
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents",
            files={"files": (file_name, file_content, "text/plain")},
        )
        assert response.status_code == 201
        document_id = response.json()["documents"][0]["id"]

        # Wait for processing to complete
        max_attempts = 30
        for _ in range(max_attempts):
            await asyncio.sleep(1.0)
            status_response = await shared_integration_client.get(
                f"/api/v1/workspaces/documents/{document_id}/status"
            )
            if status_response.json()["status"] == "ready":
                break
        else:
            pytest.fail("Document not ready after 30 attempts")

        # Act - perform similarity search using VectorRepository
        session_factory = get_session_factory()
        async with session_factory() as session:
            from ragitect.services.embedding import create_embeddings_model, embed_text

            # Create query embedding
            model = create_embeddings_model()
            query_embedding = await embed_text(
                model, "vector embeddings semantic search"
            )

            # Search for similar chunks
            vector_repo = VectorRepository(session)
            results = await vector_repo.search_similar_chunks(
                workspace_id=test_workspace.id,
                query_vector=query_embedding,
                k=5,
            )

            # Assert - should find relevant chunks
            assert len(results) > 0, "Expected to find similar chunks"

            # Verify results contain relevant content (results are tuples of (DocumentChunk, score))
            contents = [chunk.content.lower() for chunk, _score in results]
            assert any(
                "embedding" in c or "vector" in c or "semantic" in c for c in contents
            ), f"Expected relevant content in results. Got: {contents}"

    async def test_multiple_documents_embedding(
        self,
        shared_integration_client: AsyncClient,
        test_workspace,
        clean_database,
    ):
        """Test embedding multiple documents simultaneously."""
        # Arrange - multiple files
        files = [
            (
                "doc1.txt",
                b"First document about machine learning and AI. " * 10,
                "text/plain",
            ),
            (
                "doc2.txt",
                b"Second document about data science and statistics. " * 10,
                "text/plain",
            ),
            (
                "doc3.md",
                b"# Third Document\n\nAbout software engineering. " * 10,
                "text/markdown",
            ),
        ]

        # Act - upload all files
        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents",
            files=[("files", file_data) for file_data in files],
        )

        # Assert upload successful
        assert response.status_code == 201
        documents = response.json()["documents"]
        assert len(documents) == 3

        # Wait for all to process
        await asyncio.sleep(5.0)

        # Check all reached ready status
        all_ready = False
        for attempt in range(25):
            ready_count = 0
            for doc in documents:
                status_response = await shared_integration_client.get(
                    f"/api/v1/workspaces/documents/{doc['id']}/status"
                )
                if status_response.json()["status"] == "ready":
                    ready_count += 1

            if ready_count == 3:
                all_ready = True
                break

            await asyncio.sleep(1.0)

        assert all_ready, "Not all documents reached ready status"

        # Verify chunks exist for all documents
        session_factory = get_session_factory()
        async with session_factory() as session:
            doc_repo = DocumentRepository(session)

            total_chunks = 0
            for doc in documents:
                chunks = await doc_repo.get_chunks(UUID(doc["id"]))
                assert len(chunks) > 0, f"Document {doc['id']} has no chunks"
                total_chunks += len(chunks)

            assert total_chunks >= 3, (
                f"Expected at least 3 total chunks, got {total_chunks}"
            )

    async def test_embedding_status_in_polling(
        self,
        shared_integration_client: AsyncClient,
        test_workspace,
        clean_database,
    ):
        """Test that 'embedding' status is exposed during processing."""
        # Arrange - upload a document
        file_content = b"Test content for status polling. " * 50
        file_name = "status_test.txt"

        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents",
            files={"files": (file_name, file_content, "text/plain")},
        )
        assert response.status_code == 201
        document_id = response.json()["documents"][0]["id"]

        # Act - poll rapidly to catch all status transitions
        observed_statuses = set()
        max_polls = 50

        for _ in range(max_polls):
            await asyncio.sleep(0.2)  # Poll every 200ms

            status_response = await shared_integration_client.get(
                f"/api/v1/workspaces/documents/{document_id}/status"
            )
            current_status = status_response.json()["status"]
            observed_statuses.add(current_status)

            if current_status == "ready":
                break

        # Assert - "embedding" status should have been observed
        # This test will fail until the implementation adds "embedding" status
        assert "embedding" in observed_statuses, (
            f"Expected to observe 'embedding' status. Observed: {observed_statuses}"
        )


class TestEmbeddingErrorHandling:
    """Integration tests for embedding error scenarios."""

    async def test_empty_document_handling(
        self,
        shared_integration_client: AsyncClient,
        test_workspace,
        clean_database,
    ):
        """Test handling of empty documents (no text to embed)."""
        # Arrange - nearly empty file
        file_content = b""
        file_name = "empty.txt"

        response = await shared_integration_client.post(
            f"/api/v1/workspaces/{test_workspace.id}/documents",
            files={"files": (file_name, file_content, "text/plain")},
        )

        # May fail at upload or processing - either is acceptable
        if response.status_code == 201:
            document_id = response.json()["documents"][0]["id"]

            # Wait for processing
            await asyncio.sleep(3.0)

            status_response = await shared_integration_client.get(
                f"/api/v1/workspaces/documents/{document_id}/status"
            )
            # Should be either ready (with no chunks) or error
            status = status_response.json()["status"]
            assert status in ["ready", "error"], f"Unexpected status: {status}"
