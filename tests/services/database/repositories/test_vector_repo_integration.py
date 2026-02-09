"""Integration tests for VectorRepository.

Requirements:
    - Running PostgreSQL database (ragitect_test)
    - DATABASE_URL environment variable set
"""

import pytest
from sqlalchemy import text
from ragitect.services.database import get_session
from ragitect.services.database.repositories.vector_repo import VectorRepository
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.database.repositories.document_repo import DocumentRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestVectorRepositoryIntegration:
    """Integration tests for VectorRepository"""

    async def test_search_similar_chunks_integration(self, clean_database):
        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)
            vec_repo = VectorRepository(session)

            workspace = await ws_repo.create("Vector WS")
            doc = await doc_repo.create(workspace.id, "vec.pdf", "content")

            # Create 2 chunks with orthogonal vectors (simplified for high dim)
            # Vec 1: [1, 0, ..., 0]
            vec1 = [0.0] * 768
            vec1[0] = 1.0

            # Vec 2: [0, 1, ..., 0]
            vec2 = [0.0] * 768
            vec2[1] = 1.0

            await doc_repo.add_chunks(
                doc.id, [("chunk1", vec1, {}), ("chunk2", vec2, {})]
            )

            # Search for vec1
            results = await vec_repo.search_similar_chunks(workspace.id, vec1, k=1)
            assert len(results) == 1
            assert results[0][0].content == "chunk1"
            # Distance should be close to 0
            assert results[0][1] < 0.0001

    async def test_search_similar_documents_integration(self, clean_database):
        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)
            vec_repo = VectorRepository(session)

            workspace = await ws_repo.create("Vector Doc WS")

            vec1 = [0.0] * 768
            vec1[0] = 1.0

            vec2 = [0.0] * 768
            vec2[1] = 1.0

            await doc_repo.create(workspace.id, "doc1.pdf", "content", embedding=vec1)
            await doc_repo.create(workspace.id, "doc2.pdf", "content2", embedding=vec2)

            results = await vec_repo.search_similar_documents(workspace.id, vec1, k=1)
            assert len(results) == 1
            assert results[0][0].file_name == "doc1.pdf"
            assert results[0][1] < 0.0001

    async def test_batch_search_chunks_integration(self, clean_database):
        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)
            vec_repo = VectorRepository(session)

            workspace = await ws_repo.create("Batch WS")
            doc = await doc_repo.create(workspace.id, "batch.pdf", "content")

            vec1 = [0.0] * 768
            vec1[0] = 1.0

            await doc_repo.add_chunks(doc.id, [("c1", vec1, {})])

            # Search with same vector twice
            results = await vec_repo.batch_search_chunks(
                workspace.id, [vec1, vec1], k=1
            )
            assert len(results) == 2
            assert results[0][0][0].content == "c1"
            assert results[1][0][0].content == "c1"


class TestHybridSearchIntegration:
    """Integration tests for hybrid_search with real PostgreSQL."""

    @staticmethod
    async def _ensure_fts_index(session):
        """Create the GIN FTS index if it doesn't exist (not covered by metadata.create_all)."""
        await session.execute(
            text("""
                CREATE INDEX IF NOT EXISTS ix_document_chunks_content_fts
                ON document_chunks
                USING GIN (to_tsvector('english', content))
            """)
        )
        await session.commit()

    async def test_hybrid_search_combines_vector_and_keyword(self, clean_database):
        """Test that hybrid search combines both vector and keyword matches."""
        async with get_session() as session:
            await self._ensure_fts_index(session)

            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)
            vec_repo = VectorRepository(session)

            workspace = await ws_repo.create("Hybrid WS")
            doc = await doc_repo.create(workspace.id, "hybrid.pdf", "content")

            # Chunk 1: keyword "asyncio" + close vector
            vec1 = [0.0] * 768
            vec1[0] = 1.0
            # Chunk 2: keyword "pgvector" + different vector
            vec2 = [0.0] * 768
            vec2[1] = 1.0
            # Chunk 3: no keyword match, orthogonal vector
            vec3 = [0.0] * 768
            vec3[2] = 1.0

            await doc_repo.add_chunks(
                doc.id,
                [
                    ("asyncio event loop handling in python", vec1, {}),
                    ("pgvector extension for postgresql", vec2, {}),
                    ("general documentation without special terms", vec3, {}),
                ],
            )

            # Query: "asyncio" — should match chunk1 by keyword AND vector
            query_vec = [0.0] * 768
            query_vec[0] = 1.0  # Close to vec1

            results = await vec_repo.hybrid_search(
                workspace_id=workspace.id,
                query_vector=query_vec,
                query_text="asyncio",
                k=10,
            )

            assert len(results) >= 1
            # Chunk 1 should be top result (keyword + vector match)
            assert results[0][0].content == "asyncio event loop handling in python"
            assert results[0][1] > 0

    async def test_hybrid_search_keyword_boost(self, clean_database):
        """Test that keyword match boosts a chunk's RRF score over vector-only."""
        async with get_session() as session:
            await self._ensure_fts_index(session)

            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)
            vec_repo = VectorRepository(session)

            workspace = await ws_repo.create("Keyword Boost WS")
            doc = await doc_repo.create(workspace.id, "boost.pdf", "content")

            # Chunk 1: semantically similar (close vector) but NO keyword match
            vec1 = [0.0] * 768
            vec1[0] = 0.9
            vec1[1] = 0.1

            # Chunk 2: less similar vector BUT has exact keyword match
            vec2 = [0.0] * 768
            vec2[0] = 0.7
            vec2[2] = 0.3

            await doc_repo.add_chunks(
                doc.id,
                [
                    ("general programming concepts overview", vec1, {}),
                    ("asyncio coroutine patterns for concurrent programming", vec2, {}),
                ],
            )

            # Query vector close to vec1, but query text matches chunk2
            query_vec = [0.0] * 768
            query_vec[0] = 1.0

            results = await vec_repo.hybrid_search(
                workspace_id=workspace.id,
                query_vector=query_vec,
                query_text="asyncio coroutine",
                k=10,
            )

            assert len(results) == 2
            # Chunk 2 should be boosted above chunk 1 due to keyword match
            contents = [r[0].content for r in results]
            assert "asyncio coroutine patterns for concurrent programming" in contents

    async def test_hybrid_search_no_keyword_matches(self, clean_database):
        """Test graceful degradation to vector-only when no FTS matches."""
        async with get_session() as session:
            await self._ensure_fts_index(session)

            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)
            vec_repo = VectorRepository(session)

            workspace = await ws_repo.create("No Keyword WS")
            doc = await doc_repo.create(workspace.id, "nofts.pdf", "content")

            vec1 = [0.0] * 768
            vec1[0] = 1.0

            await doc_repo.add_chunks(
                doc.id,
                [("some content about programming", vec1, {})],
            )

            query_vec = [0.0] * 768
            query_vec[0] = 1.0

            # Query text that won't match any FTS
            results = await vec_repo.hybrid_search(
                workspace_id=workspace.id,
                query_vector=query_vec,
                query_text="xyznonexistent123",
                k=10,
            )

            # Should still return vector results (graceful degradation)
            assert len(results) == 1
            assert results[0][0].content == "some content about programming"
            assert results[0][1] > 0  # Has a valid RRF score from vector ranking

    async def test_hybrid_search_empty_workspace(self, clean_database):
        """Test hybrid search on empty workspace returns empty list."""
        async with get_session() as session:
            await self._ensure_fts_index(session)

            ws_repo = WorkspaceRepository(session)
            vec_repo = VectorRepository(session)

            workspace = await ws_repo.create("Empty Hybrid WS")

            results = await vec_repo.hybrid_search(
                workspace_id=workspace.id,
                query_vector=[0.1] * 768,
                query_text="anything",
                k=10,
            )

            assert results == []
