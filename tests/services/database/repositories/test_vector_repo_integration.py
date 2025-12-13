"""Integration tests for VectorRepository.

Requirements:
    - Running PostgreSQL database (ragitect_test)
    - DATABASE_URL environment variable set
"""

import pytest
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
