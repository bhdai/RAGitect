import pytest
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4
from ragitect.services.database.repositories.vector_repo import VectorRepository
from ragitect.services.database.models import Workspace, Document, DocumentChunk
from ragitect.services.database.exceptions import NotFoundError, ValidationError


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.get = AsyncMock()
    return session


class TestVectorRepository:
    @pytest.fixture
    def repo(self, mock_session):
        return VectorRepository(mock_session)

    @pytest.mark.asyncio
    async def test_search_similar_chunks(self, repo, mock_session):
        workspace_id = uuid4()
        mock_session.get.return_value = Workspace(id=workspace_id)

        chunk = DocumentChunk(id=uuid4(), content="test")
        distance = 0.1

        mock_result = MagicMock()
        mock_result.all.return_value = [(chunk, distance)]
        mock_session.execute.return_value = mock_result

        query_vector = [0.1] * 768
        results = await repo.search_similar_chunks(workspace_id, query_vector)

        assert len(results) == 1
        assert results[0][0] == chunk
        assert results[0][1] == distance

    @pytest.mark.asyncio
    async def test_search_similar_chunks_invalid_vector(self, repo, mock_session):
        with pytest.raises(ValidationError):
            await repo.search_similar_chunks(uuid4(), [0.1] * 10)  # Wrong dim

    @pytest.mark.asyncio
    async def test_search_similar_chunks_workspace_not_found(self, repo, mock_session):
        mock_session.get.return_value = None

        with pytest.raises(NotFoundError):
            await repo.search_similar_chunks(uuid4(), [0.1] * 768)

    @pytest.mark.asyncio
    async def test_search_similar_documents(self, repo, mock_session):
        workspace_id = uuid4()
        mock_session.get.return_value = Workspace(id=workspace_id)

        doc = Document(id=uuid4(), file_name="test.pdf")
        distance = 0.2

        mock_result = MagicMock()
        mock_result.all.return_value = [(doc, distance)]
        mock_session.execute.return_value = mock_result

        query_vector = [0.1] * 768
        results = await repo.search_similar_documents(workspace_id, query_vector)

        assert len(results) == 1
        assert results[0][0] == doc
        assert results[0][1] == distance

    @pytest.mark.asyncio
    async def test_get_chunk_count_by_workspace(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 100
        mock_session.execute.return_value = mock_result

        count = await repo.get_chunk_count_by_workspace(uuid4())
        assert count == 100

    @pytest.mark.asyncio
    async def test_get_chunk_by_document(self, repo, mock_session):
        document_id = uuid4()
        chunk = DocumentChunk(id=uuid4())
        distance = 0.05

        mock_result = MagicMock()
        mock_result.all.return_value = [(chunk, distance)]
        mock_session.execute.return_value = mock_result

        results = await repo.get_chunk_by_document(document_id, [0.1] * 768)

        assert len(results) == 1
        assert results[0][0] == chunk

    @pytest.mark.asyncio
    async def test_batch_search_chunks(self, repo, mock_session):
        workspace_id = uuid4()
        mock_session.get.return_value = Workspace(id=workspace_id)

        chunk = DocumentChunk(id=uuid4())

        mock_result = MagicMock()
        mock_result.all.return_value = [(chunk, 0.1)]
        mock_session.execute.return_value = mock_result

        query_vectors = [[0.1] * 768, [0.2] * 768]
        results = await repo.batch_search_chunks(workspace_id, query_vectors)

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1


class TestVectorRepositoryIntegration:
    """Integration tests for VectorRepository"""

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def db_context(self, clean_db_manager):
        """Setup database for integration tests"""
        import os
        from sqlalchemy import text
        from ragitect.services.database import get_session
        from ragitect.services.database.connection import create_table, drop_table

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        await create_table()
        try:
            yield
        finally:
            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_similar_chunks_integration(self, clean_db_manager):
        from ragitect.services.database import get_session
        from ragitect.services.database.repositories.workspace_repo import (
            WorkspaceRepository,
        )
        from ragitect.services.database.repositories.document_repo import (
            DocumentRepository,
        )

        async with self.db_context(clean_db_manager):
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

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_similar_documents_integration(self, clean_db_manager):
        from ragitect.services.database import get_session
        from ragitect.services.database.repositories.workspace_repo import (
            WorkspaceRepository,
        )
        from ragitect.services.database.repositories.document_repo import (
            DocumentRepository,
        )

        async with self.db_context(clean_db_manager):
            async with get_session() as session:
                ws_repo = WorkspaceRepository(session)
                doc_repo = DocumentRepository(session)
                vec_repo = VectorRepository(session)

                workspace = await ws_repo.create("Vector Doc WS")

                vec1 = [0.0] * 768
                vec1[0] = 1.0

                vec2 = [0.0] * 768
                vec2[1] = 1.0

                await doc_repo.create(
                    workspace.id, "doc1.pdf", "content", embedding=vec1
                )
                await doc_repo.create(
                    workspace.id, "doc2.pdf", "content2", embedding=vec2
                )

                results = await vec_repo.search_similar_documents(
                    workspace.id, vec1, k=1
                )
                assert len(results) == 1
                assert results[0][0].file_name == "doc1.pdf"
                assert results[0][1] < 0.0001

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_batch_search_chunks_integration(self, clean_db_manager):
        from ragitect.services.database import get_session
        from ragitect.services.database.repositories.workspace_repo import (
            WorkspaceRepository,
        )
        from ragitect.services.database.repositories.document_repo import (
            DocumentRepository,
        )

        async with self.db_context(clean_db_manager):
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
