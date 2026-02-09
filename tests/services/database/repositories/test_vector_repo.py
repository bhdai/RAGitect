"""Unit tests for VectorRepository."""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4
from ragitect.services.database.repositories.vector_repo import VectorRepository
from ragitect.services.database.models import Workspace, Document, DocumentChunk
from ragitect.services.database.exceptions import NotFoundError, ValidationError

pytestmark = [pytest.mark.asyncio]


class TestVectorRepository:
    @pytest.fixture
    def repo(self, mock_session):
        return VectorRepository(mock_session)

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

    async def test_search_similar_chunks_invalid_vector(self, repo, mock_session):
        with pytest.raises(ValidationError):
            await repo.search_similar_chunks(uuid4(), [0.1] * 10)  # Wrong dim

    async def test_search_similar_chunks_workspace_not_found(self, repo, mock_session):
        mock_session.get.return_value = None

        with pytest.raises(NotFoundError):
            await repo.search_similar_chunks(uuid4(), [0.1] * 768)

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

    async def test_get_chunk_count_by_workspace(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 100
        mock_session.execute.return_value = mock_result

        count = await repo.get_chunk_count_by_workspace(uuid4())
        assert count == 100

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


class TestHybridSearch:
    @pytest.fixture
    def repo(self, mock_session):
        return VectorRepository(mock_session)

    async def test_hybrid_search_basic(self, repo, mock_session):
        """Test hybrid search returns correctly ordered results."""
        workspace_id = uuid4()
        mock_session.get.return_value = Workspace(id=workspace_id)

        chunk1 = DocumentChunk(id=uuid4(), content="asyncio best practices")
        chunk2 = DocumentChunk(id=uuid4(), content="python basics")
        rrf_score1 = 0.032
        rrf_score2 = 0.016

        mock_result = MagicMock()
        mock_result.all.return_value = [
            (chunk1, rrf_score1),
            (chunk2, rrf_score2),
        ]
        mock_session.execute.return_value = mock_result

        results = await repo.hybrid_search(
            workspace_id=workspace_id,
            query_vector=[0.1] * 768,
            query_text="asyncio",
            k=10,
        )

        assert len(results) == 2
        assert results[0][0] == chunk1
        assert results[0][1] == rrf_score1
        assert results[1][0] == chunk2
        assert results[1][1] == rrf_score2
        # Higher score = better
        assert results[0][1] > results[1][1]

    async def test_hybrid_search_invalid_vector(self, repo, mock_session):
        """Test that wrong vector dimension raises ValidationError."""
        with pytest.raises(ValidationError):
            await repo.hybrid_search(
                workspace_id=uuid4(),
                query_vector=[0.1] * 10,  # Wrong dimension
                query_text="test",
            )

    async def test_hybrid_search_workspace_not_found(self, repo, mock_session):
        """Test that missing workspace raises NotFoundError."""
        mock_session.get.return_value = None

        with pytest.raises(NotFoundError):
            await repo.hybrid_search(
                workspace_id=uuid4(),
                query_vector=[0.1] * 768,
                query_text="test",
            )

    async def test_hybrid_search_empty_results(self, repo, mock_session):
        """Test that no matches returns empty list."""
        workspace_id = uuid4()
        mock_session.get.return_value = Workspace(id=workspace_id)

        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        results = await repo.hybrid_search(
            workspace_id=workspace_id,
            query_vector=[0.1] * 768,
            query_text="nonexistent query",
        )

        assert results == []

    async def test_hybrid_search_vector_only_results(self, repo, mock_session):
        """Test graceful degradation when FTS returns nothing."""
        workspace_id = uuid4()
        mock_session.get.return_value = Workspace(id=workspace_id)

        # When FTS has no matches, only vector results come through
        chunk = DocumentChunk(id=uuid4(), content="semantic content only")
        # Score from vector-only RRF: 1/(60+1) ≈ 0.0164
        rrf_score = 1.0 / (60 + 1)

        mock_result = MagicMock()
        mock_result.all.return_value = [(chunk, rrf_score)]
        mock_session.execute.return_value = mock_result

        results = await repo.hybrid_search(
            workspace_id=workspace_id,
            query_vector=[0.1] * 768,
            query_text="completely unrelated terms xyz123",
        )

        assert len(results) == 1
        assert results[0][0] == chunk
        assert results[0][1] > 0  # Still has vector score

    async def test_hybrid_search_default_rrf_k(self, repo, mock_session):
        """Test that default rrf_k parameter is 60."""
        workspace_id = uuid4()
        mock_session.get.return_value = Workspace(id=workspace_id)

        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        # Call without explicit rrf_k — should use default 60
        import inspect

        sig = inspect.signature(repo.hybrid_search)
        assert sig.parameters["rrf_k"].default == 60

        await repo.hybrid_search(
            workspace_id=workspace_id,
            query_vector=[0.1] * 768,
            query_text="test",
        )
