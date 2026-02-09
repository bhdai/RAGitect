"""Vector repository for similarity search operations"""

from sqlalchemy import func, select, type_coerce, Float
from ragitect.services.database.exceptions import NotFoundError
from ragitect.services.database.models import Document, Workspace
from ragitect.services.database.exceptions import ValidationError
from ragitect.services.database.models import DocumentChunk
from ragitect.services.config import EMBEDDING_DIMENSION
from uuid import UUID
from sqlalchemy.ext.asyncio.session import AsyncSession
import logging

logger = logging.getLogger(__name__)


class VectorRepository:
    """Repository for vector similarity search operations

    Handles all the vector similarity search operation using pgvector:
    - Chunk-level similarity search
    - Document-level similarity search
    - Configurable similarity threshold
    - Workspace-scoped searches

    Attributes:
        session: sqlalchemy AsyncSession

    Usage:
    >>>> async with get_session() as session:
    ::::    repo = VectorRepository(session)
    ::::    results = await repo.search_similar_chunks(
    ::::        workspace_id=workspace_id,
    ::::        query_vector=[0.1] * 768
    ::::        k=10
    ::::    )
    """

    session: AsyncSession

    def __init__(self, session: AsyncSession):
        """Initialize VectorRepository with an async database session.

        Args:
            session: SQLAlchemy AsyncSession for database operations.
        """
        self.session = session

    async def search_similar_chunks(
        self,
        workspace_id: UUID,
        query_vector: list[float],
        k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search for similar chunks using cosine distance.

        Note:
            Returns cosine DISTANCE (not similarity). Lower values = more similar.
            Distance range: [0.0, 2.0] where 0.0 = identical, 2.0 = opposite.
            To convert to similarity: similarity = 1.0 - distance

        Args:
            workspace_id: Workspace to search within
            query_vector: Query embedding vector (768 dims)
            k: Number of results to return
            similarity_threshold: Minimum similarity threshold (0.0-1.0)
                Internally converted to distance: distance <= (1.0 - threshold)

        Returns:
            List of (DocumentChunk, cosine_distance) tuples ordered by distance (ascending).

        Raises:
            ValidationError: If query_vector dimension is invalid
            NotFoundError: If workspace does not exist
        """
        if len(query_vector) != EMBEDDING_DIMENSION:
            raise ValidationError(
                "query_vector",
                f"Expected {EMBEDDING_DIMENSION} dimensions, got {len(query_vector)}",
            )

        # verify if workspace exists
        workspace = await self.session.get(Workspace, workspace_id)
        if workspace is None:
            raise NotFoundError("Workspace", workspace_id)

        distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label(
            "distance"
        )

        stmt = (
            select(DocumentChunk, distance_col)
            .where(DocumentChunk.workspace_id == workspace_id)
            .order_by(distance_col)
            .limit(k)
        )

        if similarity_threshold > 0.0:
            stmt = stmt.where(distance_col <= (1.0 - similarity_threshold))

        result = await self.session.execute(stmt)
        results = result.all()

        chunks_with_score = [(chunk, float(distance)) for chunk, distance in results]

        logger.info(
            f"Vector search: found {len(chunks_with_score)} chunks "
            + f"(workspace={workspace_id}, k={k}, threshold={similarity_threshold})"
        )

        if chunks_with_score:
            distances = [score for _, score in chunks_with_score]
            logger.debug(
                f"Distance range: [{min(distances):.4f}, {max(distances):.4f}], "
                + f"mean: {sum(distances) / len(distances):.4f}"
            )
        return chunks_with_score

    async def hybrid_search(
        self,
        workspace_id: UUID,
        query_vector: list[float],
        query_text: str,
        k: int = 10,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        fts_weight: float = 1.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search using hybrid RRF fusion of vector similarity and full-text search.

        Combines cosine similarity (pgvector) with PostgreSQL full-text search
        via Reciprocal Rank Fusion (RRF) in a single CTE-based SQL query.

        RRF Formula: score = Σ(weight / (k + rank_i)) for each retrieval system.

        When full-text search returns no matches, gracefully degrades to
        vector-only ranking.

        Args:
            workspace_id: Workspace to search within
            query_vector: Query embedding vector (768 dims)
            query_text: Original query text for full-text search
            k: Number of results to return
            rrf_k: RRF constant (default: 60). Higher values reduce rank impact.
            vector_weight: Weight for vector search scores (default: 1.0)
            fts_weight: Weight for full-text search scores (default: 1.0)

        Returns:
            List of (DocumentChunk, rrf_score) tuples ordered by RRF score
            (descending, higher = better).

        Raises:
            ValidationError: If query_vector dimension is invalid
            NotFoundError: If workspace does not exist
        """
        if len(query_vector) != EMBEDDING_DIMENSION:
            raise ValidationError(
                "query_vector",
                f"Expected {EMBEDDING_DIMENSION} dimensions, got {len(query_vector)}",
            )

        # Verify workspace exists
        workspace = await self.session.get(Workspace, workspace_id)
        if workspace is None:
            raise NotFoundError("Workspace", workspace_id)

        oversample = k * 3

        # CTE 1: Semantic search ranked by cosine distance (ascending = better)
        distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label(
            "distance"
        )
        semantic_cte = (
            select(
                DocumentChunk.id.label("chunk_id"),
                func.row_number()
                .over(order_by=distance_col)
                .label("semantic_rank"),
            )
            .where(DocumentChunk.workspace_id == workspace_id)
            .order_by(distance_col)
            .limit(oversample)
            .cte("semantic_search")
        )

        # CTE 2: Full-text search ranked by ts_rank_cd (descending = better)
        fts_config = "english"
        ts_vector = func.to_tsvector(fts_config, DocumentChunk.content)
        ts_query = func.plainto_tsquery(fts_config, query_text)

        keyword_cte = (
            select(
                DocumentChunk.id.label("chunk_id"),
                func.row_number()
                .over(
                    order_by=func.ts_rank_cd(ts_vector, ts_query).desc()
                )
                .label("keyword_rank"),
            )
            .where(
                DocumentChunk.workspace_id == workspace_id,
                ts_vector.op("@@")(ts_query),
            )
            .order_by(func.ts_rank_cd(ts_vector, ts_query).desc())
            .limit(oversample)
            .cte("keyword_search")
        )

        # Full outer join + RRF score computation
        # Use coalesce: if a chunk only appears in one list, the other rank is absent
        rrf_score = (
            func.coalesce(
                type_coerce(vector_weight, Float)
                / (rrf_k + semantic_cte.c.semantic_rank),
                0.0,
            )
            + func.coalesce(
                type_coerce(fts_weight, Float)
                / (rrf_k + keyword_cte.c.keyword_rank),
                0.0,
            )
        ).label("rrf_score")

        # Coalesce chunk_id from both CTEs for the join back to DocumentChunk
        chunk_id_col = func.coalesce(
            semantic_cte.c.chunk_id, keyword_cte.c.chunk_id
        ).label("chunk_id")

        # Full outer join
        fusion_query = (
            select(chunk_id_col, rrf_score)
            .select_from(
                semantic_cte.outerjoin(
                    keyword_cte,
                    semantic_cte.c.chunk_id == keyword_cte.c.chunk_id,
                    full=True,
                )
            )
            .order_by(rrf_score.desc())
            .limit(k)
            .subquery("fusion")
        )

        # Join back to DocumentChunk to get full model
        final_stmt = (
            select(DocumentChunk, fusion_query.c.rrf_score)
            .join(fusion_query, DocumentChunk.id == fusion_query.c.chunk_id)
            .order_by(fusion_query.c.rrf_score.desc())
        )

        result = await self.session.execute(final_stmt)
        results = result.all()

        chunks_with_scores = [
            (chunk, float(rrf_score)) for chunk, rrf_score in results
        ]

        logger.info(
            f"Hybrid search: found {len(chunks_with_scores)} chunks "
            + f"(workspace={workspace_id}, k={k}, rrf_k={rrf_k})"
        )

        if chunks_with_scores:
            scores = [score for _, score in chunks_with_scores]
            logger.debug(
                f"RRF score range: [{min(scores):.6f}, {max(scores):.6f}], "
                + f"mean: {sum(scores) / len(scores):.6f}"
            )

        return chunks_with_scores

    async def search_similar_documents(
        self,
        workspace_id: UUID,
        query_vector: list[float],
        k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents using document-level embedding.

        Note:
            Returns cosine DISTANCE (not similarity). Lower values = more similar.
            Distance range: [0.0, 2.0] where 0.0 = identical, 2.0 = opposite.

        Args:
            workspace_id: workspace to search within
            query_vector: Query embedding vector (768 dims)
            k: Number of results to return
            similarity_threshold: Minimum similarity threshold (0.0-1.0)
                Internally converted to distance: distance <= (1.0 - threshold)

        Returns:
            List of (Document, cosine_distance) tuples ordered by distance (ascending).

        Raises:
            ValidationError: If query_vector dimension is invalid
            NotFoundError: If workspace does not exist
        """
        if len(query_vector) != EMBEDDING_DIMENSION:
            raise ValidationError(
                "query_vector",
                f"Expected {EMBEDDING_DIMENSION} dimensions, got {len(query_vector)}",
            )

        workspace = await self.session.get(Workspace, workspace_id)
        if workspace is None:
            raise NotFoundError("Workspace", workspace_id)

        distance_col = Document.embedding.cosine_distance(query_vector).label(
            "distance"
        )

        stmt = (
            select(Document, distance_col)
            .where(
                Document.workspace_id == workspace_id, Document.embedding.is_not(None)
            )
            .order_by(distance_col)
            .limit(k)
        )

        if similarity_threshold > 0.0:
            stmt = stmt.where(distance_col <= (1.0 - similarity_threshold))

        result = await self.session.execute(stmt)
        results = result.all()

        documents_with_score = [(doc, float(distance)) for doc, distance in results]

        logger.info(
            f"Document vector search: found {len(documents_with_score)} documents "
            + f"(workspace={workspace_id}, k={k}, threshold={similarity_threshold})"
        )

        return documents_with_score

    async def get_chunk_count_by_workspace(self, workspace_id: UUID) -> int:
        """Count total chunks in a workspace

        Args:
            workspace_id: workspace UUID

        Returns:
            total number of chunks
        """
        stmt = (
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.workspace_id == workspace_id)
        )

        result = await self.session.execute(stmt)
        count = result.scalar()

        logger.debug(f"Workspace {workspace_id} has {count} document chunks")
        return count or 0

    async def get_chunk_by_document(
        self,
        document_id: UUID,
        query_vector: list[float],
        k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search similar chunks within specific document
        Useful for finding relevant sections within a single documentation

        Args:
            document_id: Document UUID
            query_vector: Query embedding vector (768 dims)
            k: number of results to return

        Returns:
            List of (DocumentChunk, similarity_score) tuples ordered by similarity.

        Raises:
            ValidationError: if query_vector dimension is invalid
        """
        if len(query_vector) != EMBEDDING_DIMENSION:
            raise ValidationError(
                "query_vector",
                f"Expected {EMBEDDING_DIMENSION} dimensions, got {len(query_vector)}",
            )

        distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label(
            "distance"
        )

        stmt = (
            select(DocumentChunk, distance_col)
            .where(DocumentChunk.document_id == document_id)
            .order_by(distance_col)
            .limit(k)
        )

        if similarity_threshold > 0.0:
            stmt = stmt.where(distance_col <= (1.0 - similarity_threshold))

        result = await self.session.execute(stmt)
        results = result.all()

        chunks_with_score = [(chunk, float(distance)) for chunk, distance in results]

        logger.debug(
            f"Found {len(chunks_with_score)} similar chunks "
            + f"(document_id={document_id}, k={k}, threshold={similarity_threshold})"
        )

        return chunks_with_score

    async def batch_search_chunks(
        self,
        workspace_id: UUID,
        query_vectors: list[list[float]],
        k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> list[list[tuple[DocumentChunk, float]]]:
        """Batch search for multiple query vector
        should be faster than running single query_vector multiple times

        Args:
            workspace_id: workspace UUID
            query_vectors: list of query embedding vectors
            k: number of results to return
            similarity_threshold: minimum similarity threshold (0.0-1.0)

        Returns:
            list of list of (DocumentChunk, similarity_score) tuples ordered by similarity.
        """
        results = []

        for query_vector in query_vectors:
            chunks_with_score = await self.search_similar_chunks(
                workspace_id=workspace_id,
                query_vector=query_vector,
                k=k,
                similarity_threshold=similarity_threshold,
            )
            results.append(chunks_with_score)

        return results
