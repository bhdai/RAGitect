"""Cross-encoder reranking service for RAG retrieval.

Story 3.1.2: Multi-Stage Retrieval Pipeline - AC2

This module provides cross-encoder based reranking to improve retrieval
quality by reordering candidates based on query-document relevance scores.
Cross-encoders are significantly more accurate than bi-encoders for relevance
ranking but are slower since they process query-document pairs jointly.
"""

import asyncio
import logging
from functools import partial

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Default model - good balance of speed and accuracy
# Options: cross-encoder/ms-marco-MiniLM-L-6-v2 (22MB, fast)
#          cross-encoder/ms-marco-MiniLM-L-12-v2 (33MB, better)
#          BAAI/bge-reranker-base (110MB, best local)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_instance: CrossEncoder | None = None
_reranker_lock = asyncio.Lock()


def get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    """Get or create reranker instance (singleton pattern).

    Uses a singleton pattern to avoid reloading the model on every request.
    The model is loaded lazily on first call.

    Args:
        model_name: Name of the cross-encoder model to use.

    Returns:
        CrossEncoder instance ready for prediction.

    Raises:
        RuntimeError: If model fails to load.
    """
    global _reranker_instance

    if _reranker_instance is None:
        try:
            logger.info(f"Loading cross-encoder model: {model_name}")
            _reranker_instance = CrossEncoder(model_name)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {model_name}: {e}")
            raise RuntimeError(f"Could not load reranker model: {e}")

    return _reranker_instance


def _predict_sync(pairs: list[tuple[str, str]]) -> list[float]:
    """Synchronous prediction wrapper for thread execution.

    Args:
        pairs: List of (query, document) pairs to score.

    Returns:
        List of relevance scores for each pair.
    """
    reranker = get_reranker()
    scores = reranker.predict(pairs)
    return scores.tolist()


async def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 20,
) -> list[dict]:
    """Rerank chunks using cross-encoder (async wrapper).

    Runs the CPU-bound prediction in a thread pool to avoid blocking
    the asyncio event loop. Each chunk is scored for relevance to the
    query using a cross-encoder model.

    Args:
        query: User query to rank documents against.
        chunks: List of chunk dicts with 'content' key.
        top_k: Number of top results to return.

    Returns:
        Reranked chunks with 'rerank_score' added, sorted by relevance.
    """
    if not chunks:
        return []

    # Prepare query-document pairs
    pairs = [(query, chunk["content"]) for chunk in chunks]

    # Run prediction in thread pool to avoid blocking event loop
    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(None, partial(_predict_sync, pairs))

    # Add scores to chunks (create copies to avoid mutating originals)
    reranked = []
    for chunk, score in zip(chunks, scores):
        chunk_copy = chunk.copy()
        chunk_copy["rerank_score"] = float(score)
        reranked.append(chunk_copy)

    # Sort by rerank score (highest first)
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    logger.info(
        f"Reranked {len(chunks)} chunks, returning top {min(top_k, len(reranked))}"
    )
    return reranked[:top_k]
