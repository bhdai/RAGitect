"""Adaptive-K selection based on score gap detection.

Story 3.1.2: Multi-Stage Retrieval Pipeline - AC4

This module implements dynamic K selection based on score distribution.
Instead of using a fixed K, it finds the largest gap in reranked similarity
scores and cuts there, returning between k_min and k_max chunks.

Benefits:
- Returns more chunks when all are highly relevant
- Returns fewer chunks when there's a clear quality drop-off
- Adapts to query complexity and document collection
"""

import logging

logger = logging.getLogger(__name__)


def select_adaptive_k(
    chunks: list[dict],
    score_key: str = "rerank_score",
    k_min: int = 4,
    k_max: int = 16,
    gap_threshold: float = 0.15,
) -> tuple[list[dict], dict]:
    """Select K chunks based on score distribution gaps.

    Finds the largest gap in sorted scores and cuts there, returning
    chunks above the gap. This adapts K based on the actual relevance
    distribution of retrieved chunks.

    Args:
        chunks: Sorted chunks with scores (highest score first).
        score_key: Key for score in chunk dict.
        k_min: Minimum chunks to return.
        k_max: Maximum chunks to return.
        gap_threshold: Minimum gap size to consider significant.

    Returns:
        Tuple of (selected chunks, metadata dict).
        Metadata contains: adaptive_k, gap_found, gap_size (if found).
    """
    if not chunks:
        return [], {"adaptive_k": 0, "gap_found": False}

    if len(chunks) <= k_min:
        return chunks, {"adaptive_k": len(chunks), "gap_found": False}

    # Get scores (assumes chunks are already sorted by score descending)
    # Look at k_max + some buffer to find gaps
    scores = [chunk.get(score_key, 0) for chunk in chunks[: k_max + 5]]

    if len(scores) < 2:
        return chunks[:k_min], {"adaptive_k": k_min, "gap_found": False}

    # Find gaps between consecutive scores
    # gaps[i] = (cutoff_position, gap_size)
    # cutoff_position is where we would cut (i.e., return chunks[:cutoff_position])
    gaps: list[tuple[int, float]] = []
    for i in range(len(scores) - 1):
        gap = scores[i] - scores[i + 1]
        # i+1 is the cutoff point (return first i+1 chunks)
        gaps.append((i + 1, gap))

    # Find largest gap within k_min to k_max range
    valid_gaps = [(pos, gap) for pos, gap in gaps if k_min <= pos <= k_max]

    if not valid_gaps:
        # No valid gap in range, use k_max
        result_k = min(k_max, len(chunks))
        return chunks[:result_k], {"adaptive_k": result_k, "gap_found": False}

    # Find largest gap
    best_pos, best_gap = max(valid_gaps, key=lambda x: x[1])

    if best_gap < gap_threshold:
        # Gap too small to be significant, use k_max
        result_k = min(k_max, len(chunks))
        logger.debug(
            f"Largest gap {best_gap:.3f} below threshold {gap_threshold}, using k_max={result_k}"
        )
        return chunks[:result_k], {"adaptive_k": result_k, "gap_found": False}

    logger.info(
        f"Adaptive-K: selected {best_pos} chunks (gap={best_gap:.3f} at position {best_pos})"
    )
    return chunks[:best_pos], {
        "adaptive_k": best_pos,
        "gap_found": True,
        "gap_size": best_gap,
    }
