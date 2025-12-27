"""Maximum Marginal Relevance (MMR) for diverse chunk selection.

MMR balances relevance to the query with diversity from already-selected chunks,
reducing redundancy in retrieved context while maintaining relevance.

Formula: MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

Where:
- λ = 1.0: Pure relevance (no diversity consideration)
- λ = 0.0: Pure diversity (ignore query relevance after first selection)
- λ = 0.7: Default - 70% relevance, 30% diversity (recommended for RAG)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def mmr_select(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunks: list[dict],
    k: int = 10,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select diverse chunks using Maximum Marginal Relevance.

    MMR iteratively selects chunks that are both relevant to the query
    and diverse from already-selected chunks. This reduces redundancy
    in the final context while maintaining relevance.

    Args:
        query_embedding: Query vector.
        chunk_embeddings: List of chunk vectors (same order as chunks).
        chunks: List of chunk dicts.
        k: Number of chunks to select.
        lambda_param: Balance between relevance (1.0) and diversity (0.0).
                      Default 0.7 means 70% relevance, 30% diversity.

    Returns:
        Selected diverse chunks (list of chunk dicts).
    """
    if not chunks or not chunk_embeddings:
        return []

    if len(chunks) <= k:
        return chunks

    query_vec = np.array(query_embedding)
    doc_vecs = np.array(chunk_embeddings)

    # Pre-calculate norms for efficiency
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)

    # Avoid division by zero
    doc_norms[doc_norms == 0] = 1e-10
    if query_norm == 0:
        query_norm = 1e-10

    # Compute query-document similarities (relevance scores)
    # CosineSim(A, B) = Dot(A, B) / (Norm(A) * Norm(B))
    query_sims = np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)

    selected_indices: list[int] = []
    remaining_indices = list(range(len(chunks)))

    # Pre-calculate doc-doc similarities matrix if needed, but for typical RAG
    # (top-50 candidates), on-the-fly calculation is okay IF we use cached norms.
    # We will compute diversity against selected set iteratively.

    for _ in range(k):
        if not remaining_indices:
            break

        mmr_scores: list[tuple[int, float]] = []

        for idx in remaining_indices:
            relevance = query_sims[idx]

            # Compute max similarity to already selected chunks
            if selected_indices:
                # Vectorized calculation against all selected docs
                selected_docs_vecs = doc_vecs[selected_indices]
                selected_docs_norms = doc_norms[selected_indices]

                target_doc_vec = doc_vecs[idx]
                target_doc_norm = doc_norms[idx]

                # Sim(target, selected[])
                sims_to_selected = np.dot(selected_docs_vecs, target_doc_vec) / (
                    selected_docs_norms * target_doc_norm
                )
                max_sim_to_selected = np.max(sims_to_selected)
            else:
                max_sim_to_selected = 0.0

            # MMR formula: λ * relevance - (1-λ) * max_similarity_to_selected
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            mmr_scores.append((idx, mmr))

        # Select chunk with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    logger.info(
        f"MMR selected {len(selected_indices)} chunks from {len(chunks)} "
        f"(lambda={lambda_param})"
    )

    return [chunks[i] for i in selected_indices]
