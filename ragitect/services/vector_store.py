import logging

import numpy as np
from faiss import IndexFlatIP
from langchain_core.documents.base import Document

logger = logging.getLogger(__name__)


def initialize_index(dimension: int) -> IndexFlatIP:
    """initialize Faiss index

    Args:
        dimension: dimension of the vectors

    Returns:
        Faiss IndexFlatIP object
    """
    index = IndexFlatIP(dimension)
    logger.info(f"Initialized Faiss index with dimension={dimension}")
    return index


def add_vectors_to_index(index: IndexFlatIP, vectors: list[list[float]]):
    """add vectors to Faiss index

    Args:
        index: Faiss index
        vectors: list of vectors to add
    """
    if not vectors:
        logger.warning("No vectors to add.")
        return
    vectors_np = np.array(vectors, dtype=np.float32)
    index.add(vectors_np)
    logger.info(f"Added {len(vectors)} vectors to index. Total vectors: {index.ntotal}")


def search_index(
    index: IndexFlatIP,
    query_vector: list[float],
    document_store: list[Document],
    k: int,
) -> list[tuple[Document, float]]:
    """search the Faiss index

    Args:
        index: index to search
        query_vector: query vector
        document_store: document store
        k: number of top results to return

    Returns:
        list of tuples (Document, score)
    """
    if index.ntotal == 0:
        logger.warning("Searching an empty index - no results will be returned")
        return []
    
    query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    D, I = index.search(query_np, k)
    
    res = []
    for idx, score in zip(I[0], D[0]):
        document = document_store[idx]
        res.append((document, score))
    
    # Log retrieval statistics
    if res:
        scores = [score for _, score in res]
        logger.info(
            f"Retrieved {len(res)} documents | "
            f"Score range: [{min(scores):.4f}, {max(scores):.4f}] | "
            f"Mean score: {np.mean(scores):.4f}"
        )
        
        # Log top result preview for debugging
        top_doc, top_score = res[0]
        preview = top_doc.page_content[:100].replace("\n", " ")
        source = top_doc.metadata.get("source", "unknown")
        logger.debug(
            f"Top result (score={top_score:.4f}, source={source}): {preview}..."
        )
    else:
        logger.warning("No documents retrieved from search")

    return res
