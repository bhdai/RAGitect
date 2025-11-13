import numpy as np
from faiss import IndexFlatIP
from langchain_core.documents.base import Document


def initialize_index(dimension: int) -> IndexFlatIP:
    """initialize Faiss index

    Args:
        dimension: dimension of the vectors

    Returns:
        Faiss IndexFlatIP object
    """
    index = IndexFlatIP(dimension)
    return index


def add_vectors_to_index(index: IndexFlatIP, vectors: list[list[float]]):
    """add vectors to Faiss index

    Args:
        index: Faiss index
        vectors: list of vectors to add
    """
    if not vectors:
        print("No vectors to add.")
        return
    vectors_np = np.array(vectors, dtype=np.float32)
    index.add(vectors_np)
    print(f"Added {len(vectors)} vectors to the index.")


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
        print("Warning searching an empty index")
        return []
    query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    D, I = index.search(query_np, k)
    res = []
    for idx, score in zip(I[0], D[0]):
        document = document_store[idx]
        res.append((document, score))

    return res
