from langchain_ollama import OllamaEmbeddings
import numpy as np


def create_embeddings_model() -> OllamaEmbeddings:
    """initialize the embedding model

    Returns:
        the embedding object
    """
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embedding_model


def embed_text(model: OllamaEmbeddings, text: str) -> list[float]:
    """Embed a single text sttring

    Args:
        model: embedding model
        text: string to embed

    Returns:
        list of float (the vector)
    """
    print(f"Embedding text: {text[:30]}...")
    vector = model.embed_query(text)
    return vector


def embed_documents(model: OllamaEmbeddings, texts: list[str]) -> list[list[float]]:
    """Embeds multiple texts

    Args:
        model: embedding model
        texts: list of strings

    Returns:
        list of vectors
    """
    print(f"Embedding {len(texts)} documents in a batch...")
    return model.embed_documents(texts)


def get_embedding_dimension() -> int:
    """Embedding dimension size

    Returns:
        int: dimension size
    """
    return 768
