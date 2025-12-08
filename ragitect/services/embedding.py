import numpy as np
from langchain_ollama import OllamaEmbeddings

from ragitect.services.config import EmbeddingConfig


def create_embeddings_model(config: EmbeddingConfig | None = None) -> OllamaEmbeddings:
    """Initialize the embedding model (async-ready)

    Args:
        config: Optional embedding configuration. If None, uses defaults.

    Returns:
        OllamaEmbeddings: the embedding object
    """
    if config is None:
        config = EmbeddingConfig()

    embedding_model = OllamaEmbeddings(
        model=config.model,
        base_url=config.base_url,
    )
    return embedding_model


async def embed_text(model: OllamaEmbeddings, text: str) -> list[float]:
    """Embed a single text string (async)

    Args:
        model: embedding model
        text: string to embed

    Returns:
        list of float (the vector)
    """
    print(f"Embedding text: {text[:100]}...")
    vector = await model.aembed_query(text)
    return vector


async def embed_documents(
    model: OllamaEmbeddings, texts: list[str]
) -> list[list[float]]:
    """Embeds multiple texts (async)

    Args:
        model: embedding model
        texts: list of strings

    Returns:
        list of vectors
    """
    print(f"Embedding {len(texts)} documents in a batch...")
    return await model.aembed_documents(texts)


def get_embedding_dimension(config: EmbeddingConfig | None = None) -> int:
    """Get embedding dimension size

    Args:
        config: Optional embedding configuration. If None, uses default dimension.

    Returns:
        int: dimension size
    """
    if config is None:
        return 768  # Default for nomic-embed-text
    return config.dimension
