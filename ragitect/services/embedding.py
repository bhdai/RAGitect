import logging

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from ragitect.services.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def create_embeddings_model(config: EmbeddingConfig | None = None) -> Embeddings:
    """Initialize the embedding model (async-ready) with multi-provider support

    Args:
        config: Optional embedding configuration. If None, uses defaults (Ollama).

    Returns:
        Embeddings: the embedding object (OllamaEmbeddings, OpenAIEmbeddings, or VertexAIEmbeddings)

    Raises:
        ValueError: If provider is unsupported or required config is missing
    """
    if config is None:
        config = EmbeddingConfig()

    provider = config.provider.lower()

    if provider == "ollama":
        if not config.base_url:
            raise ValueError("base_url is required for Ollama provider")
        return OllamaEmbeddings(
            model=config.model,
            base_url=config.base_url,
        )

    elif provider == "openai":
        if not config.api_key:
            raise ValueError("api_key is required for OpenAI provider")
        try:
            from langchain_openai import OpenAIEmbeddings
            from pydantic import SecretStr
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI embeddings. Install with: uv add langchain-openai"
            )
        return OpenAIEmbeddings(
            model=config.model,
            api_key=SecretStr(config.api_key),
        )

    elif provider == "vertex_ai":
        if not config.api_key:
            raise ValueError("api_key is required for Vertex AI provider")
        try:
            from langchain_google_vertexai import VertexAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-google-vertexai is required for Vertex AI. Install with: uv add langchain-google-vertexai"
            )
        return VertexAIEmbeddings(
            model=config.model,
        )

    elif provider == "openai_compatible":
        # Use Ollama client for OpenAI-compatible endpoints
        if not config.base_url:
            raise ValueError("base_url is required for OpenAI Compatible provider")
        return OllamaEmbeddings(
            model=config.model,
            base_url=config.base_url,
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. Must be one of: ollama, openai, vertex_ai, openai_compatible"
        )


async def embed_text(model: Embeddings, text: str) -> list[float]:
    """Embed a single text string (async)

    Args:
        model: embedding model (supports any LangChain Embeddings implementation)
        text: string to embed

    Returns:
        list of float (the vector)
    """
    logger.debug(f"Embedding text: {text[:100]}...")
    vector = await model.aembed_query(text)
    return vector


async def embed_documents(model: Embeddings, texts: list[str]) -> list[list[float]]:
    """Embeds multiple texts (async)

    Args:
        model: embedding model (supports any LangChain Embeddings implementation)
        texts: list of strings

    Returns:
        list of vectors
    """
    logger.info(f"Embedding {len(texts)} documents in a batch...")
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
