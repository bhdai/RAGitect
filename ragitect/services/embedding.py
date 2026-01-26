"""Embedding service with batch processing support.

Implements:
- Multi-provider embedding model support (Ollama, OpenAI, Vertex AI)
- Batch processing to prevent API limits with large documents
- Dimension control for Ollama models with Matryoshka support (qwen3-embedding)
"""

import logging
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from ragitect.services.config import EmbeddingConfig

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class OllamaEmbeddingsWithDimensions(Embeddings):
    """Custom Ollama embeddings wrapper with dimension control.

    This wrapper enables dimension configuration for Ollama models that support
    Matryoshka Representation Learning (MRL), such as qwen3-embedding series.

    The dimension parameter is passed directly to the Ollama API, allowing models
    to output embeddings of the specified size (e.g., 768 instead of default 1024).

    Attributes:
        model: Ollama model name (e.g., 'qwen3-embedding:0.6b')
        base_url: Ollama API base URL
        dimensions: Target embedding dimension size (e.g., 768)
        client: Async HTTP client for API calls
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        dimensions: int | None = None,
        timeout: float = 30.0,
    ):
        """Initialize Ollama embeddings with dimension support.

        Args:
            model: Ollama model identifier
            base_url: Ollama server base URL
            dimensions: Optional target dimension size. If None, uses model default.
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents asynchronously.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        url = f"{self.base_url}/api/embed"

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        # Add dimensions parameter if specified
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
            logger.debug(
                f"Embedding {len(texts)} texts with model {self.model} "
                + f"(dimensions={self.dimensions})"
            )
        else:
            logger.debug(f"Embedding {len(texts)} texts with model {self.model}")

        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        embeddings = result.get("embeddings", [])

        # Verify dimension if specified
        if embeddings and self.dimensions is not None:
            actual_dim = len(embeddings[0])
            if actual_dim != self.dimensions:
                logger.warning(
                    f"Expected {self.dimensions} dimensions but got {actual_dim}. "
                    + f"Model {self.model} may not support dimension truncation."
                )

        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query text asynchronously.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0] if embeddings else []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embed (not implemented - use async version).

        Raises:
            NotImplementedError: Always - use async methods
        """
        raise NotImplementedError(
            "Synchronous embedding not supported. Use aembed_documents instead."
        )

    def embed_query(self, text: str) -> list[float]:
        """Synchronous embed (not implemented - use async version).

        Raises:
            NotImplementedError: Always - use async methods
        """
        raise NotImplementedError(
            "Synchronous embedding not supported. Use aembed_query instead."
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup client."""
        await self.client.aclose()


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

        # Models that support dimension control via Matryoshka Representation Learning
        DIMENSION_SUPPORTED_MODELS = [
            "qwen3-embedding",  # Supports 32-1024 (0.6B), 32-4096 (4B/8B)
        ]

        # Check if model supports dimension configuration
        supports_dimensions = any(
            model_prefix in config.model.lower()
            for model_prefix in DIMENSION_SUPPORTED_MODELS
        )

        if supports_dimensions and config.dimension is not None:
            logger.info(
                f"Using Ollama embeddings with dimension control: "
                f"model={config.model}, dimensions={config.dimension}"
            )
            return OllamaEmbeddingsWithDimensions(
                model=config.model,
                base_url=config.base_url,
                dimensions=config.dimension,
            )
        else:
            # Standard Ollama embeddings without dimension control
            logger.info(f"Using standard Ollama embeddings: model={config.model}")
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
            dimensions=config.dimension,
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


async def embed_documents(
    model: Embeddings, texts: list[str], batch_size: int = 32
) -> list[list[float]]:
    """Embeds multiple texts with batching to avoid API limits.

    Args:
        model: embedding model (supports any LangChain Embeddings implementation)
        texts: list of strings
        batch_size: max texts to embed per API call (default: 32)

    Returns:
        list of vectors

    Note:
        Batching prevents API limit errors with large documents.
        - Ollama: Quality degrades beyond 16 embeddings/batch
        - OpenAI: Max 2048 embeddings/request
        - Default 32 is conservative for most providers
    """
    if not texts:
        return []

    # Small dataset - embed all at once
    if len(texts) <= batch_size:
        logger.info(f"Embedding {len(texts)} documents in a single batch...")
        return await model.aembed_documents(texts)

    # Large dataset - batch processing
    logger.info(f"Embedding {len(texts)} documents in batches of {batch_size}...")

    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.debug(
            f"Processing batch {batch_num}/{total_batches}: {len(batch)} texts"
        )

        batch_embeddings = await model.aembed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    logger.info(f"Generated {len(all_embeddings)} embeddings total")
    return all_embeddings


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


async def get_embedding_model_from_config(session: "AsyncSession") -> Embeddings:
    """Get embedding model using active DB config, fallback to env vars, then defaults.

    Priority order:
    1. Active embedding config from database (UI-configured)
    2. Environment variables (EMBEDDING_PROVIDER, EMBEDDING_API_KEY, etc.)
    3. Hardcoded defaults (Ollama)

    This is the DRY helper that encapsulates embedding config lookup, avoiding
    duplicate patterns across chat.py and document_processing_service.py.

    Args:
        session: Database session for config lookup

    Returns:
        Embeddings model configured from DB, env vars, or defaults
    """
    # Import inside function to avoid circular imports
    from ragitect.services.llm_config_service import get_active_embedding_config
    from ragitect.services.config import load_embedding_config

    embedding_config = await get_active_embedding_config(session)
    if embedding_config:
        logger.info(
            f"Using embedding config from DB: provider={embedding_config.provider_name}, "
            f"model={embedding_config.model_name}"
        )
        config = EmbeddingConfig(
            provider=embedding_config.provider_name,
            model=embedding_config.model_name or "qwen3-embedding:0.6b",
            base_url=embedding_config.base_url,
            api_key=embedding_config.api_key,
            dimension=embedding_config.dimension,
        )
    else:
        # Fall back to environment variables
        config = load_embedding_config()
        logger.info(
            f"No active embedding config in DB, using env vars: "
            f"provider={config.provider}, model={config.model}"
        )
    return create_embeddings_model(config)
