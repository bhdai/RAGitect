import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://admin:admin@localhost:5432/ragitect_db"
)

DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
# pool_timeout: seconds to wait before giving up on getting a connection
DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
# pool_recycle: recycle connections after N seconds (prevents stale connections)
DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour
DB_ECHO: bool = os.getenv("DB_ECHO", "false").lower() == "true"

# vector search settings
DEFAULT_SIMILARITY_THRESHOLD: float = float(
    os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.3")
)
DEFAULT_RETRIEVAL_K: int = int(os.getenv("DEFAULT_RETRIEVAL_K", "10"))

# Multi-stage retrieval pipeline configuration (Story 3.1.2)
RETRIEVAL_INITIAL_K: int = int(os.getenv("RETRIEVAL_INITIAL_K", "50"))
RETRIEVAL_USE_RERANKER: bool = (
    os.getenv("RETRIEVAL_USE_RERANKER", "True").lower() == "true"
)
RETRIEVAL_RERANKER_TOP_K: int = int(os.getenv("RETRIEVAL_RERANKER_TOP_K", "30"))
RETRIEVAL_MMR_K: int = int(os.getenv("RETRIEVAL_MMR_K", "20"))
RETRIEVAL_USE_MMR: bool = os.getenv("RETRIEVAL_USE_MMR", "True").lower() == "true"
RETRIEVAL_USE_ADAPTIVE_K: bool = (
    os.getenv("RETRIEVAL_USE_ADAPTIVE_K", "True").lower() == "true"
)
RETRIEVAL_MMR_LAMBDA: float = float(os.getenv("RETRIEVAL_MMR_LAMBDA", "0.7"))
RETRIEVAL_ADAPTIVE_K_MIN: int = int(os.getenv("RETRIEVAL_ADAPTIVE_K_MIN", "4"))
RETRIEVAL_ADAPTIVE_K_MAX: int = int(os.getenv("RETRIEVAL_ADAPTIVE_K_MAX", "16"))
RETRIEVAL_ADAPTIVE_K_GAP_THRESHOLD: float = float(
    os.getenv("RETRIEVAL_ADAPTIVE_K_GAP_THRESHOLD", "0.15")
)
RETRIEVAL_TOKEN_BUDGET: int = int(os.getenv("RETRIEVAL_TOKEN_BUDGET", "4000"))

# Encryption key for API key storage (required for cloud LLM providers)
ENCRYPTION_KEY: str | None = os.getenv("ENCRYPTION_KEY")


@dataclass
class LLMConfig:
    """Configuration for LLM provider settings

    Attributes:
        provider: LLM provider name('ollama', 'openai', 'anthropic', 'gemini', etc.)
        model: Model identifier (e.g., 'llama3.1:8b', 'gpt-4', etc.)
        temperature: Sampling temperature (0.0-1.0)
        base_url: Base URL for API (used for Ollama/local deployments)
        api_key: API key for cloud provider (optional)
        max_tokens: Maximum tokens for response generation (optional)
        timeout: Request timeout in seconds (optional)
    """

    provider: str = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    base_url: str | None = None
    api_key: str | None = None
    custom_provider: str | None = None
    max_tokens: int | None = None
    timeout: int = 60


@dataclass
class DocumentConfig:
    """Document processing configuration with token-based chunking.

    Story 3.3.A: Backend Citation Metadata & Markdown Chunking Improvements

    Token-based chunking provides consistent sizing across embedding models
    and prevents semantic fragmentation from orphan headers.

    Note: To load from environment variables, use load_document_config()
    """

    enable_docling: bool = True
    enable_unstructure: bool = False
    chunk_size: int = 512  # Tokens (was characters, now token-based)
    chunk_overlap: int = 50  # Tokens (10% overlap)
    min_chunk_size: int = 64  # Tokens - prevents micro-chunks/orphan headers


def load_document_config() -> DocumentConfig:
    """Load document configuration from environment variables

    Returns:
        DocumentConfig with values from env vars or defaults
    """
    return DocumentConfig(
        enable_docling=os.getenv("ENABLE_DOCLING", "true").lower() == "true",
        enable_unstructure=os.getenv("ENABLE_UNSTRUCTURE", "false").lower() == "true",
        chunk_size=int(os.getenv("CHUNK_SIZE_TOKENS", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP_TOKENS", "50")),
        min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE_TOKENS", "64")),
    )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model settings

    Story 3.3.A: Added batch_size for embedding batch processing

    Attributes:
        provider: Embedding provider ('ollama', 'openai', etc.)
        model: Model identifier (e.g., 'qwen3-embedding:0.6b', 'text-embedding-3-small')
        base_url: Base URL for API (used for Ollama/local deployments)
        api_key: API key for cloud provider (optional)
        dimension: Embedding vector dimension
        batch_size: Max embeddings per API call (default: 32, prevents API limits)
    """

    provider: str = "ollama"
    model: str = "qwen3-embedding:0.6b"  # Changed from nomic-embed-text
    base_url: str | None = "http://localhost:11434"
    api_key: str | None = None
    dimension: int = 768
    batch_size: int = 32


def load_embedding_config() -> EmbeddingConfig:
    """Load embedding configuration from environment variables

    Returns:
        EmbeddingConfig with values from env vars or defaults
    """
    return EmbeddingConfig(
        provider=os.getenv("EMBEDDING_PROVIDER", "ollama"),
        model=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
    )


def load_config_from_env() -> LLMConfig:
    """Load LLM configuration from env variables

    Returns:
        LLMConfig: configuration object
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    # set a default base_url only if the provider is ollama and no url is given
    default_base_url = None
    if provider == "ollama":
        default_base_url = "http://localhost:11434"

    base_url = os.getenv("LLM_BASE_URL", default_base_url)

    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
    except ValueError:
        logger.warning("Invalid LLM_TEMPERATURE env var. Using default 0.7")
        temperature = 0.7

    max_tokens: int | None = None
    max_tokens_str = os.getenv("LLM_MAX_TOKENS")
    if max_tokens_str:
        try:
            max_tokens = int(max_tokens_str)
        except ValueError:
            logger.warning("Invalid LLM_MAX_TOKENS env var. Using default None")

    try:
        timeout = int(os.getenv("LLM_TIMEOUT", 60))
    except ValueError:
        logger.warning("Invalid LLM_TIMEOUT env var. Using default 60")
        timeout = 60

    config = LLMConfig(
        provider=provider,
        model=os.getenv("LLM_MODEL", "llama3.1:8b"),
        temperature=temperature,
        base_url=base_url,
        api_key=os.getenv("LLM_API_KEY"),
        custom_provider=os.getenv("LLM_CUSTOM_PROVIDER"),
        max_tokens=max_tokens,
        timeout=timeout,
    )

    log_config_dict = {
        "provider": config.provider,
        "model": config.model,
        "temperature": config.temperature,
        "base_url": config.base_url,
        "api_key": "********" if config.api_key else None,
        "custom_provider": config.custom_provider,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
    }

    logger.info(f"Loaded LLM Config: {log_config_dict}")

    return config


def get_default_config() -> LLMConfig:
    """Get default configuration for local ollama

    Returns:
        LLMConfig: default configuration object
    """
    return LLMConfig(
        provider="ollama",
        model="llama3.1:8b",
        temperature=0.7,
        base_url="http://localhost:11434",
        api_key=None,
        max_tokens=None,
        timeout=60,
    )
