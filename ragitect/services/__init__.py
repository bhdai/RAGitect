# Document processing
# Configuration
from ragitect.services.config import (
    DATABASE_URL,
    DB_ECHO,
    DB_MAX_OVERFLOW,
    DB_POOL_RECYCLE,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DocumentConfig,
    EmbeddingConfig,
    LLMConfig,
    get_default_config,
    load_config_from_env,
    load_document_config,
)
from ragitect.services.document_processor import (
    create_documents,
    load_document,
    process_file_bytes,
    split_document,
)

# Embeddings
from ragitect.services.embedding import (
    create_embeddings_model,
    embed_documents,
    embed_text,
    get_embedding_dimension,
)

# LLM
from ragitect.services.llm import (
    LLMProvider,
    create_llm,
    generate_response,
    generate_response_stream,
    generate_response_with_prompt,
    validate_llm_config,
)
from ragitect.services.llm_factory import (
    create_llm_from_db,
    get_active_llm_config,
)

# Query service
from ragitect.services.query_service import (
    format_chat_history,
    reformulate_query_with_chat_history,
)

__all__ = [
    # Document processing
    "create_documents",
    "load_document",
    "process_file_bytes",
    "split_document",
    # Embeddings
    "create_embeddings_model",
    "embed_documents",
    "embed_text",
    "get_embedding_dimension",
    # LLM
    "LLMProvider",
    "create_llm",
    "create_llm_from_db",
    "generate_response",
    "generate_response_stream",
    "generate_response_with_prompt",
    "get_active_llm_config",
    "validate_llm_config",
    # Query service
    "format_chat_history",
    "reformulate_query_with_chat_history",
    # Configuration
    "DATABASE_URL",
    "DB_ECHO",
    "DB_MAX_OVERFLOW",
    "DB_POOL_RECYCLE",
    "DB_POOL_SIZE",
    "DB_POOL_TIMEOUT",
    "DEFAULT_RETRIEVAL_K",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DocumentConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "get_default_config",
    "load_config_from_env",
    "load_document_config",
]
