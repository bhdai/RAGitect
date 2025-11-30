# Document processing
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
    valididate_llm_config,
)

# Vector store
from ragitect.services.vector_store import (
    add_vectors_to_index,
    initialize_index,
    search_index,
)

# Query service
from ragitect.services.query_service import (
    format_chat_history,
    reformulate_query_with_chat_history,
)

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
    LLMConfig,
    get_default_config,
    load_config_from_env,
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
    "generate_response",
    "generate_response_stream",
    "generate_response_with_prompt",
    "valididate_llm_config",
    # Vector store
    "add_vectors_to_index",
    "initialize_index",
    "search_index",
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
    "LLMConfig",
    "get_default_config",
    "load_config_from_env",
]
