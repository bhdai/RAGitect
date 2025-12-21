"""Document processing with token-based markdown chunking.

Implements:
- Hybrid tokenization (tiktoken for BPE models, transformers for BERT/WordPiece)
- Orphan header merging (prevents small header-only chunks)
- Token-based sizing (consistent across embedding models)
"""

import logging
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer

from ragitect.services.config import EmbeddingConfig, load_embedding_config
from ragitect.services.processor.factory import ProcessorFactory

logger = logging.getLogger(__name__)

# Global tokenizer cache
_TOKENIZER = None
_TOKENIZER_TYPE: str | None = None

# Robust mapping from Ollama names to Hugging Face Tokenizers
# Ollama is lowercase/dash-separated; HF is Case-Sensitive/Org-Prefixed
OLLAMA_TO_HF_TOKENIZER: dict[str, str] = {
    # Qwen3 Embedding models
    "qwen3-embedding:0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding:4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding:8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3-embedding": "Qwen/Qwen3-Embedding-0.6B",
    # BGE models
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
    # Nomic models
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text:latest": "nomic-ai/nomic-embed-text-v1.5",
    # MiniLM models
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm:latest": "sentence-transformers/all-MiniLM-L6-v2",
    # Other popular models
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-l-v2.0",
}


def get_tokenizer_for_embedding_model(config: EmbeddingConfig | None = None):
    """Get the correct tokenizer for the active embedding model.

    - OpenAI/Qwen models: Use tiktoken (BPE)
    - HuggingFace/Ollama BERT-based: Use transformers AutoTokenizer (WordPiece)

    Args:
        config: Optional embedding config. If None, loads from environment.

    Returns:
        Tuple of (tokenizer, tokenizer_type: "tiktoken" | "transformers")
    """
    if config is None:
        config = load_embedding_config()

    provider = config.provider.lower()
    model_name = config.model.strip()

    # 1. OpenAI provider -> tiktoken (BPE)
    if provider == "openai":
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Using tiktoken (cl100k_base) [BPE Mode] for OpenAI")
            return tokenizer, "tiktoken"
        except Exception:
            pass

    # 2. Default/Fallback -> transformers AutoTokenizer (Handles BERT/WordPiece)
    # Try to map Ollama name to Hugging Face repository
    hf_model = OLLAMA_TO_HF_TOKENIZER.get(model_name)

    # If no strict mapping, try smart fallback for common substrings
    if not hf_model:
        if "nomic" in model_name:
            hf_model = "nomic-ai/nomic-embed-text-v1.5"
        elif "minilm" in model_name:
            hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        elif "bge-m3" in model_name:
            hf_model = "BAAI/bge-m3"
        elif "qwen" in model_name.lower():
            hf_model = "Qwen/Qwen3-Embedding-0.6B"
        else:
            # Last resort: use the name as-is (might fail if case-mismatch)
            hf_model = model_name

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        logger.info(f"Using AutoTokenizer for {hf_model} [WordPiece Mode]")
        return tokenizer, "transformers"
    except Exception as e:
        logger.warning(
            f"Failed to load tokenizer for {hf_model} (mapped from {model_name}): {e}. "
            f"Falling back to tiktoken (May be inaccurate for this model!)"
        )
        # Safe Fallback: tiktoken
        return tiktoken.get_encoding("cl100k_base"), "tiktoken"


def count_tokens(text: str) -> int:
    """Count tokens using the active model's tokenizer strategy.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    global _TOKENIZER, _TOKENIZER_TYPE

    if _TOKENIZER is None:
        _TOKENIZER, _TOKENIZER_TYPE = get_tokenizer_for_embedding_model()

    if _TOKENIZER_TYPE == "tiktoken":
        return len(_TOKENIZER.encode(text))
    else:  # transformers
        return len(_TOKENIZER.encode(text, add_special_tokens=False))  # type: ignore[call-arg]


def load_document(file_path: str) -> str:
    """Load the document

    Args:
        file_path: file path to the document

    Returns:
        raw text string
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        # use `replace` to avoid issues with invalid characters
        text = f.read()
    return text


def create_documents(chunks: list[str], source: str) -> list[Document]:
    """add metadata to each chunk and create Document objects

    Args:
        chunks: chunks of text
        source: source of the document

    Returns:
        chunks with metadata
    """
    documents: list[Document] = []
    for i, chunk in enumerate(chunks):
        documents.append(
            Document(page_content=chunk, metadata={"source": source, "chunk_index": i})
        )

    return documents


def split_markdown_document(
    raw_text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    min_chunk_size: int = 64,
) -> list[str]:
    """Split markdown with Hybrid Tokenization and Orphan Header Merging.

    Algorithm (2-step, single-pass):
    1. Split by headers (structural split)
    2. Process each chunk: size check → split if needed → forward-merge orphans inline

    Args:
        raw_text: Raw markdown text
        chunk_size: Target chunk size in tokens (default: 512)
        overlap: Overlap between chunks in tokens (default: 50)
        min_chunk_size: Minimum chunk size in tokens to prevent orphan headers (default: 64)

    Returns:
        List of text chunks with preserved structure
    """
    # Early return for empty text
    if not raw_text or not raw_text.strip():
        return []

    # Step 1: Structural Split by Headers
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Keep headers in chunks for context
    )

    try:
        # MarkdownHeaderTextSplitter returns list[Document]
        md_splits = markdown_splitter.split_text(raw_text)
        structural_chunks = [doc.page_content for doc in md_splits]

        logger.info(
            "Markdown chunking: %d structural chunks from header split",
            len(structural_chunks),
        )

        # Recursive splitter for oversized chunks
        # Separator order: paragraph > line > sentence > clause > word > char
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=count_tokens,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )

        # Step 2: Single-pass processing with inline orphan handling (forward merge)
        # Buffer accumulates orphan chunks to merge with the next non-orphan chunk
        final_chunks: list[str] = []
        orphan_buffer = ""

        for structural_chunk in structural_chunks:
            tokens = count_tokens(structural_chunk)

            # Case 1: Chunk is too small (orphan) - buffer for forward merge
            if tokens < min_chunk_size:
                orphan_buffer += (
                    "\n\n" + structural_chunk if orphan_buffer else structural_chunk
                )
                continue

            # Case 2: Chunk is good size (within bounds)
            if tokens <= chunk_size:
                # Prepend any buffered orphans
                if orphan_buffer:
                    final_chunks.append(orphan_buffer + "\n\n" + structural_chunk)
                    orphan_buffer = ""
                else:
                    final_chunks.append(structural_chunk)
                continue

            # Case 3: Chunk is too large - needs recursive splitting
            # First, prepend any buffered orphans to the chunk before splitting
            chunk_to_split = (
                orphan_buffer + "\n\n" + structural_chunk
                if orphan_buffer
                else structural_chunk
            )
            orphan_buffer = ""

            sub_chunks = text_splitter.split_text(chunk_to_split)

            # Process sub-chunks with inline orphan handling
            for sub_chunk in sub_chunks:
                sub_tokens = count_tokens(sub_chunk)

                if sub_tokens < min_chunk_size:
                    # Sub-chunk is orphan, buffer for forward merge
                    orphan_buffer += "\n\n" + sub_chunk if orphan_buffer else sub_chunk
                else:
                    # Prepend any buffered orphans
                    if orphan_buffer:
                        final_chunks.append(orphan_buffer + "\n\n" + sub_chunk)
                        orphan_buffer = ""
                    else:
                        final_chunks.append(sub_chunk)

        # Flush remaining orphan buffer
        if orphan_buffer:
            if final_chunks:
                # Merge with last chunk (backward merge as fallback for trailing orphans)
                final_chunks[-1] = final_chunks[-1] + "\n\n" + orphan_buffer
            else:
                # Edge case: entire document is orphan-sized
                final_chunks.append(orphan_buffer)

        # Log final statistics
        if final_chunks:
            token_counts = [count_tokens(c) for c in final_chunks]
            logger.info(
                "Final chunking: %d chunks, tokens: min=%d, max=%d, avg=%.1f",
                len(final_chunks),
                min(token_counts),
                max(token_counts),
                sum(token_counts) / len(token_counts),
            )

        return final_chunks

    except Exception as e:
        # Fallback uses same hybrid tokenizer logic
        logger.warning(f"Markdown splitting failed, using fallback: {e}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=count_tokens,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )
        return text_splitter.split_text(raw_text)


def split_document(
    raw_text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """Split text into chunks using token-based Markdown-aware splitting.

    All documents are processed to Markdown format before chunking:
    - DoclingProcessor converts PDF/DOCX/PPTX/etc. to Markdown
    - SimpleProcessor keeps TXT/MD files as-is

    Therefore, we always use Markdown-aware splitting to preserve document
    structure (headers, sections) regardless of the original file type.

    Args:
        raw_text: raw text string (in Markdown format)
        chunk_size: chunk size in tokens (default: 512)
        overlap: overlap size in tokens (default: 50)

    Returns:
        list of text chunks with preserved Markdown structure
    """
    # Always use Markdown-aware splitting since all documents are in Markdown format
    return split_markdown_document(raw_text, chunk_size, overlap)


def process_file_bytes(file_bytes: bytes, file_name: str) -> tuple[str, dict[str, str]]:
    """Process file bytes and extract text with metadata

    Args:
        file_bytes: file bytes
        file_name: file name

    Returns:
        tuple of (extracted_text, metadata_dict)
        metadata includes file_type for downstream chunking
    """
    factory = ProcessorFactory()
    processor = factory.get_processor(file_name)
    text = processor.process(file_bytes, file_name)

    # Build metadata for downstream processing
    metadata = {"file_type": Path(file_name).suffix.lower(), "file_name": file_name}

    return text, metadata
