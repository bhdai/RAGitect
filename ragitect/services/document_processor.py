import logging
from pathlib import Path

from ragitect.services.processor.factory import ProcessorFactory
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

logger = logging.getLogger(__name__)


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
    raw_text: str, chunk_size: int = 1000, overlap: int = 150
) -> list[str]:
    """Split markdown document preserving structure

    First splits by headers to preserve hierarchy, then enforces size limits.

    Args:
        raw_text: Raw markdown text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks with preserved structure
    """
    # Define headers to split on (h1, h2, h3)
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    # First pass: split by markdown structure
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Keep headers in chunks for context
    )

    try:
        # MarkdownHeaderTextSplitter returns list[Document]
        md_splits = markdown_splitter.split_text(raw_text)

        # Extract text from Documents
        # Documents may have metadata with header info - we'll preserve by including in text
        structural_chunks = []
        for doc in md_splits:
            # Reconstruct chunk with header context if available
            chunk_text = doc.page_content
            structural_chunks.append(chunk_text)

        # Second pass: enforce size limits with recursive splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )

        final_chunks = []
        for chunk in structural_chunks:
            # Split large structural chunks while preserving overlap
            sub_chunks = text_splitter.split_text(chunk)
            final_chunks.extend(sub_chunks)

        return final_chunks

    except Exception as e:
        # Fallback to simple recursive splitting if markdown parsing fails
        logger.warning(f"Markdown splitting failed, falling back to recursive: {e}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        return text_splitter.split_text(raw_text)


def split_document(
    raw_text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    file_type: str | None = None,
) -> list[str]:
    """split raw text into chunks with optional markdown awareness

    Args:
        raw_text: raw text string
        chunk_size: chunk size
        overlap: overlap size
        file_type: File extension (e.g., '.md', '.txt') for format-aware splitting

    Returns:
        list of text chunks
    """
    # Use markdown-aware splitting for markdown files
    if file_type and file_type.lower() in [".md", ".markdown", ".txt"]:
        return split_markdown_document(raw_text, chunk_size, overlap)

    # Default: recursive character splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    all_splits = text_splitter.split_text(raw_text)
    return all_splits


def process_file_bytes(file_bytes: bytes, file_name: str) -> tuple[str, dict]:
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
