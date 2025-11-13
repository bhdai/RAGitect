from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def split_document(
    raw_text: str, chunk_size: int = 500, overlap: int = 50
) -> list[str]:
    """split raw text into chunks

    Args:
        raw_text: raw text string
        chunk_size: chunk size
        overlap: overlap size

    Returns:
        list of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    all_splits = text_splitter.split_text(raw_text)
    return all_splits
