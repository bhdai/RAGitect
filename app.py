from collections.abc import Generator
from langchain_ollama.chat_models import ChatOllama
from langchain_core.documents.base import Document
from faiss import IndexFlatIP
from langchain_ollama.embeddings import OllamaEmbeddings
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st

from src.document_processor import load_document, split_document, create_documents
from src.embedding import (
    create_embeddings_model,
    embed_documents,
    embed_text,
    get_embedding_dimension,
)
from src.vector_store import initialize_index, add_vectors_to_index, search_index
from src.llm import generate_response_stream, create_llm


@st.cache_resource
def get_llm():
    return create_llm(model_name="llama3.1:8b", temperature=0.2)


@st.cache_resource
def get_embedding_model():
    return create_embeddings_model()


def initialize_session_state():
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "document_store" not in st.session_state:
        st.session_state.document_store = []
    if "messages" not in st.session_state:
        st.session_state.messages = []  # for storing chat history
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None


def process_document(
    file_bytes: bytes,
    file_name: str,
    embeddings_model: OllamaEmbeddings,
) -> tuple[IndexFlatIP, list[Document]]:
    """take raw file bytes, processes, embeds, and index them

    Args:
        file_bytes: raw bytes of the file
        file_name: file name
        embeddings_model: embeddings model

    Returns:
        tuple of Faiss index and document store
    """
    print(f"Processing document: {file_name}")
    raw_text = file_bytes.decode("utf-8", errors="replace")
    chunks = split_document(raw_text, chunk_size=500, overlap=50)
    document_store = create_documents(chunks, file_name)

    vectors = embed_documents(
        embeddings_model, [doc.page_content for doc in document_store]
    )
    dim = get_embedding_dimension()
    faiss_index = initialize_index(dim)
    add_vectors_to_index(faiss_index, vectors)

    print(f"File processed. Index created with {len(vectors)} vectors")
    return faiss_index, document_store


def get_response_stream(
    query: str,
    faiss_index: IndexFlatIP,
    document_store: list[Document],
    llm_model: ChatOllama,
    embedding_model: OllamaEmbeddings,
) -> Generator[str]:
    print(f"Generating response for query {query[:20]}...")

    query_vector = embed_text(embedding_model, query)

    retrieved_docs = search_index(faiss_index, query_vector, document_store, k=10)

    context = "\n\n".join([doc.page_content for doc, _ in retrieved_docs])

    prompt = f"""
Answer the question using ONLY the information in the retrieved context below.
If the context does not contain enough information to answer the question, respond with: "I don't have that information in the provided document."
Do not use any outside knowledge.

Context:
            {context}
Question:
            {query}
Answer:
    """

    return generate_response_stream(llm_model, prompt)


def file_item(file_name: str) -> None:
    """clickable file item when click display the file

    Args:
        file_name: file name
    """
    if st.button(file_name, key=f"view_{file_name}"):
        st.session_state["file_to_view"] = file_name


def uploaded_file_items(uploaded_files: list[UploadedFile]) -> dict[str, UploadedFile]:
    """Display file items

    Args:
        uploaded_files: list of uploaded files

    Returns:
        dictionary with key is file name and value is file object
    """
    file_dict = {f.name: f for f in uploaded_files}
    for file_name in file_dict:
        file_item(file_name)
    return file_dict


def main():
    st.set_page_config(
        page_title="RAGitect",
        page_icon="🎈",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()
    llm = get_llm()
    embedding_model = get_embedding_model()

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=["txt"],
        )
        if (
            uploaded_file is not None
            and uploaded_file.name != st.session_state.processed_file_name
        ):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                file_bytes = uploaded_file.getvalue()
                index, doc_store = process_document(
                    file_bytes, uploaded_file.name, embedding_model
                )

                st.session_state.faiss_index = index
                st.session_state.document_store = doc_store
                st.session_state.processed_file_name = uploaded_file.name
                st.session_state.messages = []  # clear the chat when upload new docs for now

            st.success(f"File '{uploaded_file.name}' processed.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.faiss_index is None:
            st.warning("Please upload a document first")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_generator = get_response_stream(
                        query=prompt,
                        faiss_index=st.session_state.faiss_index,
                        document_store=st.session_state.document_store,
                        llm_model=llm,
                        embedding_model=embedding_model,
                    )
                    full_response = st.write_stream(response_generator)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )


if __name__ == "__main__":
    main()
