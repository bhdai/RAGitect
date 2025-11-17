import logging
import streamlit as st

from ragitect.engine import ChatEngine
from ragitect.services.config import load_config_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


@st.cache_resource
def get_chat_engine() -> ChatEngine:
    """Load the ChatEngine instance"""
    config = load_config_from_env()
    return ChatEngine(config=config)


def initialize_session_state():
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "document_store" not in st.session_state:
        st.session_state.document_store = []
    if "messages" not in st.session_state:
        st.session_state.messages = []  # for storing chat history
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None


def main():
    st.set_page_config(
        page_title="RAGitect",
        page_icon="🎈",
        initial_sidebar_state="expanded",
    )

    engine = get_chat_engine()
    initialize_session_state()

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
                index, doc_store = engine.process_document(
                    file_bytes,
                    uploaded_file.name,
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
                    response_generator = engine.get_response_stream(
                        query=prompt,
                        faiss_index=st.session_state.faiss_index,
                        document_store=st.session_state.document_store,
                        chat_history=st.session_state.messages[:, -1],
                    )
                    full_response = st.write_stream(response_generator)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )


if __name__ == "__main__":
    main()
