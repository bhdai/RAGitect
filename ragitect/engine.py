from collections.abc import Sequence
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama.embeddings import OllamaEmbeddings
import logging
from collections.abc import Generator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from faiss import IndexFlatIP
from langchain_core.documents.base import Document

from ragitect.services import (
    document_processor,
    embedding,
    llm,
    query_service,
    vector_store,
)
from ragitect.services.config import LLMConfig

logger = logging.getLogger(__name__)


class ChatEngine:
    def __init__(self, config: LLMConfig | None = None):
        if config is None:
            from ragitect.services.config import load_config_from_env

            config = load_config_from_env()
        self.config: LLMConfig = config
        is_valid, error_msg = llm.valididate_llm_config(config)
        if not is_valid:
            raise ValueError(f"Invalid LLM configuration: {error_msg}")
        self.embedding_model: OllamaEmbeddings = embedding.create_embeddings_model()
        self.llm_model: BaseChatModel = llm.create_llm(config)
        self.dimension: int = embedding.get_embedding_dimension()
        logger.info(f"ChatEngine initialized with provider={config.provider}")

    def process_document(
        self,
        file_bytes: bytes,
        file_name: str,
    ) -> tuple[IndexFlatIP, list[Document]]:
        """take raw file bytes, processes, embeds, and index them

        Args:
            file_bytes: raw bytes of the file
            file_name: file name

        Returns:
            tuple of Faiss index and document store
        """
        print(f"Processing document: {file_name}")
        raw_text = file_bytes.decode("utf-8", errors="replace")
        chunks = document_processor.split_document(raw_text, chunk_size=500, overlap=50)
        document_store = document_processor.create_documents(chunks, file_name)

        vectors = embedding.embed_documents(
            self.embedding_model, [doc.page_content for doc in document_store]
        )
        faiss_index = vector_store.initialize_index(self.dimension)
        vector_store.add_vectors_to_index(faiss_index, vectors)

        print(f"File processed. Index created with {len(vectors)} vectors")
        return faiss_index, document_store

    def get_response_stream(
        self,
        query: str,
        faiss_index: IndexFlatIP,
        document_store: list[Document],
        chat_history: list[dict[str, str]] | None = None,
    ) -> Generator[str]:
        """Generate response stream for given query

        Args:
            query: query string
            faiss_index: faiss index
            document_store: document store
            chat_history: list of history messages

        Returns:
            Generator of response strings
        """
        print(f"Generating response for query {query[:20]}...")
        if chat_history is None:
            chat_history = []

        reformulated_query = query_service.reformulate_query_with_chat_history(
            self.llm_model, query, chat_history
        )

        query_vector = embedding.embed_text(self.embedding_model, reformulated_query)
        retrieved_docs = vector_store.search_index(
            faiss_index, query_vector, document_store, k=10
        )
        context = "\n\n".join([doc.page_content for doc, _ in retrieved_docs])

        system_instruction = """
        You are a helpful AI assistant with expertise in
        technical documentation.
        Your goal is to provide clear, detailed, and educational answers based on the retrieved context.

        **Instructions:**
        1. Answer the user's question using the information from the context below
        2. Provide detailed explanations with examples when relevant
        3. Structure your answer clearly (use sections if helpful)
        4. If the context contains code examples, include them in your response
        5. If the context doesn't contain enough information, acknowledge what you don't know
        6. Do not make up information outside the provided context
        """

        history_messages = _format_chat_history_for_prompt(chat_history)
        messages = _build_prompt_messages(
            system_instruction,
            context,
            query,
            history_messages,
        )

        return llm.generate_response_stream(self.llm_model, messages)


def _format_chat_history_for_prompt(
    chat_history: list[dict[str, str]],
) -> Sequence[BaseMessage]:
    """convert chat history to langchain message format

    Args:
        chat_history: list of message dict with 'role' and 'content' keys

    Returns:
        list[BaseMessage]: list of langchain message objects
    """
    messages: list[BaseMessage] = []
    for message in chat_history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        else:
            logger.warning(f"Unknown message role: {message['role']}")
    return messages


def _build_prompt_messages(
    system_instruction: str,
    context: str,
    query: str,
    chat_history: Sequence[BaseMessage],
) -> Sequence[BaseMessage]:
    """Build complete message list for llm including history

    Args:
        system_instruction: system level instruction
        context: retrieved context from vector store
        query: current user query
        chat_history: previous conversation history

    Returns:
        list[BaseMessage]: list of langchain message objects
    """
    messages: list[BaseMessage] = []
    messages.append(SystemMessage(content=system_instruction))
    # add conversation history
    messages.extend(chat_history)
    # build context messages
    context_text = f"""
    Retrieved Context:
            {context}
    Current Question:
            {query}
    """
    messages.append(HumanMessage(content=context_text))
    return messages
