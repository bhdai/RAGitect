from collections.abc import Generator

from faiss import IndexFlatIP
from langchain_core.documents.base import Document

from ragitect.services import (
    document_processor,
    embedding,
    llm,
    vector_store,
)


class ChatEngine:
    def __init__(self):
        print("Initializing ChatEngine...")
        # TODO: in app, these would be configurable
        self.embedding_model = embedding.create_embeddings_model()
        self.llm_model = llm.create_llm(model_name="llama3.1:8b", temperature=0.2)
        self.dimension = embedding.get_embedding_dimension()
        print("ChatEngine initialized")

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
    ) -> Generator[str]:
        """Generate response stream for given query

        Args:
            query: query string
            faiss_index: faiss index
            document_store: document store

        Returns:
            Generator of response strings
        """
        print(f"Generating response for query {query[:20]}...")

        query_vector = embedding.embed_text(self.embedding_model, query)
        retrieved_docs = vector_store.search_index(
            faiss_index, query_vector, document_store, k=10
        )
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

        return llm.generate_response_stream(self.llm_model, prompt)
