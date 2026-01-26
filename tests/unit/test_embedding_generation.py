"""Unit tests for embedding generation in document processing.

Tests embedding integration with DocumentProcessingService:
- Chunk generation from processed text
- Embedding model initialization with different providers
- Batch embedding with mocked model
- Error handling (model unavailable, API timeout)

These tests follow TDD (Red-Green-Refactor) - they are written FIRST
and expected to FAIL until implementation is complete.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from ragitect.services.config import EmbeddingConfig
from ragitect.services.embedding import create_embeddings_model, embed_documents


class TestEmbeddingModelInitialization:
    """Tests for embedding model initialization with different providers."""

    def test_create_embeddings_model_default_ollama(self):
        """Test default configuration uses Ollama provider."""
        model = create_embeddings_model()

        # Should return OllamaEmbeddings instance
        assert model is not None
        assert hasattr(model, "aembed_documents")

    def test_create_embeddings_model_with_custom_config(self):
        """Test creating model with custom Ollama config."""
        config = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
            base_url="http://localhost:11434",
            dimension=768,
        )

        model = create_embeddings_model(config)
        assert model is not None

    def test_create_embeddings_model_openai_requires_api_key(self):
        """Test OpenAI provider requires API key."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key=None,  # No API key
        )

        with pytest.raises(ValueError, match="api_key is required"):
            create_embeddings_model(config)

    def test_create_embeddings_model_unsupported_provider(self):
        """Test unsupported provider raises ValueError."""
        config = EmbeddingConfig(
            provider="unsupported_provider",
            model="some-model",
        )

        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            create_embeddings_model(config)

    def test_create_embeddings_model_ollama_requires_base_url(self):
        """Test Ollama provider requires base_url."""
        config = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
            base_url=None,  # No base URL
        )

        with pytest.raises(ValueError, match="base_url is required"):
            create_embeddings_model(config)


class TestGetEmbeddingModelFromConfig:
    """Tests for the get_embedding_model_from_config helper function."""

    @pytest.fixture
    def mock_session(self):
        """Mock AsyncSession."""
        return MagicMock()

    @pytest.fixture
    def mock_openai_config_dto(self):
        """Mock OpenAI embedding configuration DTO."""
        dto = MagicMock()
        dto.provider_name = "openai"
        dto.model_name = "text-embedding-3-small"
        dto.base_url = None
        dto.api_key = "sk-test-key-12345"
        dto.dimension = 1536
        return dto

    @pytest.fixture
    def mock_ollama_config_dto(self):
        """Mock Ollama embedding configuration DTO."""
        dto = MagicMock()
        dto.provider_name = "ollama"
        dto.model_name = "qwen3-embedding:0.6b"
        dto.base_url = "http://localhost:11434"
        dto.api_key = None
        dto.dimension = 768
        return dto

    @pytest.mark.asyncio
    async def test_get_embedding_model_with_openai_config(
        self, mock_session, mock_openai_config_dto
    ):
        """Test helper returns OpenAI model when active config is OpenAI."""
        from ragitect.services.embedding import get_embedding_model_from_config

        with patch(
            "ragitect.services.llm_config_service.get_active_embedding_config",
            new=AsyncMock(return_value=mock_openai_config_dto),
        ):
            model = await get_embedding_model_from_config(mock_session)

        # Should return an embeddings model (OpenAI)
        assert model is not None
        assert hasattr(model, "aembed_documents")

    @pytest.mark.asyncio
    async def test_get_embedding_model_with_ollama_config(
        self, mock_session, mock_ollama_config_dto
    ):
        """Test helper returns Ollama model when active config is Ollama."""
        from ragitect.services.embedding import get_embedding_model_from_config

        with patch(
            "ragitect.services.llm_config_service.get_active_embedding_config",
            new=AsyncMock(return_value=mock_ollama_config_dto),
        ):
            model = await get_embedding_model_from_config(mock_session)

        # Should return an embeddings model (Ollama)
        assert model is not None
        assert hasattr(model, "aembed_documents")

    @pytest.mark.asyncio
    async def test_get_embedding_model_with_no_active_config(self, mock_session):
        """Test helper returns env-configured model when no active DB config."""
        from ragitect.services.embedding import get_embedding_model_from_config
        from ragitect.services.config import EmbeddingConfig

        # Mock no DB config and mock env config to Ollama defaults
        env_config = EmbeddingConfig(
            provider="ollama",
            model="qwen3-embedding:0.6b",
            base_url="http://localhost:11434",
            dimension=768,
        )

        with patch(
            "ragitect.services.llm_config_service.get_active_embedding_config",
            new=AsyncMock(return_value=None),
        ):
            with patch(
                "ragitect.services.config.load_embedding_config",
                return_value=env_config,
            ):
                model = await get_embedding_model_from_config(mock_session)

        # Should return default embeddings model (Ollama)
        assert model is not None
        assert hasattr(model, "aembed_documents")


class TestBatchEmbedding:
    """Tests for batch embedding generation."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for unit tests."""
        model = MagicMock()
        # Return list of 768-dim vectors
        model.aembed_documents = AsyncMock(return_value=[[0.1] * 768 for _ in range(5)])
        return model

    @pytest.mark.asyncio
    async def test_embed_documents_returns_vectors(self, mock_embedding_model):
        """Test batch embedding returns list of vectors."""
        texts = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5"]

        embeddings = await embed_documents(mock_embedding_model, texts)

        assert len(embeddings) == 5
        assert all(len(emb) == 768 for emb in embeddings)
        mock_embedding_model.aembed_documents.assert_called_once_with(texts)

    @pytest.mark.asyncio
    async def test_embed_documents_empty_list(self, mock_embedding_model):
        """Test embedding empty list returns empty list."""
        mock_embedding_model.aembed_documents = AsyncMock(return_value=[])

        embeddings = await embed_documents(mock_embedding_model, [])

        assert embeddings == []

    @pytest.mark.asyncio
    async def test_embed_documents_single_text(self, mock_embedding_model):
        """Test embedding single text returns single vector."""
        mock_embedding_model.aembed_documents = AsyncMock(return_value=[[0.5] * 768])

        embeddings = await embed_documents(mock_embedding_model, ["Single chunk"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768


class TestChunkGeneration:
    """Tests for text chunking before embedding."""

    def test_split_document_generates_chunks(self):
        """Test split_document returns list of chunks."""
        from ragitect.services.document_processor import split_document

        # Long text that should be split
        text = "This is a test sentence. " * 100  # ~2500 chars

        chunks = split_document(text, chunk_size=500, overlap=50)

        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_document_short_text_single_chunk(self):
        """Test short text returns single chunk."""
        from ragitect.services.document_processor import split_document

        text = "Short text."

        chunks = split_document(text, chunk_size=1000, overlap=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_document_empty_text(self):
        """Test empty text returns empty list."""
        from ragitect.services.document_processor import split_document

        chunks = split_document("", chunk_size=1000, overlap=100)

        assert chunks == []

    def test_split_markdown_document_preserves_structure(self):
        """Test markdown splitting preserves header structure."""
        from ragitect.services.document_processor import split_document

        markdown = """# Header 1
        
Some content under header 1.

## Header 2

More content under header 2.

### Header 3

Content under header 3.
"""
        chunks = split_document(markdown, chunk_size=500, overlap=50)

        # Should split by markdown structure
        assert len(chunks) >= 1


class TestEmbeddingErrorHandling:
    """Tests for error handling during embedding generation."""

    @pytest.fixture
    def mock_failing_model(self):
        """Mock model that raises exceptions."""
        model = MagicMock()
        model.aembed_documents = AsyncMock(side_effect=Exception("Model unavailable"))
        return model

    @pytest.mark.asyncio
    async def test_embed_documents_model_unavailable(self, mock_failing_model):
        """Test embedding raises exception when model unavailable."""
        texts = ["Test chunk"]

        with pytest.raises(Exception, match="Model unavailable"):
            await embed_documents(mock_failing_model, texts)

    @pytest.mark.asyncio
    async def test_embed_documents_timeout(self):
        """Test embedding handles timeout errors."""
        import asyncio

        model = MagicMock()
        model.aembed_documents = AsyncMock(
            side_effect=asyncio.TimeoutError("API timeout")
        )

        with pytest.raises(asyncio.TimeoutError):
            await embed_documents(model, ["Test chunk"])


class TestDocumentProcessingServiceEmbedding:
    """Tests for embedding integration in DocumentProcessingService.

    These tests verify that DocumentProcessingService correctly:
    1. Transitions status to "embedding" after text extraction
    2. Splits text into chunks
    3. Generates embeddings for chunks
    4. Stores chunks with embeddings via add_chunks()
    5. Transitions status to "ready" on success
    """

    @pytest.fixture
    def mock_session(self):
        """Mock AsyncSession."""
        session = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def mock_document_repo(self):
        """Mock DocumentRepository."""
        repo = MagicMock()
        repo.get_by_id_or_raise = AsyncMock()
        repo.update_status = AsyncMock()
        repo.get_file_bytes = AsyncMock()
        repo.update_processed_content = AsyncMock()
        repo.clear_file_bytes = AsyncMock()
        repo.add_chunks = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def sample_document(self):
        """Sample document with uploaded status."""
        import base64
        from ragitect.services.database.models import Document

        doc_id = uuid4()
        file_bytes = b"Sample content"
        file_bytes_b64 = base64.b64encode(file_bytes).decode("utf-8")

        return Document(
            id=doc_id,
            workspace_id=uuid4(),
            file_name="test.txt",
            file_type=".txt",
            content_hash="abc123",
            unique_identifier_hash="unique123",
            processed_content=None,
            metadata_={
                "status": "uploaded",
                "original_size": len(file_bytes),
                "file_bytes_b64": file_bytes_b64,
            },
        )

    @pytest.mark.asyncio
    async def test_process_document_generates_embeddings(
        self,
        mock_session,
        mock_document_repo,
        sample_document,
    ):
        """Test that process_document generates and stores embeddings.

        Expected flow:
        1. Extract text (existing)
        2. Update status to "embedding" (NEW)
        3. Split text into chunks (NEW)
        4. Generate embeddings for chunks (NEW)
        5. Store chunks via add_chunks() (NEW)
        6. Update status to "ready" (existing)
        """
        from ragitect.services.document_processing_service import (
            DocumentProcessingService,
        )

        # Arrange
        mock_document_repo.get_by_id_or_raise.return_value = sample_document
        mock_document_repo.get_file_bytes.return_value = (
            b"Sample content for embedding test."
        )

        service = DocumentProcessingService(mock_session)

        # Patch the repo
        with patch.object(service, "repo", mock_document_repo):
            # Mock text extraction
            with patch(
                "ragitect.services.document_processing_service.process_file_bytes"
            ) as mock_process:
                mock_process.return_value = (
                    "Extracted text content for chunking and embedding. " * 10,
                    {"file_type": ".txt", "file_name": "test.txt"},
                )

                # Mock embedding functions
                mock_model = MagicMock()
                mock_model.aembed_documents = AsyncMock(
                    return_value=[[0.1] * 768 for _ in range(3)]
                )

                with patch(
                    "ragitect.services.document_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=mock_model),
                ):
                    with patch(
                        "ragitect.services.document_processing_service.embed_documents",
                        new=mock_model.aembed_documents,
                    ):
                        with patch(
                            "ragitect.services.document_processing_service.split_document",
                            return_value=["Chunk 1", "Chunk 2", "Chunk 3"],
                        ):
                            # Act
                            await service.process_document(sample_document.id)

        # Assert - verify status transitions include "embedding"
        status_calls = [
            call[0][1] for call in mock_document_repo.update_status.call_args_list
        ]
        assert "processing" in status_calls
        assert "embedding" in status_calls
        assert "ready" in status_calls

        # Assert - verify chunks were stored
        mock_document_repo.add_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_status_transition_to_embedding(
        self,
        mock_session,
        mock_document_repo,
        sample_document,
    ):
        """Test status correctly transitions: processing -> embedding -> ready."""
        from ragitect.services.document_processing_service import (
            DocumentProcessingService,
        )

        mock_document_repo.get_by_id_or_raise.return_value = sample_document
        mock_document_repo.get_file_bytes.return_value = b"Content"

        service = DocumentProcessingService(mock_session)

        with patch.object(service, "repo", mock_document_repo):
            with patch(
                "ragitect.services.document_processing_service.process_file_bytes",
                return_value=("Text", {"file_type": ".txt"}),
            ):
                mock_model = MagicMock()
                mock_model.aembed_documents = AsyncMock(return_value=[[0.1] * 768])
                with patch(
                    "ragitect.services.document_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=mock_model),
                ):
                    with patch(
                        "ragitect.services.document_processing_service.embed_documents",
                        new=AsyncMock(return_value=[[0.1] * 768]),
                    ):
                        with patch(
                            "ragitect.services.document_processing_service.split_document",
                            return_value=["Chunk"],
                        ):
                            await service.process_document(sample_document.id)

        # Verify status transition order
        status_calls = [
            call[0][1] for call in mock_document_repo.update_status.call_args_list
        ]

        # Find indices of each status
        processing_idx = status_calls.index("processing")
        embedding_idx = status_calls.index("embedding")
        ready_idx = status_calls.index("ready")

        # Assert order: processing -> embedding -> ready
        assert processing_idx < embedding_idx < ready_idx

    @pytest.mark.asyncio
    async def test_process_document_embedding_failure_sets_error_status(
        self,
        mock_session,
        mock_document_repo,
        sample_document,
    ):
        """Test that embedding failure updates status to 'error'."""
        from ragitect.services.document_processing_service import (
            DocumentProcessingService,
        )

        mock_document_repo.get_by_id_or_raise.return_value = sample_document
        mock_document_repo.get_file_bytes.return_value = b"Content"

        service = DocumentProcessingService(mock_session)

        with patch.object(service, "repo", mock_document_repo):
            with patch(
                "ragitect.services.document_processing_service.process_file_bytes",
                return_value=("Text", {"file_type": ".txt"}),
            ):
                with patch(
                    "ragitect.services.document_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(side_effect=Exception("Embedding model unavailable")),
                ):
                    with pytest.raises(Exception, match="Embedding model unavailable"):
                        await service.process_document(sample_document.id)

        # Verify error status was set
        mock_document_repo.update_status.assert_any_call(sample_document.id, "error")

    @pytest.mark.asyncio
    async def test_process_document_stores_chunks_with_embeddings(
        self,
        mock_session,
        mock_document_repo,
        sample_document,
    ):
        """Test chunks are stored with correct format: (content, embedding, metadata)."""
        from ragitect.services.document_processing_service import (
            DocumentProcessingService,
        )

        mock_document_repo.get_by_id_or_raise.return_value = sample_document
        mock_document_repo.get_file_bytes.return_value = b"Content"

        service = DocumentProcessingService(mock_session)

        chunks = ["Chunk 1", "Chunk 2"]
        embeddings = [[0.1] * 768, [0.2] * 768]

        with patch.object(service, "repo", mock_document_repo):
            with patch(
                "ragitect.services.document_processing_service.process_file_bytes",
                return_value=("Text", {"file_type": ".txt"}),
            ):
                mock_model = MagicMock()
                mock_model.aembed_documents = AsyncMock(return_value=embeddings)
                with patch(
                    "ragitect.services.document_processing_service.get_embedding_model_from_config",
                    new=AsyncMock(return_value=mock_model),
                ):
                    with patch(
                        "ragitect.services.document_processing_service.embed_documents",
                        new=AsyncMock(return_value=embeddings),
                    ):
                        with patch(
                            "ragitect.services.document_processing_service.split_document",
                            return_value=chunks,
                        ):
                            await service.process_document(sample_document.id)

        # Verify add_chunks called with correct format
        add_chunks_call = mock_document_repo.add_chunks.call_args
        assert add_chunks_call is not None

        doc_id, chunk_data = add_chunks_call[0]
        assert doc_id == sample_document.id
        assert len(chunk_data) == 2

        # Each chunk should be (content, embedding, metadata)
        for i, (content, embedding, metadata) in enumerate(chunk_data):
            assert content == chunks[i]
            assert embedding == embeddings[i]
            assert "chunk_index" in metadata
            assert metadata["chunk_index"] == i

    @pytest.mark.asyncio
    async def test_process_document_empty_text_skips_embedding(
        self,
        mock_session,
        mock_document_repo,
        sample_document,
    ):
        """Test that empty extracted text skips embedding step gracefully."""
        from ragitect.services.document_processing_service import (
            DocumentProcessingService,
        )

        mock_document_repo.get_by_id_or_raise.return_value = sample_document
        mock_document_repo.get_file_bytes.return_value = b"Empty"

        service = DocumentProcessingService(mock_session)

        with patch.object(service, "repo", mock_document_repo):
            with patch(
                "ragitect.services.document_processing_service.process_file_bytes",
                return_value=("", {"file_type": ".txt"}),  # Empty text
            ):
                with patch(
                    "ragitect.services.document_processing_service.split_document",
                    return_value=[],  # No chunks
                ):
                    await service.process_document(sample_document.id)

        # Should still complete successfully with ready status
        mock_document_repo.update_status.assert_any_call(sample_document.id, "ready")
        # add_chunks should not be called with empty chunks
        # (or called with empty list - implementation dependent)
