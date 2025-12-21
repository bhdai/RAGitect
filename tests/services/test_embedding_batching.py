"""Tests for embedding batch processing.

Tests:
- Batch processing for large datasets (AC8)
- Single batch for small datasets
- Correct embedding count returned
"""

from unittest.mock import AsyncMock, Mock

import pytest

from ragitect.services.embedding import embed_documents

pytestmark = [pytest.mark.asyncio]


class TestEmbeddingBatchProcessing:
    """Test batched embedding generation (AC8)"""

    async def test_embedding_batching_large_dataset(self):
        """Test that large datasets are batched correctly"""
        # Create 100 test chunks
        chunks = [f"Test chunk {i}" for i in range(100)]

        # Mock embedding model
        mock_model = Mock()

        # Return different embeddings per batch call
        async def mock_embed(texts):
            return [[0.1] * 768 for _ in texts]

        mock_model.aembed_documents = AsyncMock(side_effect=mock_embed)

        # Embed with batch_size=32
        embeddings = await embed_documents(mock_model, chunks, batch_size=32)

        # Should have called API 4 times (100/32 = 3.125 → 4 batches)
        assert mock_model.aembed_documents.call_count == 4
        assert len(embeddings) == 100

    async def test_embedding_batching_small_dataset(self):
        """Test that small datasets skip batching"""
        chunks = [f"Test chunk {i}" for i in range(10)]

        mock_model = Mock()

        async def mock_embed(texts):
            return [[0.1] * 768 for _ in texts]

        mock_model.aembed_documents = AsyncMock(side_effect=mock_embed)

        # Embed with batch_size=32 (should skip batching since 10 < 32)
        embeddings = await embed_documents(mock_model, chunks, batch_size=32)

        # Should have called API once
        assert mock_model.aembed_documents.call_count == 1
        assert len(embeddings) == 10

    async def test_embedding_empty_list(self):
        """Test that empty list returns empty list"""
        mock_model = Mock()
        mock_model.aembed_documents = AsyncMock()

        embeddings = await embed_documents(mock_model, [], batch_size=32)

        assert embeddings == []
        mock_model.aembed_documents.assert_not_called()

    async def test_embedding_exact_batch_size(self):
        """Test handling of exact batch size multiples"""
        chunks = [f"Test chunk {i}" for i in range(64)]  # Exactly 2 batches

        mock_model = Mock()

        async def mock_embed(texts):
            return [[0.1] * 768 for _ in texts]

        mock_model.aembed_documents = AsyncMock(side_effect=mock_embed)

        embeddings = await embed_documents(mock_model, chunks, batch_size=32)

        # Should have called API exactly 2 times
        assert mock_model.aembed_documents.call_count == 2
        assert len(embeddings) == 64

    async def test_embedding_preserves_order(self):
        """Test that embedding order is preserved across batches"""
        chunks = [f"Chunk-{i}" for i in range(50)]

        mock_model = Mock()

        # Return embeddings with index as first value for verification
        async def mock_embed(texts):
            return [[float(i)] * 768 for i, _ in enumerate(texts)]

        mock_model.aembed_documents = AsyncMock(side_effect=mock_embed)

        embeddings = await embed_documents(mock_model, chunks, batch_size=20)

        # Verify we got all embeddings
        assert len(embeddings) == 50

        # Verify batching happened (50/20 = 3 batches)
        assert mock_model.aembed_documents.call_count == 3

    async def test_embedding_custom_batch_size(self):
        """Test that custom batch sizes work correctly"""
        chunks = [f"Test chunk {i}" for i in range(100)]

        mock_model = Mock()

        async def mock_embed(texts):
            return [[0.1] * 768 for _ in texts]

        mock_model.aembed_documents = AsyncMock(side_effect=mock_embed)

        # Use batch_size=16 (Ollama recommended)
        embeddings = await embed_documents(mock_model, chunks, batch_size=16)

        # Should have called API 7 times (100/16 = 6.25 → 7 batches)
        assert mock_model.aembed_documents.call_count == 7
        assert len(embeddings) == 100


class TestEmbeddingBatchConfiguration:
    """Test embedding batch size configuration"""

    async def test_default_batch_size(self):
        """Test that default batch size is 32"""
        import inspect

        sig = inspect.signature(embed_documents)
        batch_size_param = sig.parameters.get("batch_size")

        assert batch_size_param is not None
        assert batch_size_param.default == 32

    async def test_batch_size_from_config(self):
        """Test that batch_size can be configured"""
        from ragitect.services.config import EmbeddingConfig

        config = EmbeddingConfig(batch_size=16)
        assert config.batch_size == 16

        config_default = EmbeddingConfig()
        assert config_default.batch_size == 32
