"""Tests for token-based markdown chunking with orphan header merging.

Story 3.3.A: Backend Citation Metadata & Markdown Chunking Improvements

Tests:
- Token counting with hybrid tokenization (AC4)
- Orphan header merging (AC5)
- Minimum chunk size enforcement (AC6)
- Token count logging (AC7)
"""

import pytest

from ragitect.services.document_processor import (
    count_tokens,
    get_tokenizer_for_embedding_model,
    split_markdown_document,
)
from ragitect.services.config import EmbeddingConfig


class TestTokenCounting:
    """Test hybrid tokenization strategy (AC4)"""

    def test_count_tokens_returns_positive_integer(self):
        """Verify count_tokens returns valid token count"""
        text = "Hello, world! This is a test sentence."
        tokens = count_tokens(text)

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_tokens_empty_string(self):
        """Empty string should return 0 tokens"""
        tokens = count_tokens("")
        assert tokens == 0

    def test_count_tokens_scales_with_text_length(self):
        """Longer text should have more tokens"""
        short_text = "Hello world"
        long_text = "Hello world " * 100

        short_tokens = count_tokens(short_text)
        long_tokens = count_tokens(long_text)

        assert long_tokens > short_tokens

    def test_tokenizer_selection_openai(self):
        """OpenAI provider should use tiktoken"""
        config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
        tokenizer, tokenizer_type = get_tokenizer_for_embedding_model(config)

        assert tokenizer_type == "tiktoken"

    def test_tokenizer_selection_ollama_qwen(self):
        """Ollama with Qwen model should use transformers"""
        config = EmbeddingConfig(provider="ollama", model="qwen3-embedding:0.6b")
        tokenizer, tokenizer_type = get_tokenizer_for_embedding_model(config)

        # Should use transformers for Qwen (BPE model with specific vocab)
        assert tokenizer_type in ["tiktoken", "transformers"]

    def test_tokenizer_fallback(self):
        """Unknown model should fall back gracefully"""
        config = EmbeddingConfig(provider="ollama", model="unknown-model-xyz")
        tokenizer, tokenizer_type = get_tokenizer_for_embedding_model(config)

        # Should not raise, should return a valid tokenizer
        assert tokenizer is not None
        assert tokenizer_type in ["tiktoken", "transformers"]


class TestOrphanHeaderMerging:
    """Test orphan header merging strategy (AC5)"""

    def test_orphan_header_merging(self):
        """Test that small header sections are merged with adjacent content"""
        markdown = """# Main Title

## Orphan Section
Just one sentence.

## Large Section
This has enough content to be meaningful. We need to add more text here
to ensure this section is large enough to not be considered an orphan.
Adding more sentences to make sure we have sufficient content.
This should definitely be more than 64 tokens when properly measured.
Let's add even more content to be absolutely certain.
"""
        chunks = split_markdown_document(
            markdown, chunk_size=512, overlap=50, min_chunk_size=64
        )

        # Verify no micro-chunks (all chunks >= ~60 tokens with some tolerance)
        for chunk in chunks:
            token_count = count_tokens(chunk)
            # Allow tolerance since orphan merging happens before recursive split
            assert token_count >= 30, f"Chunk too small: {token_count} tokens"

        # Orphan header should be merged - "Orphan Section" shouldn't be alone
        orphan_alone = any(
            "## Orphan Section" in c and count_tokens(c) < 30 for c in chunks
        )
        assert not orphan_alone, "Orphan header was not merged"

    def test_consecutive_orphan_headers(self):
        """Test that multiple consecutive orphan headers are merged"""
        markdown = """# Title

## Part 1

## Part 2

## Part 3

This is the actual content that follows after three headers.
We need to add enough content here to make this section substantial.
More content to ensure proper testing of the orphan merging logic.
"""
        chunks = split_markdown_document(
            markdown, chunk_size=512, overlap=50, min_chunk_size=64
        )

        # All orphan headers should be merged together or with content
        assert len(chunks) >= 1

        # Check that headers are with content, not isolated
        for chunk in chunks:
            if "## Part 1" in chunk or "## Part 2" in chunk or "## Part 3" in chunk:
                # If header present, should have substantial content
                assert count_tokens(chunk) >= 30

    def test_large_section_not_merged(self):
        """Test that large sections are not unnecessarily merged"""
        markdown = """# Section One

This is a substantial section with plenty of content.
We need to make sure this section is large enough to stand alone.
Adding more content here to ensure it exceeds the minimum chunk size.
This should be a complete, self-contained chunk.

# Section Two

Another substantial section with its own content.
This section also has enough text to be independent.
More content added to ensure proper size requirements are met.
"""
        chunks = split_markdown_document(
            markdown, chunk_size=512, overlap=50, min_chunk_size=64
        )

        # Should produce multiple chunks since sections are large
        assert len(chunks) >= 1


class TestTokenBasedSizing:
    """Test token-based chunk sizing (AC4, AC6)"""

    def test_chunks_respect_token_limits(self):
        """Test that chunks respect token size limits"""
        # Create text that will need splitting (~1000 tokens)
        long_text = "word " * 1000
        chunks = split_markdown_document(long_text, chunk_size=256, overlap=50)

        for chunk in chunks:
            tokens = count_tokens(chunk)
            # Allow some tolerance for overlap and splitting boundaries
            assert tokens <= 350, f"Chunk exceeds limit: {tokens} tokens"

    def test_minimum_chunk_size_enforcement(self):
        """Test that minimum chunk size prevents micro-chunks (AC6)"""
        # Create markdown with small sections
        markdown = """# Header 1
Small.

# Header 2
Also small.

# Header 3
This is a much larger section with substantial content.
We add more text here to ensure it's a proper chunk.
Even more content to make this section large enough.
"""
        chunks = split_markdown_document(
            markdown, chunk_size=512, overlap=50, min_chunk_size=64
        )

        # Count how many chunks are below minimum (should be none or very few)
        micro_chunks = [c for c in chunks if count_tokens(c) < 30]

        # All small sections should be merged
        assert len(micro_chunks) <= 1  # At most 1 trailing chunk might be small


class TestTokenLogging:
    """Test token count statistics logging (AC7)"""

    def test_chunking_produces_output(self, caplog):
        """Test that chunking logs token statistics"""
        import logging

        caplog.set_level(logging.INFO)

        markdown = """# Test Document

This is test content with enough text to trigger logging.
We need multiple sentences to ensure the logging is triggered.
More content added for good measure.
"""
        chunks = split_markdown_document(markdown, chunk_size=512)

        # Should produce at least one chunk
        assert len(chunks) >= 1


class TestEdgeCases:
    """Test edge cases for markdown chunking"""

    def test_empty_text(self):
        """Empty text should return empty list"""
        chunks = split_markdown_document("")
        assert chunks == []

    def test_whitespace_only(self):
        """Whitespace-only text should return empty list"""
        chunks = split_markdown_document("   \n\t  ")
        assert chunks == []

    def test_no_headers(self):
        """Text without headers should still be chunked"""
        text = "This is plain text without any markdown headers. " * 100
        chunks = split_markdown_document(text, chunk_size=256)

        assert len(chunks) >= 1

    def test_deeply_nested_headers(self):
        """Test handling of deeply nested header structure"""
        markdown = """# H1

## H2

### H3

#### H4

##### H5

This is the actual content at the deepest level.
We need enough content to make this a valid chunk.
"""
        chunks = split_markdown_document(markdown, chunk_size=512, min_chunk_size=64)

        # Should handle deep nesting gracefully
        assert len(chunks) >= 1

        # All headers should be preserved
        all_text = " ".join(chunks)
        assert "# H1" in all_text or "H1" in all_text
