"""Tests for Chat API schemas.

Story 3.2.B: Streaming LLM Responses with Citations - Task 1.2
Story 3.3.A: Backend Citation Metadata & Markdown Chunking Improvements

Tests for Citation model serialization and factory methods.
"""

import pytest

from ragitect.api.schemas.chat import Citation, CitationMetadata


class TestCitationMetadata:
    """Tests for CitationMetadata model."""

    def test_serialization_camel_case(self):
        """Test that metadata fields serialize to camelCase."""
        metadata = CitationMetadata(
            chunk_index=5,
            similarity=0.92,
            preview="Python is a programming language...",
            document_id="550e8400-e29b-41d4-a716-446655440000",
        )

        data = metadata.model_dump(by_alias=True)

        assert data["chunkIndex"] == 5
        assert data["similarity"] == 0.92
        assert data["preview"] == "Python is a programming language..."
        assert data["documentId"] == "550e8400-e29b-41d4-a716-446655440000"

    def test_similarity_bounds(self):
        """Test that similarity is validated between 0 and 1."""
        # Valid bounds
        CitationMetadata(
            chunk_index=0,
            similarity=0.0,
            preview="test",
            document_id="test-id",
        )
        CitationMetadata(
            chunk_index=0,
            similarity=1.0,
            preview="test",
            document_id="test-id",
        )
        CitationMetadata(
            chunk_index=0,
            similarity=0.5,
            preview="test",
            document_id="test-id",
        )

        # Invalid bounds
        with pytest.raises(ValueError):
            CitationMetadata(
                chunk_index=0,
                similarity=-0.1,
                preview="test",
                document_id="test-id",
            )

        with pytest.raises(ValueError):
            CitationMetadata(
                chunk_index=0,
                similarity=1.1,
                preview="test",
                document_id="test-id",
            )

    def test_preview_accepts_long_content(self):
        """Test that preview accepts full content without truncation."""
        long_text = "x" * 5000
        # Should not raise ValueError - full content is accepted
        metadata = CitationMetadata(
            chunk_index=0,
            similarity=0.5,
            preview=long_text,
            document_id="test-id",
        )
        assert len(metadata.preview) == 5000


class TestCitation:
    """Tests for Citation model."""

    def test_from_context_chunk(self):
        """Test creating Citation from context chunk data."""
        citation = Citation.from_context_chunk(
            index=0,
            document_id="550e8400-e29b-41d4-a716-446655440000",
            document_name="intro.pdf",
            chunk_index=3,
            similarity=0.95,
            content="Python is a powerful programming language used for...",
        )

        assert citation.source_id == "cite-0"
        assert citation.media_type == "text/plain"
        assert citation.title == "intro.pdf"
        assert citation.provider_metadata["ragitect"]["chunkIndex"] == 3
        assert citation.provider_metadata["ragitect"]["similarity"] == 0.95
        assert (
            "Python is a powerful" in citation.provider_metadata["ragitect"]["preview"]
        )
        assert (
            citation.provider_metadata["ragitect"]["documentId"]
            == "550e8400-e29b-41d4-a716-446655440000"
        )

    def test_from_context_chunk_preserves_full_content(self):
        """Test that full content is preserved without truncation."""
        long_content = "x" * 1000

        citation = Citation.from_context_chunk(
            index=1,
            document_id="test-doc-id",
            document_name="test.pdf",
            chunk_index=0,
            similarity=0.8,
            content=long_content,
        )

        preview = citation.provider_metadata["ragitect"]["preview"]
        assert len(preview) == 1000  # Full content preserved
        assert preview == long_content

    def test_to_sse_dict(self):
        """Test conversion to AI SDK source-document event format."""
        citation = Citation.from_context_chunk(
            index=2,
            document_id="doc-uuid-123",
            document_name="advanced.pdf",
            chunk_index=7,
            similarity=0.87,
            content="This chapter covers advanced topics...",
        )

        sse_dict = citation.to_sse_dict()

        assert sse_dict["type"] == "source-document"
        assert sse_dict["sourceId"] == "cite-2"
        assert sse_dict["mediaType"] == "text/plain"
        assert sse_dict["title"] == "advanced.pdf"
        assert sse_dict["providerMetadata"]["ragitect"]["chunkIndex"] == 7
        assert sse_dict["providerMetadata"]["ragitect"]["similarity"] == 0.87
        assert sse_dict["providerMetadata"]["ragitect"]["documentId"] == "doc-uuid-123"

    def test_serialization_camel_case(self):
        """Test that Citation fields serialize to camelCase."""
        citation = Citation(
            source_id="cite-0",
            media_type="text/plain",
            title="test.pdf",
            provider_metadata={"ragitect": {"chunkIndex": 0, "documentId": "doc-id"}},
        )

        data = citation.model_dump(by_alias=True)

        assert data["sourceId"] == "cite-0"
        assert data["mediaType"] == "text/plain"
        assert data["providerMetadata"]["ragitect"]["chunkIndex"] == 0

    def test_multiple_citations_sequence(self):
        """Test creating multiple sequential citations."""
        chunks = [
            {
                "document_id": "doc-1-uuid",
                "document_name": "doc1.pdf",
                "chunk_index": 0,
                "similarity": 0.95,
                "content": "First chunk...",
            },
            {
                "document_id": "doc-2-uuid",
                "document_name": "doc2.pdf",
                "chunk_index": 2,
                "similarity": 0.88,
                "content": "Second chunk...",
            },
            {
                "document_id": "doc-1-uuid",
                "document_name": "doc1.pdf",
                "chunk_index": 5,
                "similarity": 0.82,
                "content": "Third chunk...",
            },
        ]

        citations = [
            Citation.from_context_chunk(
                index=i,
                document_id=c["document_id"],
                document_name=c["document_name"],
                chunk_index=c["chunk_index"],
                similarity=c["similarity"],
                content=c["content"],
            )
            for i, c in enumerate(chunks)
        ]

        assert len(citations) == 3
        assert citations[0].source_id == "cite-0"
        assert citations[1].source_id == "cite-1"
        assert citations[2].source_id == "cite-2"
        assert citations[0].title == "doc1.pdf"
        assert citations[1].title == "doc2.pdf"
