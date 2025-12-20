"""Chat API schemas for streaming citations.

Story 3.2.B: Streaming LLM Responses with Citations

Pydantic models for citation metadata in AI SDK streaming responses.
Uses camelCase serialization for frontend compatibility.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CitationMetadata(BaseModel):
    """Custom RAGitect citation metadata for providerMetadata.ragitect.

    This data is embedded in AI SDK source-document parts under providerMetadata.
    Frontend extracts this for tooltip display.

    Attributes:
        chunk_index: Index of the chunk within the document
        similarity: Relevance score 0-1 from vector search/reranking
        preview: Full chunk content for tooltip preview
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    chunk_index: int = Field(..., description="Chunk index within document")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0-1")
    preview: str = Field(..., description="Full chunk content for tooltip preview")


class Citation(BaseModel):
    """Citation metadata for AI SDK UI Message Stream Protocol v1.

    This model represents a source-document part in the streaming protocol.
    Each citation corresponds to a context chunk from RAG retrieval.

    Story 3.2.B: AC1 (backend embeds citation markers), AC2 (citations have metadata)

    Protocol Reference:
    - type: "source-document" (AI SDK standard)
    - sourceId: "cite-0", "cite-1", etc.
    - mediaType: "text/plain" for text chunks
    - title: Document filename
    - providerMetadata.ragitect: Custom citation metadata

    Attributes:
        source_id: Unique citation ID (cite-0, cite-1, ...)
        media_type: MIME type of source (always "text/plain" for text chunks)
        title: Source document filename
        provider_metadata: Container for ragitect-specific metadata
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    # source_id maps to sourceId in JSON (AI SDK format)
    source_id: str = Field(
        ...,
        description="Unique citation ID (cite-0, cite-1, ...)",
    )
    media_type: Literal["text/plain"] = Field(
        "text/plain",
        description="MIME type of source content",
    )
    title: str = Field(..., description="Source document filename")
    provider_metadata: dict = Field(
        default_factory=dict,
        description="Container for ragitect-specific metadata",
    )

    @classmethod
    def from_context_chunk(
        cls,
        index: int,
        document_name: str,
        chunk_index: int,
        similarity: float,
        content: str,
    ) -> "Citation":
        """Create a Citation from a RAG context chunk.

        Args:
            index: Citation index (0, 1, 2, ...)
            document_name: Source document filename
            chunk_index: Chunk index within document
            similarity: Relevance score from retrieval
            content: Full chunk content (sent in its entirety)

        Returns:
            Citation instance ready for streaming
        """
        return cls(
            source_id=f"cite-{index}",
            media_type="text/plain",
            title=document_name,
            provider_metadata={
                "ragitect": {
                    "chunkIndex": chunk_index,
                    "similarity": similarity,
                    "preview": content,
                }
            },
        )

    def to_sse_dict(self) -> dict:
        """Convert to dict for SSE source-document event.

        Returns:
            Dict matching AI SDK source-document part format
        """
        return {
            "type": "source-document",
            "sourceId": self.source_id,
            "mediaType": self.media_type,
            "title": self.title,
            "providerMetadata": self.provider_metadata,
        }
