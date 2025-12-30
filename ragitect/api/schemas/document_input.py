"""Document input API schemas for URL-based ingestion.

Pydantic models for document input requests supporting both file uploads
and URL-based ingestion with discriminated union pattern.
"""

from typing import Annotated, Literal

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, UrlConstraints
from pydantic.alias_generators import to_camel


# Custom URL type with explicit constraints.
# NOTE: We intentionally do NOT constrain allowed schemes here so the API can
# return a 400 with the required error message for non-HTTP(S) schemes (AC2).
SafeIngestUrl = Annotated[
    AnyUrl,
    UrlConstraints(
        max_length=2000,
        host_required=False,
    ),
]


class URLUploadInput(BaseModel):
    """Schema for URL-based document upload input.

    Used for submitting URLs for document ingestion (web pages, YouTube, PDFs).
    source_type determines the processing strategy.

    Attributes:
        source_type: Type of URL source - "url" (web page), "youtube", or "pdf"
        url: The HTTP/HTTPS URL to ingest

    Example:
        ```json
        {
            "sourceType": "url",
            "url": "https://example.com/article"
        }
        ```

    Security Notes:
        - Only HTTP and HTTPS URLs are allowed
        - Private IPs (10.x.x.x, 172.16.x.x, 192.168.x.x) are blocked
        - Localhost addresses are blocked
        - Cloud metadata endpoints (169.254.x.x) are blocked
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        json_schema_extra={
            "examples": [
                {"sourceType": "url", "url": "https://example.com/article"},
                {
                    "sourceType": "youtube",
                    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                },
                {"sourceType": "pdf", "url": "https://arxiv.org/pdf/2301.00001.pdf"},
            ]
        },
    )

    source_type: Literal["url", "youtube", "pdf"] = Field(
        ...,
        description="Type of URL source: 'url' for web pages, 'youtube' for videos, 'pdf' for PDF files",
    )
    url: SafeIngestUrl = Field(
        ...,
        description="The HTTP/HTTPS URL to ingest",
    )


class URLUploadResponse(BaseModel):
    """Schema for URL upload response.

    Same structure as DocumentUploadResponse but with URL-specific metadata.

    Attributes:
        id: Unique document identifier (UUID)
        source_type: Type of URL source
        source_url: The submitted URL
        status: Processing status (backlog = queued for fetching)
        message: Human-readable status message

    Example:
        ```json
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "sourceType": "url",
            "sourceUrl": "https://example.com/article",
            "status": "backlog",
            "message": "URL submitted for ingestion. Processing will begin shortly."
        }
        ```
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "sourceType": "url",
                "sourceUrl": "https://example.com/article",
                "status": "backlog",
                "message": "URL submitted for ingestion. Processing will begin shortly.",
            }
        },
    )

    id: str = Field(..., description="Unique document identifier (UUID)")
    source_type: Literal["url", "youtube", "pdf"] = Field(
        ..., description="Type of URL source"
    )
    source_url: str = Field(..., description="The submitted URL")
    status: str = Field(
        default="backlog",
        description="Document status: 'backlog' means queued for fetching",
    )
    message: str = Field(
        default="URL submitted for ingestion",
        description="Human-readable status message",
    )
