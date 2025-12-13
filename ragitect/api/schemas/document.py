"""Document API schemas

Pydantic models for document-related API request/response bodies.
All response schemas use camelCase serialization for frontend compatibility.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response with camelCase serialization

    Attributes:
        id: Unique document identifier (UUID)
        file_name: Original file name
        file_type: File extension or type
        status: Upload/processing status (uploaded, processing, ready, error)
        created_at: Upload timestamp
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        from_attributes=True,
    )

    id: uuid.UUID = Field(..., description="Unique document identifier")
    file_name: str = Field(..., description="Original file name")
    file_type: str | None = Field(None, description="File extension or type")
    status: str = Field(default="uploaded", description="Document status")
    created_at: datetime = Field(..., description="Upload timestamp")


class DocumentListResponse(BaseModel):
    """Schema for list of documents response

    Attributes:
        documents: List of document response objects
        total: Total count of documents
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    documents: list[DocumentUploadResponse] = Field(
        default_factory=list, description="List of documents"
    )
    total: int = Field(..., description="Total count of documents")


class DocumentStatusResponse(BaseModel):
    """Schema for document status response

    Attributes:
        id: Unique document identifier (UUID)
        status: Processing status (uploaded, processing, embedding, ready, error)
        file_name: Original file name
        phase: Current processing phase for detailed progress (parsing, embedding, null)
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        from_attributes=True,
    )

    id: uuid.UUID = Field(..., description="Unique document identifier")
    status: str = Field(..., description="Document processing status")
    file_name: str = Field(..., description="Original file name")
    phase: str | None = Field(
        None,
        description="Current processing phase: 'parsing', 'embedding', or null when complete",
    )


class DocumentDetailResponse(BaseModel):
    """Schema for document detail response with full content

    Attributes:
        id: Unique document identifier (UUID)
        file_name: Original file name
        file_type: File extension or type
        status: Processing status (uploaded, processing, embedding, ready, error)
        processed_content: Extracted text content (markdown format from docling)
        summary: Optional document summary
        created_at: Upload timestamp
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        from_attributes=True,
    )

    id: uuid.UUID = Field(..., description="Unique document identifier")
    file_name: str = Field(..., description="Original file name")
    file_type: str | None = Field(None, description="File extension or type")
    status: str = Field(default="uploaded", description="Document processing status")
    processed_content: str | None = Field(
        None, description="Extracted text content in markdown format"
    )
    summary: str | None = Field(None, description="Optional document summary")
    created_at: datetime = Field(..., description="Upload timestamp")
