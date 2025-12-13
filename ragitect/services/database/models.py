"""SQLAlchemy ORM models for the RAGitect database

This module defines the database schema using SQLAlchemy's declarative base:
- Workspace: Container for document collections
- Document: Processed files with embeddings
- DocumentChunk: Text segments from documents with embeddings
"""

import uuid
from datetime import datetime
from typing import Any, override

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql.functions import func


class Base(DeclarativeBase):
    """Base class for all of the ORM models"""

    # prevent base talbe from created
    __abstract__: bool = True


class Workspace(Base):
    """Workspace model representing a container for document collections.

    Attributes:
        id: Unique identifier (UUID)
        name: Workspace name (unique)
        description: Optional workspace description
        created_at: Creation timestamp
        updated_at: Last update timestamp
        documents: Relationship to associated documents

    Constraints:
        - name must be unique
        - name cannot be empty
    """

    __tablename__: str = "workspaces"

    # primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique workspace identifier",
    )

    # core fields
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,  # automatically creates an index
        comment="Workspace name (unique)",
    )

    description: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Optional workspace description"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Creation timestamp",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last update timestamp",
    )

    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="workspace",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (  # pyright: ignore[reportAny, reportUnannotatedClassAttribute]
        CheckConstraint(sqltext="length(name) > 0", name="workspace_name_not_empty"),
        {"comment": "Workspaces for organizing document collections"},
    )

    @override
    def __repr__(self) -> str:
        return f"<Workspace(id={self.id}, name='{self.name}')>"


class Document(Base):
    """Document model representing processed files with embeddings.

    Attributes:
        id: Unique document identifier (UUID)
        workspace_id: Parent workspace id (UUID)
        file_name: Original file name
        file_type: File type or extension (e.g., 'pdf', 'docx')
        content_hash: SHA-256 hash of 'workspace_id:content'
        unique_identifier_hash: Unique hash across the entire system
        processed_content: Full text content after ETL processing (should be md)
        summary: Optional document summary
        embedding: Document-level embedding vector (768 dims)
        processed_at: When processing was completed
        metadata_: Flexible JSONB field for additional metadata (embedding model, page count, etc.)
        workspace: Relationship to parent Workspace
        chunks: Relationship to document chunks

    """

    __tablename__: str = "documents"

    # pk
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique document identifier",
    )

    # fk
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(column="workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent workspace id",
    )

    # core fields
    file_name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Original file name",
    )

    file_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="File type or extension (e.g., 'pdf', 'docx')",
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash of 'content'",
    )

    unique_identifier_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        comment="Unique hash across the entire system",
    )

    processed_content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Full text content after ETL processing (should be md)",
    )

    summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Optional document summary"
    )

    # vector embedding
    embedding: Mapped[Any] = mapped_column(
        Vector(768), nullable=True, comment="Document-level embedding vector (768 dims)"
    )

    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Processing completion timestamp",
    )

    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        __name_pos="metadata",
        __type_pos=MutableDict.as_mutable(JSONB),
        nullable=True,
        default=dict,
        comment="Additional metadata (embedding model, page count, etc.)",
    )

    # relationship
    workspace: Mapped["Workspace"] = relationship(
        "Workspace", back_populates="documents"
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (  # pyright: ignore[reportAny, reportUnannotatedClassAttribute]
        {"comment": "Processed documents with embeddings"},
    )

    @override
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, file_name='{self.file_name}', workspace_id={self.workspace_id})>"


class DocumentChunk(Base):
    """DocumentChunk model representing text segments from documents with embeddings.

    Attributes:
        id: Unique identifier (UUID)
        document_id: Parent document id (UUID)
        workspace_id: Parent workspace id (UUID)
        chunk_index: Order index of the chunk within the document (0-based)
        content: Chunk text content
        embedding: Chunk-level embedding vector (768 dims)
        metadata_: Flexible JSONB field for additional metadata (page_number, section, etc.)
        created_at: Creation timestamp
        document: Relationship to parent Document

    Constraints:
        - document_id and chunk_index must be unique together
    """

    __tablename__: str = "document_chunks"

    # pk
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique document chunk identifier",
    )

    # fk
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
        comment="Parent document id",
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
        comment="Parent workspace id",
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Index of the chunk within the document (0-based)",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Chunk text content",
    )

    embedding: Mapped[Any] = mapped_column(
        Vector(768),
        nullable=False,
        comment="Chunk-level embedding vector (768 dims)",
    )

    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        __name_pos="metadata",
        __type_pos=MutableDict.as_mutable(JSONB),
        nullable=True,
        default=dict,
        comment="Additional metadata (page_number, section, etc.)",
    )

    # timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Creation timestamp",
    )

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )

    __table_args__ = (  # pyright: ignore[reportAny, reportUnannotatedClassAttribute]
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunk_index",
        ),
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        CheckConstraint("length(content) > 0", name="chunk_content_not_empty"),
        {"comment": "Document chunks with embeddings for retrieval"},
    )

    @override
    def __repr__(self) -> str:
        content_preview = (
            self.content[:50] + "..." if len(self.content) > 50 else self.content
        )
        return f"<DocumentChunk(id={self.id}, document_id='{self.document_id}', chunk_index={self.chunk_index}, content='{content_preview}')>"


class LLMProviderConfig(Base):
    """LLM provider configuration model for storing provider settings.

    Attributes:
        id: Unique identifier (UUID)
        provider_name: Provider name (ollama, openai, anthropic)
        config_data: JSONB field for flexible configuration storage
        is_active: Whether this configuration is currently active
        created_at: Creation timestamp
        updated_at: Last update timestamp

    Constraints:
        - provider_name must be unique
        - provider_name must be one of: ollama, openai, anthropic
    """

    __tablename__: str = "llm_provider_configs"

    # pk
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique configuration identifier",
    )

    # core fields
    provider_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
        comment="Provider name (ollama, openai, anthropic, gemini)",
    )

    config_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Configuration data (base_url, api_key, model, etc.)",
    )

    is_active: Mapped[bool] = mapped_column(
        nullable=False,
        server_default="true",
        comment="Whether this configuration is active",
    )

    # timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Creation timestamp",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last update timestamp",
    )

    __table_args__ = (  # pyright: ignore[reportAny, reportUnannotatedClassAttribute]
        CheckConstraint(
            "provider_name IN ('ollama', 'openai', 'anthropic', 'gemini', 'openai_compatible')",
            name="valid_provider_name",
        ),
        {"comment": "LLM provider configurations with encrypted API keys"},
    )

    @override
    def __repr__(self) -> str:
        return f"<LLMProviderConfig(id={self.id}, provider_name='{self.provider_name}', is_active={self.is_active})>"


class EmbeddingProviderConfig(Base):
    """Embedding provider configuration model.

    Attributes:
        id: Unique identifier (UUID)
        provider_name: Embedding provider name (ollama, openai, vertex_ai)
        config_data: JSONB field for flexible configuration storage
        is_active: Whether this configuration is currently active
        created_at: Creation timestamp
        updated_at: Last update timestamp

    Constraints:
        - provider_name must be unique
        - provider_name must be one of: ollama, openai, vertex_ai, openai_compatible
    """

    __tablename__: str = "embedding_configs"

    # pk
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique embedding config identifier",
    )

    # core fields
    provider_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
        comment="Embedding provider name",
    )

    config_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Configuration data (base_url, api_key, model, dimension, etc.)",
    )

    is_active: Mapped[bool] = mapped_column(
        nullable=False,
        server_default="true",
        comment="Whether this configuration is active",
    )

    # timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Creation timestamp",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last update timestamp",
    )

    __table_args__ = (  # pyright: ignore[reportAny, reportUnannotatedClassAttribute]
        CheckConstraint(
            "provider_name IN ('ollama', 'openai', 'vertex_ai', 'openai_compatible')",
            name="valid_embedding_provider_name",
        ),
        {"comment": "Embedding provider configurations"},
    )

    @override
    def __repr__(self) -> str:
        return f"<EmbeddingProviderConfig(id={self.id}, provider_name='{self.provider_name}', is_active={self.is_active})>"
