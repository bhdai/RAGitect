"""Workspace API schemas

Pydantic models for workspace-related API request/response bodies.
All response schemas use camelCase serialization for frontend compatibility.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


class WorkspaceCreate(BaseModel):
    """Schema for creating a new workspace

    Attributes:
        name: Workspace name (required, non-empty, max 255 chars)
        description: Optional workspace description
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Workspace name (required, non-empty, max 255 chars)",
    )
    description: str | None = Field(
        default=None, description="Optional workspace description"
    )

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v: str) -> str:
        """Strip whitespace from name before validation"""
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("name", mode="after")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Ensure name is not empty after stripping"""
        if not v:
            raise ValueError("name cannot be empty")
        return v


class WorkspaceResponse(BaseModel):
    """Schema for workspace response with camelCase serialization

    This schema is used for API responses. All field names are serialized
    as camelCase for frontend consumption while accepting snake_case input
    from ORM models.

    Attributes:
        id: Unique workspace identifier (UUID)
        name: Workspace name
        description: Optional workspace description
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        from_attributes=True,
    )

    id: uuid.UUID = Field(..., description="Unique workspace identifier")
    name: str = Field(..., description="Workspace name")
    description: str | None = Field(None, description="Optional workspace description")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class WorkspaceListResponse(BaseModel):
    """Schema for list of workspaces response

    Attributes:
        workspaces: List of workspace response objects
        total: Total count of workspaces
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    workspaces: list[WorkspaceResponse] = Field(
        default_factory=list, description="List of workspaces"
    )
    total: int = Field(..., description="Total count of workspaces")
