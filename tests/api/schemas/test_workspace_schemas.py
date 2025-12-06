"""Tests for workspace API schemas

Tests verify:
- WorkspaceCreate input validation
- WorkspaceResponse camelCase serialization
- WorkspaceListResponse structure
"""

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ragitect.api.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceResponse,
    WorkspaceListResponse,
)


class TestWorkspaceCreate:
    """Tests for WorkspaceCreate schema"""

    def test_valid_workspace_create_with_name_only(self):
        """Test creating workspace with only name"""
        schema = WorkspaceCreate(name="My Workspace")
        assert schema.name == "My Workspace"
        assert schema.description is None

    def test_valid_workspace_create_with_name_and_description(self):
        """Test creating workspace with name and description"""
        schema = WorkspaceCreate(
            name="AI Research", description="Documents for AI research"
        )
        assert schema.name == "AI Research"
        assert schema.description == "Documents for AI research"

    def test_empty_name_raises_validation_error(self):
        """Test that empty name raises validation error"""
        with pytest.raises(ValidationError) as exc_info:
            WorkspaceCreate(name="")
        assert "name" in str(exc_info.value).lower()

    def test_whitespace_only_name_raises_validation_error(self):
        """Test that whitespace-only name raises validation error"""
        with pytest.raises(ValidationError) as exc_info:
            WorkspaceCreate(name="   ")
        assert "name" in str(exc_info.value).lower()

    def test_name_is_stripped(self):
        """Test that name whitespace is stripped"""
        schema = WorkspaceCreate(name="  My Workspace  ")
        assert schema.name == "My Workspace"

    def test_name_too_long_raises_validation_error(self):
        """Test that name exceeding 255 chars raises validation error"""
        with pytest.raises(ValidationError) as exc_info:
            WorkspaceCreate(name="a" * 256)
        assert "name" in str(exc_info.value).lower()


class TestWorkspaceResponse:
    """Tests for WorkspaceResponse schema"""

    def test_serializes_to_camel_case(self):
        """Test that response serializes keys as camelCase"""
        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        schema = WorkspaceResponse(
            id=workspace_id,
            name="Test Workspace",
            description="Test description",
            created_at=now,
            updated_at=now,
        )

        # model_dump with by_alias=True should produce camelCase
        data = schema.model_dump(by_alias=True, mode="json")

        # Check camelCase keys
        assert "id" in data
        assert "name" in data
        assert "description" in data
        assert "createdAt" in data
        assert "updatedAt" in data

        # Check snake_case keys are NOT present
        assert "created_at" not in data
        assert "updated_at" not in data

    def test_can_populate_by_snake_case_name(self):
        """Test that schema can be populated using snake_case names"""
        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        # Using snake_case field names (populate_by_name=True)
        schema = WorkspaceResponse(
            id=workspace_id,
            name="Test",
            description=None,
            created_at=now,
            updated_at=now,
        )

        assert schema.id == workspace_id
        assert schema.created_at == now
        assert schema.updated_at == now

    def test_optional_description(self):
        """Test that description is optional and can be None"""
        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        schema = WorkspaceResponse(
            id=workspace_id,
            name="Test",
            description=None,
            created_at=now,
            updated_at=now,
        )

        assert schema.description is None

    def test_from_orm_model(self):
        """Test creating response from ORM-like data (dict)"""
        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        # Simulate ORM model attributes as dict
        orm_data = {
            "id": workspace_id,
            "name": "From ORM",
            "description": "ORM description",
            "created_at": now,
            "updated_at": now,
        }

        schema = WorkspaceResponse.model_validate(orm_data)
        assert schema.name == "From ORM"
        assert schema.description == "ORM description"


class TestWorkspaceListResponse:
    """Tests for WorkspaceListResponse schema"""

    def test_empty_list(self):
        """Test response with empty workspaces list"""
        schema = WorkspaceListResponse(workspaces=[], total=0)

        data = schema.model_dump(by_alias=True, mode="json")
        assert data["workspaces"] == []
        assert data["total"] == 0

    def test_with_workspaces(self):
        """Test response with multiple workspaces"""
        workspace_id_1 = uuid.uuid4()
        workspace_id_2 = uuid.uuid4()
        now = datetime.now(timezone.utc)

        workspaces = [
            WorkspaceResponse(
                id=workspace_id_1,
                name="Workspace 1",
                description=None,
                created_at=now,
                updated_at=now,
            ),
            WorkspaceResponse(
                id=workspace_id_2,
                name="Workspace 2",
                description="Description 2",
                created_at=now,
                updated_at=now,
            ),
        ]

        schema = WorkspaceListResponse(workspaces=workspaces, total=2)

        data = schema.model_dump(by_alias=True, mode="json")
        assert len(data["workspaces"]) == 2
        assert data["total"] == 2
        # Verify nested workspaces also use camelCase
        assert "createdAt" in data["workspaces"][0]
        assert "updatedAt" in data["workspaces"][0]
