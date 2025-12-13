"""Integration tests for document upload API endpoints

These tests run against a real database to verify end-to-end functionality.
They use shared fixtures from conftest.py which handle database setup,
cleanup, and test client creation.

To run these tests:
    uv run --env-file .env.test pytest tests/api/v1/test_documents_integration.py -m integration
"""

import io
import uuid

import pytest
from httpx import AsyncClient

# Apply asyncio and integration markers to all tests in this module
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_upload_single_document(
    shared_integration_client: AsyncClient, test_workspace
):
    """Test uploading a single document to workspace"""
    # Create test file
    file_content = b"This is a test PDF content"
    files = {"files": ("test.pdf", io.BytesIO(file_content), "application/pdf")}

    response = await shared_integration_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/documents",
        files=files,
    )

    assert response.status_code == 201
    data = response.json()

    assert "documents" in data
    assert "total" in data
    assert data["total"] == 1
    assert len(data["documents"]) == 1

    doc = data["documents"][0]
    assert doc["fileName"] == "test.pdf"
    assert doc["fileType"] == ".pdf"
    assert doc["status"] == "uploaded"
    assert "id" in doc
    assert "createdAt" in doc


async def test_upload_multiple_documents(
    shared_integration_client: AsyncClient, test_workspace
):
    """Test uploading multiple documents at once"""
    # Create multiple test files
    files = [
        ("files", ("test1.pdf", io.BytesIO(b"content1"), "application/pdf")),
        ("files", ("test2.txt", io.BytesIO(b"content2"), "text/plain")),
        (
            "files",
            ("test3.md", io.BytesIO(b"# Markdown"), "text/markdown"),
        ),
    ]

    response = await shared_integration_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/documents",
        files=files,
    )

    assert response.status_code == 201
    data = response.json()

    assert data["total"] == 3
    assert len(data["documents"]) == 3

    filenames = [doc["fileName"] for doc in data["documents"]]
    assert "test1.pdf" in filenames
    assert "test2.txt" in filenames
    assert "test3.md" in filenames


async def test_upload_unsupported_format(
    shared_integration_client: AsyncClient, test_workspace
):
    """Test uploading an unsupported file format"""
    files = {
        "files": (
            "test.exe",
            io.BytesIO(b"executable content"),
            "application/x-executable",
        )
    }

    response = await shared_integration_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/documents",
        files=files,
    )

    assert response.status_code == 400
    data = response.json()
    assert "Unsupported format" in data["detail"]


async def test_upload_to_nonexistent_workspace(
    shared_integration_client: AsyncClient, clean_database
):
    """Test uploading to a workspace that doesn't exist"""
    fake_workspace_id = uuid.uuid4()
    files = {"files": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}

    response = await shared_integration_client.post(
        f"/api/v1/workspaces/{fake_workspace_id}/documents",
        files=files,
    )

    assert response.status_code == 404
    data = response.json()
    assert "Workspace not found" in data["detail"]


async def test_response_uses_camel_case(
    shared_integration_client: AsyncClient, test_workspace
):
    """Test that API response uses camelCase for field names"""
    files = {"files": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}

    response = await shared_integration_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/documents",
        files=files,
    )

    assert response.status_code == 201
    data = response.json()

    # Verify camelCase keys
    doc = data["documents"][0]
    assert "fileName" in doc  # not file_name
    assert "fileType" in doc  # not file_type
    assert "createdAt" in doc  # not created_at

    # Verify snake_case keys are NOT present
    assert "file_name" not in doc
    assert "file_type" not in doc
    assert "created_at" not in doc
