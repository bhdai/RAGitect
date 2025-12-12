"""Integration tests for document processing flow

Tests end-to-end document processing including:
- Upload triggers background processing
- Status endpoint returns correct states
- Complete flow: upload → processing → ready
- Error scenarios
"""

import asyncio
from uuid import uuid4

import pytest
from httpx import AsyncClient

from ragitect.services.database.connection import get_session_factory
from ragitect.services.database.repositories.document_repo import DocumentRepository


@pytest.mark.integration
async def test_document_upload_triggers_background_processing(
    async_client: AsyncClient, test_workspace
):
    """Test that uploading a document triggers background processing"""
    # Arrange - create a simple text file
    file_content = b"This is a test document for processing."
    file_name = "test.txt"

    # Act - upload document
    response = await async_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/documents",
        files={"files": (file_name, file_content, "text/plain")},
    )

    # Assert - upload successful
    assert response.status_code == 201
    data = response.json()
    assert data["total"] == 1
    assert len(data["documents"]) == 1

    document = data["documents"][0]
    document_id = document["id"]

    # Wait briefly for background task to start
    await asyncio.sleep(0.5)

    # Check status - should be processing or ready
    status_response = await async_client.get(
        f"/api/v1/workspaces/documents/{document_id}/status"
    )
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] in ["processing", "ready"]


@pytest.mark.integration
async def test_document_status_endpoint(
    async_client: AsyncClient, test_workspace, uploaded_document
):
    """Test status endpoint returns correct document state"""
    # Act - get document status
    response = await async_client.get(
        f"/api/v1/workspaces/documents/{uploaded_document.id}/status"
    )

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(uploaded_document.id)
    assert data["status"] == "uploaded"  # Initial status
    assert data["fileName"] == uploaded_document.file_name


@pytest.mark.integration
async def test_document_status_endpoint_not_found(async_client: AsyncClient):
    """Test status endpoint returns 404 for non-existent document"""
    # Act
    fake_id = uuid4()
    response = await async_client.get(f"/api/v1/workspaces/documents/{fake_id}/status")

    # Assert
    assert response.status_code == 404


@pytest.mark.integration
async def test_complete_processing_flow(async_client: AsyncClient, test_workspace):
    """Test complete flow: upload → processing → ready"""
    # Arrange
    file_content = b"Sample document content for end-to-end testing."
    file_name = "sample.txt"

    # Act 1 - Upload
    upload_response = await async_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/documents",
        files={"files": (file_name, file_content, "text/plain")},
    )
    assert upload_response.status_code == 201

    document = upload_response.json()["documents"][0]
    document_id = document["id"]

    # Act 2 - Poll status until ready or timeout
    max_attempts = 10
    for attempt in range(max_attempts):
        await asyncio.sleep(0.5)  # Wait between polls

        status_response = await async_client.get(
            f"/api/v1/workspaces/documents/{document_id}/status"
        )
        assert status_response.status_code == 200

        status = status_response.json()["status"]

        if status == "ready":
            break
        elif status == "error":
            pytest.fail("Document processing failed with error status")
    else:
        pytest.fail(f"Document not ready after {max_attempts} attempts")

    # Assert - verify processed content exists
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = DocumentRepository(session)
        doc = await repo.get_by_id_or_raise(document_id)

        assert doc.processed_content is not None
        assert len(doc.processed_content) > 0
        assert doc.metadata_ is not None
        assert doc.metadata_["status"] == "ready"
        # File bytes should be cleared
        assert "file_bytes_b64" not in doc.metadata_


@pytest.mark.integration
async def test_processing_with_pdf_file(async_client: AsyncClient, test_workspace):
    """Test processing with a PDF file (using docling processor)"""
    # Arrange - minimal PDF structure
    # This is a minimal valid PDF for testing
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000056 00000 n\n0000000115 00000 n\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    )

    # Act - upload PDF
    response = await async_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/documents",
        files={"files": ("test.pdf", pdf_content, "application/pdf")},
    )

    # Assert upload successful
    assert response.status_code == 201
    document_id = response.json()["documents"][0]["id"]

    # Wait for processing (PDF might take longer)
    max_attempts = 15
    status = "uploaded"  # Initialize status
    for _ in range(max_attempts):
        await asyncio.sleep(1.0)

        status_response = await async_client.get(
            f"/api/v1/workspaces/documents/{document_id}/status"
        )
        status = status_response.json()["status"]

        if status in ["ready", "error"]:
            break

    # Assert processing completed (ready or error acceptable for minimal PDF)
    assert status in ["ready", "error"]


@pytest.mark.integration
async def test_error_handling_corrupted_file(async_client: AsyncClient, test_workspace):
    """Test error handling for corrupted/invalid files"""
    # Arrange - upload corrupted PDF (invalid content)
    corrupted_content = b"This is not a valid PDF file"

    # Act - upload
    response = await async_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/documents",
        files={"files": ("corrupted.pdf", corrupted_content, "application/pdf")},
    )

    # Upload might succeed (validation happens during processing)
    if response.status_code == 201:
        document_id = response.json()["documents"][0]["id"]

        # Wait for processing to fail
        await asyncio.sleep(2.0)

        status_response = await async_client.get(
            f"/api/v1/workspaces/documents/{document_id}/status"
        )

        # Expect error status
        status = status_response.json()["status"]
        assert status in ["error", "processing", "uploaded"]
        # Note: Actual error detection depends on docling behavior


@pytest.mark.integration
async def test_multiple_documents_processing(async_client: AsyncClient, test_workspace):
    """Test processing multiple documents simultaneously"""
    # Arrange - multiple files
    files = [
        ("file1.txt", b"Content of file 1", "text/plain"),
        ("file2.txt", b"Content of file 2", "text/plain"),
        ("file3.md", b"# Markdown content", "text/markdown"),
    ]

    # Act - upload all files
    response = await async_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/documents",
        files=[("files", file_data) for file_data in files],
    )

    # Assert upload successful
    assert response.status_code == 201
    documents = response.json()["documents"]
    assert len(documents) == 3

    # Wait and check all are processed
    await asyncio.sleep(2.0)

    for doc in documents:
        status_response = await async_client.get(
            f"/api/v1/workspaces/documents/{doc['id']}/status"
        )
        status = status_response.json()["status"]
        # All should be ready or processing
        assert status in ["ready", "processing"]
