"""Tests for chat streaming endpoint (SSE) with RAG integration.

Tests verify:
- POST /api/v1/workspaces/{workspace_id}/chat/stream returns SSE content-type
- Endpoint streams data in SSE format (data: ...)
- Workspace validation (404 for invalid workspace)
- Empty message handling
- RAG retrieval from workspace documents (Story 3.1)
- Context used in LLM prompt (Story 3.1)
- Empty workspace handling (Story 3.1)
- Chat history support (Story 3.1)

Story 3.0: Streaming Infrastructure (Prep)
Story 3.1: Natural Language Querying
"""

import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio


class TestChatStreamEndpoint:
    """Tests for POST /api/v1/workspaces/{workspace_id}/chat/stream."""

    async def test_stream_returns_sse_content_type(self, async_client, mocker):
        """Test endpoint returns text/event-stream content type."""
        from ragitect.services.database.models import Workspace
        from datetime import datetime, timezone

        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        mock_workspace = Workspace(
            id=workspace_id,
            name="Test Workspace",
        )
        mock_workspace.created_at = now
        mock_workspace.updated_at = now

        # Mock WorkspaceRepository
        mock_ws_repo = mocker.AsyncMock()
        mock_ws_repo.get_by_id.return_value = mock_workspace

        mocker.patch(
            "ragitect.api.v1.chat.WorkspaceRepository",
            return_value=mock_ws_repo,
        )

        # Mock DocumentRepository to return docs exist (skip empty workspace handling)
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_workspace_count.return_value = 5

        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        # Mock retrieve_context to skip actual RAG
        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context",
            return_value=[
                {"content": "test", "document_name": "test.txt", "chunk_index": 0}
            ],
        )

        # Mock LLM streaming
        async def mock_stream(llm, messages):
            for chunk in ["Hello", " ", "World"]:
                yield chunk

        mocker.patch(
            "ragitect.api.v1.chat.generate_response_stream",
            side_effect=mock_stream,
        )

        # Mock create_llm_from_db
        mock_llm = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_from_db",
            return_value=mock_llm,
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "Hello"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    async def test_stream_format_is_sse(self, async_client, mocker):
        """Test response is formatted as Server-Sent Events."""
        from ragitect.services.database.models import Workspace
        from datetime import datetime, timezone

        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        mock_workspace = Workspace(id=workspace_id, name="Test")
        mock_workspace.created_at = now
        mock_workspace.updated_at = now

        mock_ws_repo = mocker.AsyncMock()
        mock_ws_repo.get_by_id.return_value = mock_workspace

        mocker.patch(
            "ragitect.api.v1.chat.WorkspaceRepository",
            return_value=mock_ws_repo,
        )

        # Mock DocumentRepository to return docs exist
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_workspace_count.return_value = 5

        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        # Mock retrieve_context
        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context",
            return_value=[
                {"content": "test", "document_name": "test.txt", "chunk_index": 0}
            ],
        )

        # Mock LLM to yield test chunks
        async def mock_stream(llm, messages):
            yield "Test chunk"

        mocker.patch(
            "ragitect.api.v1.chat.generate_response_stream",
            side_effect=mock_stream,
        )

        mock_llm = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_from_db",
            return_value=mock_llm,
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "Hello"},
        )

        # SSE format: "data: {...}\n\n"
        content = response.text
        assert "data:" in content
        # AI SDK Data Stream Protocol: should have type-based message parts
        assert '"type": "text-delta"' in content or '"type":"text-delta"' in content
        # Should end with finish message instead of [DONE]
        assert '"type": "finish"' in content or '"type":"finish"' in content

    async def test_invalid_workspace_returns_404(self, async_client, mocker):
        """Test 404 for non-existent workspace."""
        workspace_id = uuid.uuid4()

        mock_ws_repo = mocker.AsyncMock()
        mock_ws_repo.get_by_id.return_value = None

        mocker.patch(
            "ragitect.api.v1.chat.WorkspaceRepository",
            return_value=mock_ws_repo,
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "Hello"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_empty_message_returns_422(self, async_client):
        """Test 422 for empty message."""
        workspace_id = uuid.uuid4()

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": ""},
        )

        assert response.status_code == 422

    async def test_missing_message_returns_422(self, async_client):
        """Test 422 when message field is missing."""
        workspace_id = uuid.uuid4()

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={},
        )

        assert response.status_code == 422

    async def test_invalid_workspace_uuid_returns_422(self, async_client):
        """Test 422 for invalid workspace UUID format."""
        response = await async_client.post(
            "/api/v1/workspaces/invalid-uuid/chat/stream",
            json={"message": "Hello"},
        )

        assert response.status_code == 422


class TestChatRAGIntegration:
    """Tests for RAG pipeline integration in chat endpoint.

    Story 3.1: Natural Language Querying - AC2, AC3, AC4, AC6
    """

    async def test_chat_request_accepts_chat_history(self, async_client, mocker):
        """Test that ChatRequest accepts chat_history parameter (AC5)."""
        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        from ragitect.services.database.models import Workspace

        mock_workspace = Workspace(id=workspace_id, name="Test")
        mock_workspace.created_at = now
        mock_workspace.updated_at = now

        mock_ws_repo = mocker.AsyncMock()
        mock_ws_repo.get_by_id.return_value = mock_workspace

        mocker.patch(
            "ragitect.api.v1.chat.WorkspaceRepository",
            return_value=mock_ws_repo,
        )

        # Mock DocumentRepository to return 0 docs (empty workspace)
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_workspace_count.return_value = 0

        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={
                "message": "What is it about?",
                "chat_history": [
                    {"role": "user", "content": "Tell me about Python"},
                    {
                        "role": "assistant",
                        "content": "Python is a programming language.",
                    },
                ],
            },
        )

        # Should not return 422 for chat_history field
        assert response.status_code == 200

    async def test_chat_handles_empty_workspace(self, async_client, mocker):
        """Test graceful handling when no documents exist (AC6)."""
        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        from ragitect.services.database.models import Workspace

        mock_workspace = Workspace(id=workspace_id, name="Test")
        mock_workspace.created_at = now
        mock_workspace.updated_at = now

        mock_ws_repo = mocker.AsyncMock()
        mock_ws_repo.get_by_id.return_value = mock_workspace

        mocker.patch(
            "ragitect.api.v1.chat.WorkspaceRepository",
            return_value=mock_ws_repo,
        )

        # Mock DocumentRepository to return 0 docs
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_workspace_count.return_value = 0

        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "What documents do I have?"},
        )

        assert response.status_code == 200
        content = response.text
        # Should contain helpful message about no documents
        assert "data:" in content
        # The message should mention uploading documents
        assert "upload" in content.lower() or "document" in content.lower()

    async def test_chat_retrieves_relevant_chunks(self, async_client, mocker):
        """Test that chat retrieves chunks from workspace documents (AC2)."""
        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        from ragitect.services.database.models import Workspace

        mock_workspace = Workspace(id=workspace_id, name="Test")
        mock_workspace.created_at = now
        mock_workspace.updated_at = now

        mock_ws_repo = mocker.AsyncMock()
        mock_ws_repo.get_by_id.return_value = mock_workspace

        mocker.patch(
            "ragitect.api.v1.chat.WorkspaceRepository",
            return_value=mock_ws_repo,
        )

        # Mock DocumentRepository to return docs exist
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_workspace_count.return_value = 5

        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        # Track retrieve_context call
        mock_retrieve_context = mocker.AsyncMock()
        mock_retrieve_context.return_value = [
            {
                "content": "Python is a programming language used for web development.",
                "document_name": "python-intro.txt",
                "chunk_index": 0,
            }
        ]

        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context",
            mock_retrieve_context,
        )

        # Mock LLM streaming
        async def mock_stream(llm, messages):
            yield "Python is great!"

        mocker.patch(
            "ragitect.api.v1.chat.generate_response_stream",
            side_effect=mock_stream,
        )

        mock_llm = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_from_db",
            return_value=mock_llm,
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "What is Python?"},
        )

        assert response.status_code == 200
        # Verify retrieve_context was called with correct workspace_id and query
        mock_retrieve_context.assert_called_once()
        call_args = mock_retrieve_context.call_args
        assert (
            call_args.kwargs.get("workspace_id") == workspace_id
            or call_args.args[1] == workspace_id
        )

    async def test_chat_uses_context_in_prompt(self, async_client, mocker):
        """Test that retrieved context is included in LLM prompt (AC3)."""
        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        from ragitect.services.database.models import Workspace

        mock_workspace = Workspace(id=workspace_id, name="Test")
        mock_workspace.created_at = now
        mock_workspace.updated_at = now

        mock_ws_repo = mocker.AsyncMock()
        mock_ws_repo.get_by_id.return_value = mock_workspace

        mocker.patch(
            "ragitect.api.v1.chat.WorkspaceRepository",
            return_value=mock_ws_repo,
        )

        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_workspace_count.return_value = 5

        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        # Mock retrieve_context
        mock_retrieve_context = mocker.AsyncMock()
        mock_retrieve_context.return_value = [
            {
                "content": "FastAPI is a modern Python web framework.",
                "document_name": "fastapi-docs.txt",
                "chunk_index": 0,
            }
        ]

        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context",
            mock_retrieve_context,
        )

        # Capture the messages passed to generate_response_stream
        captured_messages = []

        async def mock_stream(llm, messages):
            captured_messages.extend(messages)
            yield "FastAPI is great!"

        mocker.patch(
            "ragitect.api.v1.chat.generate_response_stream",
            side_effect=mock_stream,
        )

        mock_llm = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_from_db",
            return_value=mock_llm,
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "What is FastAPI?"},
        )

        assert response.status_code == 200
        # Verify context appears in messages
        assert len(captured_messages) >= 1
        # System message should contain the context
        system_content = str(captured_messages[0].content)
        assert "FastAPI" in system_content or "fastapi" in system_content.lower()

    async def test_chat_invalid_chat_history_format_returns_422(self, async_client):
        """Test 422 for invalid chat_history format."""
        workspace_id = uuid.uuid4()

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={
                "message": "Hello",
                "chat_history": "invalid string instead of list",
            },
        )

        assert response.status_code == 422
