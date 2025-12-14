"""Tests for chat streaming endpoint (SSE).

Tests verify:
- POST /api/v1/workspaces/{workspace_id}/chat/stream returns SSE content-type
- Endpoint streams data in SSE format (data: ...)
- Workspace validation (404 for invalid workspace)
- Empty message handling

Story 3.0: Streaming Infrastructure (Prep)
"""

import uuid

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
        # Should contain [DONE] at the end
        assert "[DONE]" in content

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
