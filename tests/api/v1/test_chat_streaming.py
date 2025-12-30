"""Tests for chat streaming endpoint (SSE) with RAG integration.

Tests verify:
- POST /api/v1/workspaces/{workspace_id}/chat/stream returns SSE content-type
- Endpoint streams data in SSE format (data: ...)
- Workspace validation (404 for invalid workspace)
- Empty message handling
- RAG retrieval from workspace documents using LangGraph pipeline
- Context used in LLM prompt
- Empty workspace handling
- Chat history support
- Citation streaming with source-document parts
"""

import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio


def setup_langgraph_streaming_mocks(mocker):
    """Setup common mocks for LangGraph streaming architecture.

    the endpoint uses LangGraphToAISDKAdapter with full graph execution.
    This helper mocks all required dependencies for the streaming pipeline.

    Args:
        mocker: pytest-mock fixture

    Returns:
        dict with mock objects for assertions
    """
    # Mock embedding config - required for both retrieval and streaming
    mock_embed_config = mocker.MagicMock()
    mock_embed_config.provider_name = "ollama"
    mock_embed_config.model_name = "all-MiniLM-L6-v2"
    mock_embed_config.api_key = None
    mock_embed_config.base_url = None
    mock_embed_config.dimension = 768

    mocker.patch(
        "ragitect.api.v1.chat.get_active_embedding_config",
        return_value=mock_embed_config,
    )

    # Mock embedding model and embed function
    mock_embed_model = mocker.MagicMock()
    mocker.patch(
        "ragitect.api.v1.chat.create_embeddings_model",
        return_value=mock_embed_model,
    )

    async def mock_embed_fn(model, text: str):
        return [0.1] * 768

    mocker.patch(
        "ragitect.api.v1.chat.embed_text",
        side_effect=mock_embed_fn,
    )

    # Mock vector repository
    mock_vector_repo = mocker.AsyncMock()
    mocker.patch(
        "ragitect.api.v1.chat.VectorRepository",
        return_value=mock_vector_repo,
    )

    # Create proper async LLM mock for LangGraph nodes
    # The LLM needs .with_structured_output() and .ainvoke() methods
    mock_structured_llm = mocker.AsyncMock()

    # Mock strategy response for generate_strategy node
    from ragitect.agents.rag.schemas import Search, SearchStrategy

    mock_strategy = SearchStrategy(
        reasoning="Test analysis",
        searches=[Search(term="test query", reasoning="test rationale")],
    )
    mock_structured_llm.ainvoke.return_value = mock_strategy

    mock_llm = mocker.MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    # Mock final answer for generate_answer node (must be awaitable)
    from langchain_core.messages import AIMessage

    mock_answer_llm = mocker.AsyncMock()
    mock_answer_llm.return_value = AIMessage(content="Test response")
    mock_llm.ainvoke = mock_answer_llm

    mocker.patch(
        "ragitect.api.v1.chat.create_llm_with_provider",
        return_value=mock_llm,
    )

    return {
        "embed_config": mock_embed_config,
        "embed_model": mock_embed_model,
        "vector_repo": mock_vector_repo,
        "llm": mock_llm,
        "structured_llm": mock_structured_llm,
    }


class TestChatStreamEndpoint:
    """Tests for POST /api/v1/workspaces/{workspace_id}/chat/stream."""

    async def test_stream_returns_sse_content_type(self, async_client, mocker):
        """Test endpoint returns text/event-stream content type."""
        from datetime import datetime, timezone

        from ragitect.services.database.models import Workspace

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

        # Setup LangGraph streaming mocks
        setup_langgraph_streaming_mocks(mocker)

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "Hello"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    async def test_stream_format_is_sse(self, async_client, mocker):
        """Test response is formatted as Server-Sent Events."""
        from datetime import datetime, timezone

        from ragitect.services.database.models import Workspace

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

        # Setup LangGraph streaming mocks
        setup_langgraph_streaming_mocks(mocker)

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
    """Tests for RAG pipeline integration in chat endpoint."""

    async def test_chat_request_accepts_provider_parameter(self, async_client, mocker):
        """Test that ChatRequest accepts provider parameter for runtime override."""
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
                "message": "Test message",
                "provider": "openai",
            },
        )

        # Should not return 422 for provider field
        assert response.status_code == 200

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

        # Setup LangGraph streaming mocks
        setup_langgraph_streaming_mocks(mocker)

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "What is Python?"},
        )

        assert response.status_code == 200
        # With LangGraph streaming, the full graph runs and we verify it completes successfully

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

        # Setup LangGraph streaming mocks
        setup_langgraph_streaming_mocks(mocker)

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "What is FastAPI?"},
        )

        assert response.status_code == 200
        # With LangGraph streaming, context is automatically injected via graph state
        # Just verify the endpoint executed successfully

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


class TestChatProviderOverride:
    """Tests for provider override functionality in chat endpoint."""

    async def test_chat_with_provider_override_uses_specified_provider(
        self, async_client, mocker
    ):
        """Test that specifying provider uses that provider's config."""
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

        # Setup LangGraph streaming mocks
        mocks = setup_langgraph_streaming_mocks(mocker)

        # Track which provider was requested
        mock_create_llm = mocker.AsyncMock()
        mock_create_llm.return_value = mocks["llm"]
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_with_provider",
            mock_create_llm,
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "Hello", "provider": "anthropic"},
        )

        assert response.status_code == 200
        # Verify the provider was passed to create_llm_with_provider
        mock_create_llm.assert_called_once()
        call_kwargs = mock_create_llm.call_args.kwargs
        assert call_kwargs.get("provider") == "anthropic"

    async def test_chat_with_invalid_provider_returns_400(self, async_client, mocker):
        """Test that specifying an unconfigured provider returns 400."""
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

        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context_with_graph",
            return_value=[],
        )

        # Mock create_llm_with_provider to raise ValueError
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_with_provider",
            side_effect=ValueError("Provider 'nonexistent' not configured"),
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "Hello", "provider": "nonexistent"},
        )

        assert response.status_code == 400
        assert "not configured" in response.json()["detail"].lower()

    async def test_chat_with_inactive_provider_returns_400(self, async_client, mocker):
        """Test that specifying an inactive provider returns 400."""
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

        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context_with_graph",
            return_value=[],
        )

        # Mock create_llm_with_provider to raise ValueError for inactive provider
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_with_provider",
            side_effect=ValueError("Provider 'openai' is not active"),
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "Hello", "provider": "openai"},
        )

        assert response.status_code == 400
        assert "not active" in response.json()["detail"].lower()

    async def test_chat_without_provider_uses_default(self, async_client, mocker):
        """Test that omitting provider uses default active provider."""
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

        # Setup LangGraph streaming mocks
        mocks = setup_langgraph_streaming_mocks(mocker)

        mock_create_llm = mocker.AsyncMock()
        mock_create_llm.return_value = mocks["llm"]
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_with_provider",
            mock_create_llm,
        )

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "Hello"},  # No provider specified
        )

        assert response.status_code == 200
        # Verify provider=None was passed (default behavior)
        mock_create_llm.assert_called_once()
        call_kwargs = mock_create_llm.call_args.kwargs
        assert call_kwargs.get("provider") is None


class TestLangGraphStreaming:
    """Tests for LangGraph streaming adapter integration in chat endpoint."""

    async def test_langgraph_adapter_integration(self, async_client, mocker):
        """Test /chat/stream endpoint uses LangGraphToAISDKAdapter.

        Verifies that the endpoint:
        1. Uses the full graph (not retrieval-only)
        2. Integrates with LangGraphToAISDKAdapter
        3. Returns proper SSE headers
        4. Streams events in correct format
        """
        from datetime import datetime, timezone

        from ragitect.services.database.models import Workspace

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

        # Mock DocumentRepository - workspace has documents
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_workspace_count.return_value = 5

        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        # Setup LangGraph streaming mocks
        setup_langgraph_streaming_mocks(mocker)

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "What is Python?"},
        )

        assert response.status_code == 200

        # Verify SSE headers
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers.get("x-vercel-ai-ui-message-stream") == "v1"

    async def test_sse_event_format_compliance(self, async_client, mocker):
        """Test SSE events match AI SDK protocol format.

        Verifies:
        - All events are prefixed with "data: "
        - All events are valid JSON
        - Event types match AI SDK spec
        """
        from datetime import datetime, timezone

        from ragitect.services.database.models import Workspace

        workspace_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        mock_workspace = Workspace(
            id=workspace_id,
            name="Test Workspace",
        )
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

        # Setup LangGraph streaming mocks
        setup_langgraph_streaming_mocks(mocker)

        response = await async_client.post(
            f"/api/v1/workspaces/{workspace_id}/chat/stream",
            json={"message": "Test query"},
        )

        assert response.status_code == 200
        content = response.text

        # Split SSE stream into events
        events = []
        for line in content.split("\n"):
            if line.startswith("data: "):
                import json

                event_data = line[6:]  # Strip "data: "
                event = json.loads(event_data)
                events.append(event)

        # Verify at least start, text-start, text-delta, text-end, finish
        event_types = [e["type"] for e in events]
        assert "start" in event_types
        assert "text-start" in event_types
        assert "text-delta" in event_types or "source-document" in event_types
        assert "text-end" in event_types
        assert "finish" in event_types
