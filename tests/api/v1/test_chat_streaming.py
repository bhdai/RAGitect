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
                {
                    "content": "test",
                    "document_name": "test.txt",
                    "chunk_index": 0,
                    "similarity": 0.85,
                    "chunk_label": "Chunk 1",
                }
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

        # Mock create_llm_with_provider
        mock_llm = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_with_provider",
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
                {
                    "content": "test",
                    "document_name": "test.txt",
                    "chunk_index": 0,
                    "similarity": 0.85,
                    "chunk_label": "Chunk 1",
                }
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
            "ragitect.api.v1.chat.create_llm_with_provider",
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

        # Track retrieve_context call
        mock_retrieve_context = mocker.AsyncMock()
        mock_retrieve_context.return_value = [
            {
                "content": "Python is a programming language used for web development.",
                "document_name": "python-intro.txt",
                "chunk_index": 0,
                "similarity": 0.85,
                "chunk_label": "Chunk 1",
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
            "ragitect.api.v1.chat.create_llm_with_provider",
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
                "similarity": 0.9,
                "chunk_label": "Chunk 1",
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
            "ragitect.api.v1.chat.create_llm_with_provider",
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

        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context",
            return_value=[
                {
                    "content": "test",
                    "document_name": "test.txt",
                    "chunk_index": 0,
                    "similarity": 0.85,
                    "chunk_label": "Chunk 1",
                }
            ],
        )

        async def mock_stream(llm, messages):
            yield "Test"

        mocker.patch(
            "ragitect.api.v1.chat.generate_response_stream",
            side_effect=mock_stream,
        )

        # Track which provider was requested
        mock_create_llm = mocker.AsyncMock()
        mock_create_llm.return_value = mocker.MagicMock()
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
            "ragitect.api.v1.chat.retrieve_context",
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
            "ragitect.api.v1.chat.retrieve_context",
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

        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context",
            return_value=[
                {
                    "content": "test",
                    "document_name": "test.txt",
                    "chunk_index": 0,
                    "similarity": 0.85,
                    "chunk_label": "Chunk 1",
                }
            ],
        )

        async def mock_stream(llm, messages):
            yield "Test"

        mocker.patch(
            "ragitect.api.v1.chat.generate_response_stream",
            side_effect=mock_stream,
        )

        mock_create_llm = mocker.AsyncMock()
        mock_create_llm.return_value = mocker.MagicMock()
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


class TestRetrievalThresholdFiltering:
    """Tests for similarity threshold filtering.

    Story 3.1.1: Retrieval Tuning & Prompt Enhancement - AC1, AC2
    """

    def test_retrieve_context_default_k_is_configurable(self):
        """Test that retrieve_context uses DEFAULT_RETRIEVAL_K from config (AC2)."""
        import inspect
        from ragitect.api.v1.chat import retrieve_context
        from ragitect.services.config import DEFAULT_RETRIEVAL_K

        sig = inspect.signature(retrieve_context)
        k_param = sig.parameters.get("k")
        assert k_param is not None
        assert k_param.default == DEFAULT_RETRIEVAL_K

    async def test_retrieve_context_passes_similarity_threshold(self, mocker):
        """Test that retrieve_context passes similarity_threshold=0.3 to vector search (AC1)."""
        from ragitect.api.v1.chat import retrieve_context

        workspace_id = uuid.uuid4()
        mock_session = mocker.AsyncMock()

        # Mock LLM
        mock_llm = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_with_provider",
            return_value=mock_llm,
        )

        # Mock embedding config
        mocker.patch(
            "ragitect.api.v1.chat.get_active_embedding_config",
            return_value=None,
        )

        # Mock embedding model
        mock_embed_model = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_embeddings_model",
            return_value=mock_embed_model,
        )
        mocker.patch(
            "ragitect.api.v1.chat.embed_text",
            return_value=[0.1] * 768,
        )

        # Mock VectorRepository to capture arguments
        from ragitect.services.database.models import DocumentChunk

        mock_chunk = mocker.MagicMock(spec=DocumentChunk)
        mock_chunk.content = "Test chunk content"
        mock_chunk.document_id = uuid.uuid4()
        mock_chunk.chunk_index = 0

        mock_vector_repo = mocker.AsyncMock()
        mock_vector_repo.search_similar_chunks.return_value = [(mock_chunk, 0.5)]

        mocker.patch(
            "ragitect.api.v1.chat.VectorRepository",
            return_value=mock_vector_repo,
        )

        # Mock DocumentRepository
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_id.return_value = None
        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        # Mock query_with_iterative_fallback to call the vector_search_fn callback
        async def mock_iterative_fallback(llm, query, chat_history, vector_search_fn):
            # Call the vector search function to trigger the actual search
            await vector_search_fn(query)
            return (
                ["Test content"],
                {
                    "final_query": query,
                    "classification": "simple",
                    "used_reformulation": False,
                },
            )

        mocker.patch(
            "ragitect.api.v1.chat.query_with_iterative_fallback",
            side_effect=mock_iterative_fallback,
        )

        # Call retrieve_context
        await retrieve_context(
            session=mock_session,
            workspace_id=workspace_id,
            query="Test query",
            chat_history=[],
        )

        # Verify search_similar_chunks was called with similarity_threshold=0.3
        mock_vector_repo.search_similar_chunks.assert_called_once()
        call_args = mock_vector_repo.search_similar_chunks.call_args
        assert call_args.kwargs.get("similarity_threshold") == 0.3

    async def test_retrieve_context_uses_k_parameter(self, mocker):
        """Test that retrieve_context passes the k parameter correctly (AC2)."""
        from ragitect.api.v1.chat import retrieve_context

        workspace_id = uuid.uuid4()
        mock_session = mocker.AsyncMock()

        # Mock LLM
        mock_llm = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_with_provider",
            return_value=mock_llm,
        )

        # Mock embedding config
        mocker.patch(
            "ragitect.api.v1.chat.get_active_embedding_config",
            return_value=None,
        )

        # Mock embedding model
        mock_embed_model = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_embeddings_model",
            return_value=mock_embed_model,
        )
        mocker.patch(
            "ragitect.api.v1.chat.embed_text",
            return_value=[0.1] * 768,
        )

        # Mock VectorRepository to capture arguments
        from ragitect.services.database.models import DocumentChunk

        mock_chunk = mocker.MagicMock(spec=DocumentChunk)
        mock_chunk.content = "Test chunk content"
        mock_chunk.document_id = uuid.uuid4()
        mock_chunk.chunk_index = 0

        mock_vector_repo = mocker.AsyncMock()
        mock_vector_repo.search_similar_chunks.return_value = [(mock_chunk, 0.5)]

        mocker.patch(
            "ragitect.api.v1.chat.VectorRepository",
            return_value=mock_vector_repo,
        )

        # Mock DocumentRepository
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_id.return_value = None
        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        # Mock query_with_iterative_fallback to call the vector_search_fn callback
        async def mock_iterative_fallback(llm, query, chat_history, vector_search_fn):
            await vector_search_fn(query)
            return (
                ["Test content"],
                {
                    "final_query": query,
                    "classification": "simple",
                    "used_reformulation": False,
                },
            )

        mocker.patch(
            "ragitect.api.v1.chat.query_with_iterative_fallback",
            side_effect=mock_iterative_fallback,
        )

        # Call retrieve_context with default k (uses DEFAULT_RETRIEVAL_K)
        await retrieve_context(
            session=mock_session,
            workspace_id=workspace_id,
            query="Test query",
            chat_history=[],
        )

        # Verify search_similar_chunks was called with DEFAULT_RETRIEVAL_K
        from ragitect.services.config import DEFAULT_RETRIEVAL_K

        mock_vector_repo.search_similar_chunks.assert_called_once()
        call_args = mock_vector_repo.search_similar_chunks.call_args
        assert call_args.kwargs.get("k") == DEFAULT_RETRIEVAL_K

    async def test_retrieve_context_includes_chunk_label(self, mocker):
        """Test that retrieve_context adds chunk_label to results (AC4)."""
        from ragitect.api.v1.chat import retrieve_context

        workspace_id = uuid.uuid4()
        mock_session = mocker.AsyncMock()

        # Mock LLM
        mock_llm = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_with_provider",
            return_value=mock_llm,
        )

        # Mock embedding config
        mocker.patch(
            "ragitect.api.v1.chat.get_active_embedding_config",
            return_value=None,
        )

        # Mock embedding model
        mocker.patch(
            "ragitect.api.v1.chat.create_embeddings_model",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "ragitect.api.v1.chat.embed_text",
            return_value=[0.1] * 768,
        )

        # Mock VectorRepository
        from ragitect.services.database.models import DocumentChunk

        mock_chunk1 = mocker.MagicMock(spec=DocumentChunk)
        mock_chunk1.content = "First chunk"
        mock_chunk1.document_id = uuid.uuid4()
        mock_chunk1.chunk_index = 0

        mock_chunk2 = mocker.MagicMock(spec=DocumentChunk)
        mock_chunk2.content = "Second chunk"
        mock_chunk2.document_id = uuid.uuid4()
        mock_chunk2.chunk_index = 1

        mock_vector_repo = mocker.AsyncMock()
        mock_vector_repo.search_similar_chunks.return_value = [
            (mock_chunk1, 0.3),
            (mock_chunk2, 0.4),
        ]

        mocker.patch(
            "ragitect.api.v1.chat.VectorRepository",
            return_value=mock_vector_repo,
        )

        # Mock DocumentRepository
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_id.return_value = None
        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        # Mock query_with_iterative_fallback to call the vector_search_fn callback
        async def mock_iterative_fallback(llm, query, chat_history, vector_search_fn):
            await vector_search_fn(query)
            return (
                ["First chunk", "Second chunk"],
                {
                    "final_query": query,
                    "classification": "simple",
                    "used_reformulation": False,
                },
            )

        mocker.patch(
            "ragitect.api.v1.chat.query_with_iterative_fallback",
            side_effect=mock_iterative_fallback,
        )

        results = await retrieve_context(
            session=mock_session,
            workspace_id=workspace_id,
            query="Test query",
            chat_history=[],
        )

        # Verify chunk_label is included in results
        assert len(results) == 2
        assert results[0]["chunk_label"] == "Chunk 1"
        assert results[1]["chunk_label"] == "Chunk 2"


class TestPromptEnhancements:
    """Tests for prompt structure and citation format.

    Story 3.1.1: Retrieval Tuning & Prompt Enhancement - AC3, AC4
    """

    def test_build_rag_prompt_uses_xml_structure(self, mocker):
        """Test prompt uses XML structure for context isolation (AC3)."""
        from ragitect.api.v1.chat import build_rag_prompt

        context_chunks = [
            {
                "content": "Python is a programming language.",
                "document_name": "python.txt",
                "similarity": 0.85,
                "chunk_label": "Chunk 1",
            }
        ]

        messages = build_rag_prompt("What is Python?", context_chunks, [])

        system_message = messages[0].content
        assert "<system_instructions>" in system_message
        assert "</system_instructions>" in system_message
        assert "<context>" in system_message
        assert "</context>" in system_message
        # Note: user_query is intentionally NOT in system prompt per review feedback
        # Query comes only as final HumanMessage to respect semantic causal flow
        assert "<user_query>" not in system_message

    def test_build_rag_prompt_includes_negative_constraints(self, mocker):
        """Test prompt includes negative constraints to prevent hallucination (AC3)."""
        from ragitect.api.v1.chat import build_rag_prompt

        context_chunks = [
            {
                "content": "FastAPI docs.",
                "document_name": "fastapi.txt",
                "similarity": 0.9,
                "chunk_label": "Chunk 1",
            }
        ]

        messages = build_rag_prompt("What is FastAPI?", context_chunks, [])

        system_message = messages[0].content
        # Should contain negative constraints
        assert "DO NOT" in system_message or "do not" in system_message.lower()
        assert (
            "fabricate" in system_message.lower()
            or "outside knowledge" in system_message.lower()
        )

    def test_build_rag_prompt_includes_refusal_protocol(self, mocker):
        """Test prompt specifies refusal protocol for out-of-context questions (AC3)."""
        from ragitect.api.v1.chat import build_rag_prompt

        context_chunks = [
            {
                "content": "Some content",
                "document_name": "doc.txt",
                "similarity": 0.8,
                "chunk_label": "Chunk 1",
            }
        ]

        messages = build_rag_prompt("Random question", context_chunks, [])

        system_message = messages[0].content
        # Should contain refusal instruction
        assert (
            "cannot find" in system_message.lower() or "I cannot find" in system_message
        )

    def test_build_rag_prompt_uses_chunk_labels(self, mocker):
        """Test context chunks are labeled as [Chunk 1], [Chunk 2], etc. (AC4)."""
        from ragitect.api.v1.chat import build_rag_prompt

        context_chunks = [
            {
                "content": "First chunk content",
                "document_name": "doc1.txt",
                "similarity": 0.9,
                "chunk_label": "Chunk 1",
            },
            {
                "content": "Second chunk content",
                "document_name": "doc2.txt",
                "similarity": 0.85,
                "chunk_label": "Chunk 2",
            },
        ]

        messages = build_rag_prompt("Query", context_chunks, [])

        system_message = messages[0].content
        assert "[Chunk 1]" in system_message
        assert "[Chunk 2]" in system_message

    def test_build_rag_prompt_includes_similarity_scores(self, mocker):
        """Test each chunk includes similarity score for transparency (AC4)."""
        from ragitect.api.v1.chat import build_rag_prompt

        context_chunks = [
            {
                "content": "Content here",
                "document_name": "test.txt",
                "similarity": 0.87,
                "chunk_label": "Chunk 1",
            }
        ]

        messages = build_rag_prompt("Query", context_chunks, [])

        system_message = messages[0].content
        # Should include similarity in format like "Similarity: 0.87"
        assert "0.87" in system_message or "Similarity" in system_message

    def test_build_rag_prompt_includes_citation_rules(self, mocker):
        """Test prompt includes citation rules for [N] format (AC4)."""
        from ragitect.api.v1.chat import build_rag_prompt

        context_chunks = [
            {
                "content": "Test content",
                "document_name": "doc.txt",
                "similarity": 0.8,
                "chunk_label": "Chunk 1",
            }
        ]

        messages = build_rag_prompt("Test query", context_chunks, [])

        system_message = messages[0].content
        # Should contain citation instruction
        assert "[N]" in system_message or "cite" in system_message.lower()


class TestRetrievalLogging:
    """Tests for retrieval logging and observability.

    Story 3.1.1: Retrieval Tuning & Prompt Enhancement - AC5
    """

    async def test_retrieval_logs_score_distribution(self, mocker, caplog):
        """Test that retrieval logs similarity score distribution (AC5)."""
        import logging
        from ragitect.api.v1.chat import retrieve_context

        workspace_id = uuid.uuid4()
        mock_session = mocker.AsyncMock()

        # Mock LLM
        mock_llm = mocker.MagicMock()
        mocker.patch(
            "ragitect.api.v1.chat.create_llm_with_provider",
            return_value=mock_llm,
        )

        # Mock embedding config
        mocker.patch(
            "ragitect.api.v1.chat.get_active_embedding_config",
            return_value=None,
        )

        # Mock embedding model
        mocker.patch(
            "ragitect.api.v1.chat.create_embeddings_model",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "ragitect.api.v1.chat.embed_text",
            return_value=[0.1] * 768,
        )

        # Create multiple mock chunks with different distances
        from ragitect.services.database.models import DocumentChunk

        mock_chunks = []
        for i, distance in enumerate([0.2, 0.4, 0.6]):
            mock_chunk = mocker.MagicMock(spec=DocumentChunk)
            mock_chunk.content = f"Chunk {i} content"
            mock_chunk.document_id = uuid.uuid4()
            mock_chunk.chunk_index = i
            mock_chunks.append((mock_chunk, distance))

        mock_vector_repo = mocker.AsyncMock()
        mock_vector_repo.search_similar_chunks.return_value = mock_chunks

        mocker.patch(
            "ragitect.api.v1.chat.VectorRepository",
            return_value=mock_vector_repo,
        )

        # Mock DocumentRepository
        mock_doc_repo = mocker.AsyncMock()
        mock_doc_repo.get_by_id.return_value = None
        mocker.patch(
            "ragitect.api.v1.chat.DocumentRepository",
            return_value=mock_doc_repo,
        )

        # Mock query_with_iterative_fallback to call the vector_search_fn callback
        async def mock_iterative_fallback(llm, query, chat_history, vector_search_fn):
            await vector_search_fn(query)
            return (
                ["Chunk 0 content"],
                {
                    "final_query": query,
                    "classification": "simple",
                    "used_reformulation": False,
                },
            )

        mocker.patch(
            "ragitect.api.v1.chat.query_with_iterative_fallback",
            side_effect=mock_iterative_fallback,
        )

        with caplog.at_level(logging.INFO, logger="ragitect.api.v1.chat"):
            await retrieve_context(
                session=mock_session,
                workspace_id=workspace_id,
                query="Test query",
                chat_history=[],
            )

        # Check that retrieval stats were logged
        log_text = caplog.text
        assert "Retrieval stats" in log_text
        assert "chunks" in log_text
        assert "similarity" in log_text.lower()
