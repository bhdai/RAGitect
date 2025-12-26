"""Tests for chat streaming endpoint (SSE) with RAG integration.

Tests verify:
- POST /api/v1/workspaces/{workspace_id}/chat/stream returns SSE content-type
- Endpoint streams data in SSE format (data: ...)
- Workspace validation (404 for invalid workspace)
- Empty message handling
- RAG retrieval from workspace documents
- Context used in LLM prompt
- Empty workspace handling
- Chat history support
- Citation streaming with source-document parts
"""

import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio


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
    """Tests for similarity threshold filtering."""

    def test_retrieve_context_initial_k_is_configurable(self):
        """Test that retrieve_context uses RETRIEVAL_INITIAL_K for over-retrieval (AC1)."""
        import inspect

        from ragitect.api.v1.chat import retrieve_context
        from ragitect.services.config import RETRIEVAL_INITIAL_K

        sig = inspect.signature(retrieve_context)
        initial_k_param = sig.parameters.get("initial_k")
        assert initial_k_param is not None
        assert initial_k_param.default == RETRIEVAL_INITIAL_K

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

    async def test_retrieve_context_uses_initial_k_parameter(self, mocker):
        """Test that retrieve_context passes the initial_k parameter correctly for over-retrieval"""
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

        # Call retrieve_context with default initial_k (uses RETRIEVAL_INITIAL_K)
        await retrieve_context(
            session=mock_session,
            workspace_id=workspace_id,
            query="Test query",
            chat_history=[],
        )

        # Verify search_similar_chunks was called with RETRIEVAL_INITIAL_K (over-retrieval)
        from ragitect.services.config import RETRIEVAL_INITIAL_K

        mock_vector_repo.search_similar_chunks.assert_called_once()
        call_args = mock_vector_repo.search_similar_chunks.call_args
        assert call_args.kwargs.get("k") == RETRIEVAL_INITIAL_K

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
        # Should have chunk label (1-based)
        assert results[0]["chunk_label"] == "Chunk 1"
        assert results[1]["chunk_label"] == "Chunk 2"


class TestPromptEnhancements:
    """Tests for prompt structure and citation format."""

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
        assert "<documents>" in system_message
        assert "</documents>" in system_message
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

    def test_build_rag_prompt_uses_xml_document_format(self, mocker):
        """Test context chunks are formatted as XML documents with index attribute (AC4)."""
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
        # Should use XML format, not [Chunk N]
        assert '<document index="1">' in system_message
        assert '<document index="2">' in system_message
        assert "<source>doc1.txt</source>" in system_message
        assert "<source>doc2.txt</source>" in system_message
        assert "<content>" in system_message
        # Old format should not be present
        assert "[Chunk 1]" not in system_message
        assert "[Chunk 2]" not in system_message

    def test_build_rag_prompt_hides_similarity_scores(self, mocker):
        """Test that similarity scores are hidden from LLM context to prevent bias (AC4)."""
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
        # Similarity should NOT be included (intentionally hidden to prevent LLM bias)
        assert "0.87" not in system_message
        assert "Similarity" not in system_message
        # Should still include document source
        assert "<source>test.txt</source>" in system_message

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
    """Tests for retrieval logging and observability."""

    async def test_retrieval_logs_score_distribution(self, mocker, caplog):
        """Test that retrieval logs similarity score distribution (AC5, AC6)."""
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
            mock_chunk.embedding = [0.1] * 768  # Mock embedding for MMR
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

        # Check that initial retrieval stats were logged (AC6)
        log_text = caplog.text
        assert "Initial retrieval" in log_text
        assert "chunks" in log_text
        assert "similarity" in log_text.lower()


class TestCitationStreaming:
    """Tests for citation detection and streaming."""

    async def test_citation_parser_uses_cite_format(self):
        """Test that CitationStreamParser uses [cite: N] format"""
        from ragitect.api.schemas.chat import Citation
        from ragitect.api.v1.chat import CitationStreamParser

        citations = [
            Citation.from_context_chunk(1, "doc-id-1", "doc1.pdf", 0, 0.9, "Content 1"),
            Citation.from_context_chunk(2, "doc-id-2", "doc2.pdf", 1, 0.8, "Content 2"),
        ]

        parser = CitationStreamParser(citations)

        # Parse chunks with [cite: N] format
        text1, found1 = parser.parse_chunk("Python is great[cite: 1] and ")
        text2, found2 = parser.parse_chunk("versatile[cite: 2].")
        remaining = parser.flush()

        # Should find both citations with new format
        assert len(found1) == 1
        assert found1[0].source_id == "cite-1"
        assert len(found2) == 1
        assert found2[0].source_id == "cite-2"

    async def test_citation_parser_ignores_old_bare_bracket_format(self):
        """Test that parser does NOT match old [N] format"""
        from ragitect.api.schemas.chat import Citation
        from ragitect.api.v1.chat import CitationStreamParser

        citations = [
            Citation.from_context_chunk(1, "doc-id-1", "doc1.pdf", 0, 0.9, "Content 1"),
        ]

        parser = CitationStreamParser(citations)

        # Parse chunks with OLD bare [N] format - should NOT match
        text1, found1 = parser.parse_chunk("Python is great[1] but this won't match.")
        remaining = parser.flush()

        # Should NOT find the citation with old format
        assert len(found1) == 0

    async def test_build_citation_metadata_creates_citations_from_chunks(self):
        """Test that build_citation_metadata creates Citation objects from context chunks (AC1, AC2)."""
        from ragitect.api.v1.chat import build_citation_metadata

        context_chunks = [
            {
                "content": "Python is a programming language used for many applications.",
                "document_name": "python-intro.pdf",
                "chunk_index": 0,
                "similarity": 0.95,
            },
            {
                "content": "FastAPI is a modern web framework for building APIs.",
                "document_name": "fastapi-docs.pdf",
                "chunk_index": 3,
                "rerank_score": 0.88,  # Should use rerank_score over similarity
            },
        ]

        citations = build_citation_metadata(context_chunks)

        assert len(citations) == 2
        assert citations[0].source_id == "cite-1"
        assert citations[0].title == "python-intro.pdf"
        assert citations[1].source_id == "cite-2"
        assert citations[1].title == "fastapi-docs.pdf"

    async def test_citation_stream_parser_detects_markers(self):
        """Test that CitationStreamParser detects [N] markers in text (AC1)."""
        from ragitect.api.schemas.chat import Citation
        from ragitect.api.v1.chat import CitationStreamParser

        citations = [
            Citation.from_context_chunk(1, "doc-id-1", "doc1.pdf", 0, 0.9, "Content 1"),
            Citation.from_context_chunk(2, "doc-id-2", "doc2.pdf", 1, 0.8, "Content 2"),
        ]

        parser = CitationStreamParser(citations)

        # Parse chunks with citation markers using [cite: N] format
        text1, found1 = parser.parse_chunk("Python is great[cite: 1] and ")
        text2, found2 = parser.parse_chunk("versatile[cite: 2].")
        remaining = parser.flush()

        # Should find both citations
        assert len(found1) == 1
        assert found1[0].source_id == "cite-1"
        assert len(found2) == 1
        assert found2[0].source_id == "cite-2"

    async def test_citation_stream_parser_handles_split_markers(self):
        """Test that parser handles citation markers split across chunks (AC1)."""
        from ragitect.api.schemas.chat import Citation
        from ragitect.api.v1.chat import CitationStreamParser

        citations = [
            Citation.from_context_chunk(1, "doc-id", "doc.pdf", 0, 0.9, "Content"),
        ]

        parser = CitationStreamParser(citations)

        # Split "[cite: 1]" across chunks
        text1, found1 = parser.parse_chunk("Hello [cite:")
        text2, found2 = parser.parse_chunk(" 1] world")
        remaining = parser.flush()

        # Should eventually find the citation
        all_citations = found1 + found2
        assert len(all_citations) == 1
        assert all_citations[0].source_id == "cite-1"

    async def test_citation_stream_parser_ignores_invalid_citations(self, caplog):
        """Test that parser logs warning for invalid citation indices (AC6 - hallucination handling)."""
        import logging

        from ragitect.api.schemas.chat import Citation
        from ragitect.api.v1.chat import CitationStreamParser

        citations = [
            Citation.from_context_chunk(0, "doc-id", "doc.pdf", 0, 0.9, "Content"),
        ]

        parser = CitationStreamParser(citations)

        with caplog.at_level(logging.WARNING, logger="ragitect.api.v1.chat"):
            # Try to cite [cite: 99] which doesn't exist
            text, found = parser.parse_chunk("Test [cite: 99] content")
            remaining = parser.flush()

        # Should not find any citations (invalid index)
        assert len(found) == 0
        # Should have logged a warning
        assert "cited non-existent source" in caplog.text or "99" in caplog.text

    async def test_citation_stream_emits_each_citation_once(self):
        """Test that each citation is only emitted once even if marker appears multiple times."""
        from ragitect.api.schemas.chat import Citation
        from ragitect.api.v1.chat import CitationStreamParser

        citations = [
            Citation.from_context_chunk(0, "doc-id", "doc.pdf", 0, 0.9, "Content"),
        ]

        parser = CitationStreamParser(citations)

        # Same citation marker appears twice (using [cite: N] format)
        assert len(citations) >= 1
        # Test with [cite: 1] which maps to index 0
        text1, found1 = parser.parse_chunk("First[cite: 1] and ")
        text2, found2 = parser.parse_chunk("again[cite: 1].")
        remaining = parser.flush()

        # Should only emit once
        total_found = len(found1) + len(found2)
        assert total_found == 1

    async def test_format_sse_stream_with_citations_emits_source_documents(self):
        """Test that format_sse_stream_with_citations emits source-document events (AC1, AC2)."""
        from ragitect.api.schemas.chat import Citation
        from ragitect.api.v1.chat import format_sse_stream_with_citations

        citations = [
            Citation.from_context_chunk(
                1, "doc-uuid", "intro.pdf", 0, 0.95, "Python is..."
            ),
        ]

        async def mock_chunks():
            yield "Python"
            yield " is great"
            yield "[cite: 1]"
            yield "."

        events = []
        async for event in format_sse_stream_with_citations(mock_chunks(), citations):
            events.append(event)

        # Should contain source-document event
        source_doc_events = [e for e in events if "source-document" in e]
        assert len(source_doc_events) >= 1

        # Verify source-document contains expected fields
        source_event = source_doc_events[0]
        # Expect cite-1 because Citation.from_context_chunk will now create cite-1 if we mock correctly or if we pass explicit ID
        # Wait, from_context_chunk uses index arg.
        # If we pass index=1 manually:
        assert "cite-1" in source_event
        assert "intro.pdf" in source_event

    async def test_format_sse_stream_with_citations_handles_zero_citations(
        self, caplog
    ):
        """Test that stream works when LLM doesn't cite any sources (AC6)."""
        import logging

        from ragitect.api.schemas.chat import Citation
        from ragitect.api.v1.chat import format_sse_stream_with_citations

        # Citations available but LLM doesn't use them
        citations = [
            Citation.from_context_chunk(1, "doc-id", "doc.pdf", 0, 0.9, "Content"),
        ]

        async def mock_chunks():
            yield "2 plus 2 equals 4."  # No citations

        with caplog.at_level(logging.INFO, logger="ragitect.api.v1.chat"):
            events = []
            async for event in format_sse_stream_with_citations(
                mock_chunks(), citations
            ):
                events.append(event)

        # Should NOT contain source-document events
        source_doc_events = [e for e in events if "source-document" in e]
        assert len(source_doc_events) == 0

        # Should have text-delta events
        text_events = [e for e in events if "text-delta" in e]
        assert len(text_events) >= 1

        # Should log that no citations were used
        assert "no citations" in caplog.text.lower()

    async def test_chat_endpoint_emits_citation_metadata(self, async_client, mocker):
        """Test that chat endpoint includes source-document events in stream (AC1, AC2)."""
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

        # Mock retrieve_context with context that should be cited
        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context",
            return_value=[
                {
                    "content": "Python is a powerful programming language.",
                    "document_name": "python-intro.pdf",
                    "chunk_index": 0,
                    "similarity": 0.95,
                    "chunk_label": "Chunk 1",
                }
            ],
        )

        # Mock LLM to return response with citation (using [cite: N] format)
        async def mock_stream(llm, messages):
            yield "Python is powerful"
            yield "[cite: 1]"
            yield "."

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
        content = response.text

        # Should contain source-document event
        assert "source-document" in content
        assert "cite-1" in content
        assert "python-intro.pdf" in content


class TestLangGraphRetrieval:
    """Tests for LangGraph-based retrieval pipeline (Story 4.2)."""

    async def test_langgraph_retrieval_flag_uses_graph_based_retrieval(
        self, async_client, mocker
    ):
        """Test that USE_LANGGRAPH_RETRIEVAL=true uses graph-based retrieval."""
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

        # Enable LangGraph retrieval
        mocker.patch("ragitect.api.v1.chat.USE_LANGGRAPH_RETRIEVAL", True)

        # Track which retrieval function is called
        mock_retrieve_context = mocker.AsyncMock(return_value=[])
        mock_retrieve_with_graph = mocker.AsyncMock(
            return_value=[
                {
                    "content": "Graph-based retrieval content",
                    "document_name": "graph-doc.pdf",
                    "chunk_index": 0,
                    "similarity": 0.9,
                    "chunk_label": "Chunk 1",
                }
            ]
        )

        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context",
            mock_retrieve_context,
        )
        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context_with_graph",
            mock_retrieve_with_graph,
        )

        async def mock_stream(llm, messages):
            yield "Test response"

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
            json={"message": "Test query"},
        )

        assert response.status_code == 200

        # Graph-based retrieval should be called
        mock_retrieve_with_graph.assert_called_once()
        # Legacy retrieval should NOT be called
        mock_retrieve_context.assert_not_called()

    async def test_langgraph_retrieval_disabled_uses_legacy_retrieval(
        self, async_client, mocker
    ):
        """Test that USE_LANGGRAPH_RETRIEVAL=false uses legacy retrieval."""
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

        # Disable LangGraph retrieval (default)
        mocker.patch("ragitect.api.v1.chat.USE_LANGGRAPH_RETRIEVAL", False)

        mock_retrieve_context = mocker.AsyncMock(
            return_value=[
                {
                    "content": "Legacy retrieval content",
                    "document_name": "legacy-doc.pdf",
                    "chunk_index": 0,
                    "similarity": 0.9,
                    "chunk_label": "Chunk 1",
                }
            ]
        )
        mock_retrieve_with_graph = mocker.AsyncMock(return_value=[])

        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context",
            mock_retrieve_context,
        )
        mocker.patch(
            "ragitect.api.v1.chat.retrieve_context_with_graph",
            mock_retrieve_with_graph,
        )

        async def mock_stream(llm, messages):
            yield "Test response"

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
            json={"message": "Test query"},
        )

        assert response.status_code == 200

        # Legacy retrieval should be called
        mock_retrieve_context.assert_called_once()
        # Graph-based retrieval should NOT be called
        mock_retrieve_with_graph.assert_not_called()
