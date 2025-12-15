"""Unit tests for LLM service functions.

Comprehensive tests for the LLM service rewrite (Story 3.0).
Covers:
- DRY helper function `_build_litellm_kwargs()`
- Structured logging with `_log_llm_metrics()`
- All public API functions
- Error handling in streaming operations
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from ragitect.services.config import LLMConfig
from ragitect.services.database.models import LLMProviderConfig
from ragitect.services.llm import (
    _build_litellm_kwargs,
    create_llm,
    create_llm_from_db,
    generate_response,
    generate_response_stream,
    generate_response_with_prompt,
    get_active_llm_config,
    validate_llm_config,
)

pytestmark = [pytest.mark.asyncio]


class TestBuildLiteLLMKwargs:
    """Tests for _build_litellm_kwargs helper (DRY extraction)."""

    def test_builds_kwargs_for_ollama(self):
        """Test kwargs building for Ollama provider."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.1:8b",
            base_url="http://localhost:11434",
            temperature=0.7,
        )

        result = _build_litellm_kwargs(config)

        assert result["model"] == "ollama/llama3.1:8b"
        assert result["temperature"] == 0.7
        assert result["api_base"] == "http://localhost:11434"
        assert result["api_key"] is None

    def test_builds_kwargs_for_openai(self):
        """Test kwargs building for OpenAI provider."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test123",
            temperature=0.5,
            max_tokens=1000,
        )

        result = _build_litellm_kwargs(config)

        assert result["model"] == "openai/gpt-4o"
        assert result["api_key"] == "sk-test123"
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 1000

    def test_builds_kwargs_with_custom_provider(self):
        """Test kwargs building with custom provider."""
        config = LLMConfig(
            provider="openai",
            model="my-custom-model",
            custom_provider="custom-provider",
            api_key="test-key",
        )

        result = _build_litellm_kwargs(config)

        # Custom provider should override the provider field
        assert result["model"] == "custom-provider/my-custom-model"

    def test_builds_kwargs_with_timeout(self):
        """Test kwargs includes request_timeout."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            timeout=120,
        )

        result = _build_litellm_kwargs(config)

        assert result["request_timeout"] == 120

    def test_omits_api_base_when_not_set(self):
        """Test api_base is not included when base_url is None."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            base_url=None,
        )

        result = _build_litellm_kwargs(config)

        assert "api_base" not in result


class TestCreateLLM:
    """Tests for create_llm function."""

    async def test_creates_chat_model(self):
        """Test LLM model creation returns ChatLiteLLM instance."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.1:8b",
            base_url="http://localhost:11434",
        )

        with patch("ragitect.services.llm.ChatLiteLLM") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            result = await create_llm(config)

            assert result == mock_instance
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "ollama/llama3.1:8b"


class TestValidateLLMConfig:
    """Tests for validate_llm_config function."""

    async def test_successful_validation(self):
        """Test successful LLM config validation."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.1:8b",
        )

        with patch("ragitect.services.llm.ChatLiteLLM") as mock_chat:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Hello!"
            mock_llm.ainvoke.return_value = mock_response
            mock_chat.return_value = mock_llm

            is_valid, error = await validate_llm_config(config)

            assert is_valid is True
            assert error == ""

    async def test_validation_fails_on_empty_response(self):
        """Test validation fails when LLM returns empty response."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.1:8b",
        )

        with patch("ragitect.services.llm.ChatLiteLLM") as mock_chat:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = ""
            mock_llm.ainvoke.return_value = mock_response
            mock_chat.return_value = mock_llm

            is_valid, error = await validate_llm_config(config)

            assert is_valid is False
            assert "empty response" in error.lower()

    async def test_validation_fails_on_exception(self):
        """Test validation catches and reports exceptions."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="invalid-key",
        )

        with patch("ragitect.services.llm.ChatLiteLLM") as mock_chat:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = Exception("API error")
            mock_chat.return_value = mock_llm

            is_valid, error = await validate_llm_config(config)

            assert is_valid is False
            assert "API error" in error


class TestGenerateResponse:
    """Tests for generate_response function."""

    async def test_generates_response_from_messages(self):
        """Test response generation with message list."""
        from langchain_core.messages import HumanMessage

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Generated response"
        mock_llm.ainvoke.return_value = mock_response

        messages = [HumanMessage(content="Hello")]
        result = await generate_response(mock_llm, messages)

        assert result == "Generated response"
        mock_llm.ainvoke.assert_awaited_once_with(messages)

    async def test_handles_empty_content(self):
        """Test handling of empty content response."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = None
        mock_llm.ainvoke.return_value = mock_response

        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Hello")]

        result = await generate_response(mock_llm, messages)

        assert result == ""

    async def test_raises_on_llm_error(self):
        """Test exception propagation from LLM."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("LLM error")

        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Hello")]

        with pytest.raises(Exception, match="LLM error"):
            await generate_response(mock_llm, messages)


class TestGenerateResponseStream:
    """Tests for streaming with proper error handling."""

    async def test_stream_yields_chunks(self):
        """Test successful streaming yields content chunks."""
        mock_llm = MagicMock()

        async def mock_astream(messages):
            for text in ["Hello", " ", "World"]:
                chunk = MagicMock()
                chunk.content = text
                yield chunk

        mock_llm.astream = mock_astream

        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Hi")]

        chunks = []
        async for chunk in generate_response_stream(mock_llm, messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "World"]

    async def test_stream_filters_empty_chunks(self):
        """Test streaming filters out empty content."""
        mock_llm = MagicMock()

        async def mock_astream(messages):
            for text in ["Hello", "", None, "World"]:
                chunk = MagicMock()
                chunk.content = text
                yield chunk

        mock_llm.astream = mock_astream

        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Hi")]

        chunks = []
        async for chunk in generate_response_stream(mock_llm, messages):
            chunks.append(chunk)

        # Empty and None content should be filtered
        assert chunks == ["Hello", "World"]

    async def test_stream_handles_llm_error_gracefully(self):
        """Test error handling during streaming."""
        mock_llm = MagicMock()

        async def mock_astream_with_error(messages):
            yield MagicMock(content="First chunk")
            raise Exception("Stream error")

        mock_llm.astream = mock_astream_with_error

        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Hi")]

        chunks = []
        with pytest.raises(Exception, match="Stream error"):
            async for chunk in generate_response_stream(mock_llm, messages):
                chunks.append(chunk)

        # Should have yielded first chunk before error
        assert chunks == ["First chunk"]

    async def test_stream_logs_metrics(self):
        """Test structured logging for streaming operations."""
        mock_llm = MagicMock()

        async def mock_astream(messages):
            for text in ["a", "b"]:
                chunk = MagicMock()
                chunk.content = text
                yield chunk

        mock_llm.astream = mock_astream

        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Hi")]

        with patch("ragitect.services.llm.logger") as mock_logger:
            async for _ in generate_response_stream(mock_llm, messages):
                pass

            # Verify logging was called
            assert mock_logger.info.called


class TestGenerateResponseWithPrompt:
    """Tests for generate_response_with_prompt convenience function."""

    async def test_converts_prompt_to_message(self):
        """Test prompt is converted to HumanMessage."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_llm.ainvoke.return_value = mock_response

        result = await generate_response_with_prompt(mock_llm, "Hello")

        assert result == "Response"
        # Verify HumanMessage was created
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].content == "Hello"


@pytest.mark.asyncio
class TestGetActiveLLMConfig:
    """Test suite for get_active_llm_config function."""

    async def test_returns_config_from_database(self):
        """Test loading active config from database."""
        # Arrange
        mock_session = AsyncMock()
        active_config = LLMProviderConfig(
            id=uuid.uuid4(),
            provider_name="openai",
            config_data={
                "api_key": "sk-test123",
                "model": "gpt-4o",
                "temperature": 0.5,
            },
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        with patch(
            "ragitect.services.llm_config_service.get_active_config"
        ) as mock_get_active:
            mock_get_active.return_value = active_config

            # Act
            result = await get_active_llm_config(mock_session)

            # Assert
            assert result.provider == "openai"
            assert result.api_key == "sk-test123"
            assert result.model == "gpt-4o"
            assert result.temperature == 0.5
            mock_get_active.assert_awaited_once_with(mock_session)

    async def test_returns_default_when_no_active_config(self):
        """Test fallback to default Ollama config when no active config."""
        mock_session = AsyncMock()

        with patch(
            "ragitect.services.llm_config_service.get_active_config"
        ) as mock_get_active:
            mock_get_active.return_value = None

            # Act
            result = await get_active_llm_config(mock_session)

            # Assert
            assert result.provider == "ollama"
            assert result.model == "llama3.1:8b"
            assert result.base_url == "http://localhost:11434"
            assert result.api_key is None

    async def test_returns_default_on_exception(self):
        """Test fallback to default config on database error."""
        mock_session = AsyncMock()

        with patch(
            "ragitect.services.llm_config_service.get_active_config"
        ) as mock_get_active:
            mock_get_active.side_effect = Exception("Database error")

            # Act
            result = await get_active_llm_config(mock_session)

            # Assert - should return default config
            assert result.provider == "ollama"
            assert result.model == "llama3.1:8b"

    async def test_uses_default_values_for_missing_fields(self):
        """Test that missing config fields get default values."""
        mock_session = AsyncMock()
        active_config = LLMProviderConfig(
            id=uuid.uuid4(),
            provider_name="anthropic",
            config_data={"api_key": "sk-ant-test"},  # minimal config
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        with patch(
            "ragitect.services.llm_config_service.get_active_config"
        ) as mock_get_active:
            mock_get_active.return_value = active_config

            # Act
            result = await get_active_llm_config(mock_session)

            # Assert - check defaults are applied
            assert result.provider == "anthropic"
            assert result.model == "llama3.1:8b"  # default
            assert result.temperature == 0.7  # default
            assert result.timeout == 60  # default


@pytest.mark.asyncio
class TestCreateLLMFromDB:
    """Test suite for create_llm_from_db function."""

    async def test_creates_llm_from_active_config(self):
        """Test creating LLM model from database config."""
        mock_session = AsyncMock()
        test_config = LLMConfig(
            provider="ollama",
            model="llama3.2",
            base_url="http://localhost:11434",
        )

        with (
            patch("ragitect.services.llm.get_active_llm_config") as mock_get_config,
            patch("ragitect.services.llm.create_llm") as mock_create,
        ):
            mock_get_config.return_value = test_config
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            # Act
            result = await create_llm_from_db(mock_session)

            # Assert
            mock_get_config.assert_awaited_once_with(mock_session)
            mock_create.assert_awaited_once_with(test_config)
            assert result == mock_llm

    async def test_creates_llm_with_default_on_no_config(self):
        """Test creating LLM with default config when none exists."""
        mock_session = AsyncMock()

        with (
            patch("ragitect.services.llm.get_active_llm_config") as mock_get_config,
            patch("ragitect.services.llm.create_llm") as mock_create,
        ):
            # Return default config
            mock_get_config.return_value = LLMConfig()
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            # Act
            result = await create_llm_from_db(mock_session)

            # Assert
            assert result == mock_llm
            # Verify default config was used
            config_arg = mock_create.call_args[0][0]
            assert config_arg.provider == "ollama"
