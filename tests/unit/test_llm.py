"""Unit tests for LLM service functions."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from ragitect.services.config import LLMConfig
from ragitect.services.database.models import LLMProviderConfig
from ragitect.services.llm import (
    create_llm_from_db,
    get_active_llm_config,
)


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
