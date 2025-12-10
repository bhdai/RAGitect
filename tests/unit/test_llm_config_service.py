"""Unit tests for LLM config service layer."""

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.services.llm_config_service import (
    delete_config,
    get_active_config,
    get_all_configs,
    get_config,
    save_config,
    validate_api_key,
    validate_ollama_url,
)
from ragitect.services.database.models import LLMProviderConfig


@pytest.mark.asyncio
class TestLLMConfigService:
    """Test suite for LLM configuration service layer."""

    async def test_save_config_new(self):
        """Test saving new configuration."""
        # Arrange
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        config_data = {
            "base_url": "http://localhost:11434",
            "model": "llama3.2",
            "is_active": True,
        }

        # Mock encryption
        with patch(
            "ragitect.services.llm_config_service.encrypt_value"
        ) as mock_encrypt:
            mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

            # Act
            await save_config(mock_session, "ollama", config_data)

            # Assert
            mock_session.add.assert_called_once()
            mock_session.flush.assert_awaited_once()
            mock_session.refresh.assert_awaited_once()

    async def test_save_config_update_existing(self):
        """Test updating existing configuration."""
        # Arrange
        mock_session = AsyncMock(spec=AsyncSession)
        existing_config = LLMProviderConfig(
            id=uuid.uuid4(),
            provider_name="openai",
            config_data={"api_key": "encrypted_old_key"},
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_config
        mock_session.execute = AsyncMock(return_value=mock_result)

        config_data = {
            "api_key": "sk-newkey123",
            "model": "gpt-4",
            "is_active": False,
        }

        # Mock encryption
        with patch(
            "ragitect.services.llm_config_service.encrypt_value"
        ) as mock_encrypt:
            mock_encrypt.side_effect = lambda x: f"encrypted_{x}"

            # Act
            await save_config(mock_session, "openai", config_data)

            # Assert
            assert not existing_config.is_active
            mock_session.flush.assert_awaited_once()
            mock_session.refresh.assert_awaited_once()

    async def test_save_config_invalid_provider(self):
        """Test saving config with invalid provider name."""
        mock_session = AsyncMock(spec=AsyncSession)

        with pytest.raises(ValueError, match="Invalid provider_name"):
            await save_config(mock_session, "invalid_provider", {})

    async def test_get_config_found(self):
        """Test retrieving existing configuration."""
        # Arrange
        mock_session = AsyncMock(spec=AsyncSession)
        config_id = uuid.uuid4()
        config = LLMProviderConfig(
            id=config_id,
            provider_name="openai",
            config_data={"api_key": "encrypted_key", "model": "gpt-4"},
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = config
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Mock decryption
        with patch(
            "ragitect.services.llm_config_service.decrypt_value"
        ) as mock_decrypt:
            mock_decrypt.return_value = "sk-decrypted"

            # Act
            result = await get_config(mock_session, "openai")

            # Assert
            assert result is not None
            assert result.provider_name == "openai"
            assert result.config_data["api_key"] == "sk-decrypted"
            mock_decrypt.assert_called_once_with("encrypted_key")

    async def test_get_config_not_found(self):
        """Test retrieving non-existent configuration."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await get_config(mock_session, "nonexistent")
        assert result is None

    async def test_get_all_configs(self):
        """Test retrieving all configurations."""
        mock_session = AsyncMock(spec=AsyncSession)
        configs = [
            LLMProviderConfig(
                id=uuid.uuid4(),
                provider_name="ollama",
                config_data={"base_url": "http://localhost:11434"},
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
            LLMProviderConfig(
                id=uuid.uuid4(),
                provider_name="openai",
                config_data={"api_key": "encrypted_key"},
                is_active=False,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = configs
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await get_all_configs(mock_session)
        assert len(result) == 2
        assert result[0].provider_name == "ollama"
        assert result[1].provider_name == "openai"

    async def test_delete_config_success(self):
        """Test deleting existing configuration."""
        mock_session = AsyncMock(spec=AsyncSession)
        config = LLMProviderConfig(
            id=uuid.uuid4(),
            provider_name="anthropic",
            config_data={},
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = config
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await delete_config(mock_session, "anthropic")
        assert result is True
        mock_session.delete.assert_awaited_once_with(config)
        mock_session.flush.assert_awaited_once()

    async def test_delete_config_not_found(self):
        """Test deleting non-existent configuration."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await delete_config(mock_session, "nonexistent")
        assert result is False

    async def test_validate_ollama_url_success(self):
        """Test successful Ollama URL validation."""
        with patch(
            "ragitect.services.llm_config_service.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            is_valid, message = await validate_ollama_url("http://localhost:11434")
            assert is_valid is True
            assert "Successfully connected" in message

    async def test_validate_ollama_url_connection_error(self):
        """Test Ollama URL validation with connection error."""
        with patch(
            "ragitect.services.llm_config_service.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            import httpx

            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            is_valid, message = await validate_ollama_url("http://invalid:11434")
            assert is_valid is False
            assert "Could not connect" in message

    async def test_validate_api_key_success(self):
        """Test successful API key validation."""
        with patch(
            "ragitect.services.llm_config_service.validate_llm_config"
        ) as mock_validate:
            mock_validate.return_value = (True, "")

            is_valid, message = await validate_api_key("openai", "sk-test123")
            assert is_valid is True
            assert "valid" in message.lower()

    async def test_validate_api_key_failure(self):
        """Test API key validation failure."""
        with patch(
            "ragitect.services.llm_config_service.validate_llm_config"
        ) as mock_validate:
            mock_validate.return_value = (False, "Authentication failed")

            is_valid, message = await validate_api_key("anthropic", "sk-ant-invalid")
            assert is_valid is False
            assert "Authentication failed" in message

    async def test_get_active_config_found(self):
        """Test retrieving active configuration with decrypted API key."""
        # Arrange
        mock_session = AsyncMock(spec=AsyncSession)
        config = LLMProviderConfig(
            id=uuid.uuid4(),
            provider_name="openai",
            config_data={"api_key": "encrypted_key", "model": "gpt-4"},
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = config
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Mock decryption
        with patch(
            "ragitect.services.llm_config_service.decrypt_value"
        ) as mock_decrypt:
            mock_decrypt.return_value = "sk-decrypted"

            # Act
            result = await get_active_config(mock_session)

            # Assert
            assert result is not None
            assert result.provider_name == "openai"
            assert result.config_data["api_key"] == "sk-decrypted"
            mock_decrypt.assert_called_once_with("encrypted_key")

    async def test_get_active_config_not_found(self):
        """Test retrieving active configuration when none exists."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await get_active_config(mock_session)
        assert result is None

    async def test_get_active_config_no_api_key(self):
        """Test retrieving active Ollama config without API key."""
        # Arrange
        mock_session = AsyncMock(spec=AsyncSession)
        config = LLMProviderConfig(
            id=uuid.uuid4(),
            provider_name="ollama",
            config_data={"base_url": "http://localhost:11434", "model": "llama3.2"},
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = config
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Act
        result = await get_active_config(mock_session)

        # Assert
        assert result is not None
        assert result.provider_name == "ollama"
        assert result.config_data["base_url"] == "http://localhost:11434"
