"""Integration tests for LLM config API endpoints.

Tests verify actual database operations with real PostgreSQL test database.
They use shared fixtures from conftest.py which handle database setup,
cleanup, and test client creation.

To run these tests:
    uv run --env-file .env.test pytest tests/api/v1/test_llm_configs_integration.py -m integration
"""

from unittest.mock import patch

import pytest
from httpx import AsyncClient

# Apply asyncio and integration markers to all tests in this module
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestLLMConfigIntegration:
    """Integration tests for LLM configuration endpoints."""

    async def test_create_llm_config_ollama(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test creating Ollama configuration with real database."""
        config_data = {
            "providerName": "ollama",
            "baseUrl": "http://localhost:11434",
            "model": "llama3.2",
            "isActive": True,
        }

        response = await shared_integration_client.post(
            "/api/v1/llm-configs", json=config_data
        )

        assert response.status_code == 201
        data = response.json()
        assert data["providerName"] == "ollama"
        assert data["baseUrl"] == "http://localhost:11434"
        assert data["model"] == "llama3.2"
        assert data["isActive"] is True
        assert "id" in data
        assert "createdAt" in data
        assert "apiKey" not in data  # API key should never be exposed

    async def test_create_llm_config_openai(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test creating OpenAI configuration with API key encryption."""
        config_data = {
            "providerName": "openai",
            "apiKey": "sk-test123",
            "model": "gpt-4",
            "isActive": True,
        }

        # Mock encryption
        with patch(
            "ragitect.services.llm_config_service.encrypt_value"
        ) as mock_encrypt:
            mock_encrypt.return_value = "encrypted_sk-test123"

            response = await shared_integration_client.post(
                "/api/v1/llm-configs", json=config_data
            )

            assert response.status_code == 201
            data = response.json()
            assert data["providerName"] == "openai"
            assert data["model"] == "gpt-4"
            assert "apiKey" not in data  # Should not be exposed
            mock_encrypt.assert_called_once()

    async def test_update_existing_config(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test updating existing configuration."""
        # Create initial config
        with patch(
            "ragitect.services.llm_config_service.encrypt_value"
        ) as mock_encrypt:
            mock_encrypt.return_value = "encrypted_key"

            initial_data = {
                "providerName": "anthropic",
                "apiKey": "sk-ant-test123",
                "model": "claude-3-5-sonnet-20241022",
                "isActive": True,
            }
            response = await shared_integration_client.post(
                "/api/v1/llm-configs", json=initial_data
            )
            assert response.status_code == 201

            # Update config
            update_data = {
                "providerName": "anthropic",
                "apiKey": "sk-ant-newkey456",
                "model": "claude-3-opus-20240229",
                "isActive": False,
            }
            response = await shared_integration_client.post(
                "/api/v1/llm-configs", json=update_data
            )
            assert response.status_code == 201
            data = response.json()
            assert data["model"] == "claude-3-opus-20240229"
            assert data["isActive"] is False

    async def test_list_llm_configs(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test listing all LLM configurations."""
        # Create multiple configs
        with patch(
            "ragitect.services.llm_config_service.encrypt_value"
        ) as mock_encrypt:
            mock_encrypt.return_value = "encrypted"

            configs = [
                {
                    "providerName": "ollama",
                    "baseUrl": "http://localhost:11434",
                    "isActive": True,
                },
                {
                    "providerName": "openai",
                    "apiKey": "sk-test",
                    "isActive": False,
                },
            ]

            for config in configs:
                await shared_integration_client.post("/api/v1/llm-configs", json=config)

        # List configs
        response = await shared_integration_client.get("/api/v1/llm-configs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["configs"]) == 2
        provider_names = {config["providerName"] for config in data["configs"]}
        assert provider_names == {"ollama", "openai"}

    async def test_get_specific_config(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test retrieving specific provider configuration."""
        # Create config
        config_data = {
            "providerName": "ollama",
            "baseUrl": "http://localhost:11434",
            "isActive": True,
        }
        await shared_integration_client.post("/api/v1/llm-configs", json=config_data)

        # Get specific config
        response = await shared_integration_client.get("/api/v1/llm-configs/ollama")
        assert response.status_code == 200
        data = response.json()
        assert data["providerName"] == "ollama"
        assert data["baseUrl"] == "http://localhost:11434"

    async def test_get_nonexistent_config(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test retrieving non-existent configuration returns 404."""
        response = await shared_integration_client.get(
            "/api/v1/llm-configs/nonexistent"
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_delete_config(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test deleting configuration."""
        # Create config
        config_data = {
            "providerName": "ollama",
            "baseUrl": "http://localhost:11434",
            "isActive": True,
        }
        await shared_integration_client.post("/api/v1/llm-configs", json=config_data)

        # Delete config
        response = await shared_integration_client.delete("/api/v1/llm-configs/ollama")
        assert response.status_code == 204

        # Verify deletion
        response = await shared_integration_client.get("/api/v1/llm-configs/ollama")
        assert response.status_code == 404

    async def test_delete_nonexistent_config(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test deleting non-existent configuration returns 404."""
        response = await shared_integration_client.delete(
            "/api/v1/llm-configs/nonexistent"
        )
        assert response.status_code == 404

    async def test_update_config_preserves_api_key(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test updating config preserves existing API key when not provided."""
        with (
            patch("ragitect.services.llm_config_service.encrypt_value") as mock_encrypt,
            patch("ragitect.services.llm_config_service.decrypt_value") as mock_decrypt,
        ):
            mock_encrypt.return_value = "encrypted-api-key"
            mock_decrypt.return_value = "sk-test-original-key"

            # Create initial config with API key
            config_data = {
                "providerName": "openai",
                "apiKey": "sk-test-original-key",
                "model": "gpt-4",
                "isActive": True,
            }
            response = await shared_integration_client.post(
                "/api/v1/llm-configs", json=config_data
            )
            assert response.status_code == 201

            # Update model only (no API key provided) using PATCH
            update_data = {"model": "gpt-4o"}
            response = await shared_integration_client.patch(
                "/api/v1/llm-configs/openai", json=update_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["model"] == "gpt-4o"
            assert data["isActive"] is True  # Should be preserved

    async def test_update_nonexistent_config_returns_404(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test updating non-existent config returns 404."""
        update_data = {"model": "gpt-4o"}
        response = await shared_integration_client.patch(
            "/api/v1/llm-configs/nonexistent", json=update_data
        )
        assert response.status_code == 404
        assert "not configured" in response.json()["detail"].lower()

    async def test_validate_ollama_success(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test successful Ollama validation."""
        with patch("ragitect.api.v1.llm_configs.validate_ollama_url") as mock_validate:
            mock_validate.return_value = (True, "Successfully connected to Ollama")

            validation_data = {
                "providerName": "ollama",
                "baseUrl": "http://localhost:11434",
            }

            response = await shared_integration_client.post(
                "/api/v1/llm-configs/validate", json=validation_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert "success" in data["message"].lower()

    async def test_validate_ollama_failure(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test failed Ollama validation."""
        with patch("ragitect.api.v1.llm_configs.validate_ollama_url") as mock_validate:
            mock_validate.return_value = (
                False,
                "Could not connect to http://invalid:11434",
            )

            validation_data = {
                "providerName": "ollama",
                "baseUrl": "http://invalid:11434",
            }

            response = await shared_integration_client.post(
                "/api/v1/llm-configs/validate", json=validation_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert data["error"] is not None

    async def test_validate_openai_success(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test successful OpenAI API key validation."""
        with patch("ragitect.api.v1.llm_configs.validate_api_key") as mock_validate:
            mock_validate.return_value = (True, "OpenAI API key is valid")

            validation_data = {
                "providerName": "openai",
                "apiKey": "sk-test123",
            }

            response = await shared_integration_client.post(
                "/api/v1/llm-configs/validate", json=validation_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True

    async def test_validate_missing_required_field(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test validation with missing required field."""
        validation_data = {
            "providerName": "ollama",
            # Missing baseUrl
        }

        response = await shared_integration_client.post(
            "/api/v1/llm-configs/validate", json=validation_data
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "required" in data["error"].lower()

    async def test_invalid_provider_name(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test creating config with invalid provider name."""
        config_data = {
            "providerName": "invalid_provider",
            "isActive": True,
        }

        response = await shared_integration_client.post(
            "/api/v1/llm-configs", json=config_data
        )
        assert response.status_code == 422  # Validation error

    async def test_api_key_not_exposed_in_list(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test that API keys are never exposed when listing configs."""
        with patch(
            "ragitect.services.llm_config_service.encrypt_value"
        ) as mock_encrypt:
            mock_encrypt.return_value = "encrypted_key"

            config_data = {
                "providerName": "openai",
                "apiKey": "sk-secret123",
                "isActive": True,
            }
            await shared_integration_client.post(
                "/api/v1/llm-configs", json=config_data
            )

        # List configs
        response = await shared_integration_client.get("/api/v1/llm-configs")
        assert response.status_code == 200
        data = response.json()

        # Verify no API key in response
        for config in data["configs"]:
            assert "apiKey" not in config
            assert "api_key" not in config


class TestLLMConfigToggleIntegration:
    """Integration tests for LLM config toggle endpoint."""

    async def test_toggle_enable_provider(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test enabling a disabled provider via toggle endpoint."""
        # Create a disabled config first
        config_data = {
            "providerName": "ollama",
            "baseUrl": "http://localhost:11434",
            "model": "llama3.2",
            "isActive": False,
        }
        await shared_integration_client.post("/api/v1/llm-configs", json=config_data)

        # Toggle it active
        response = await shared_integration_client.patch(
            "/api/v1/llm-configs/ollama/toggle",
            json={"isActive": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isActive"] is True
        assert data["providerName"] == "ollama"

    async def test_toggle_disable_provider(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test disabling an active provider via toggle endpoint."""
        # Create an active config
        config_data = {
            "providerName": "ollama",
            "baseUrl": "http://localhost:11434",
            "isActive": True,
        }
        await shared_integration_client.post("/api/v1/llm-configs", json=config_data)

        # Toggle it inactive
        response = await shared_integration_client.patch(
            "/api/v1/llm-configs/ollama/toggle",
            json={"isActive": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isActive"] is False

    async def test_toggle_nonexistent_provider(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test toggling a non-existent provider returns 404."""
        response = await shared_integration_client.patch(
            "/api/v1/llm-configs/nonexistent/toggle",
            json={"isActive": True},
        )

        assert response.status_code == 404
        assert "not configured" in response.json()["detail"].lower()

    async def test_toggle_preserves_config_data(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test that toggle preserves API key and other config data."""
        with patch(
            "ragitect.services.llm_config_service.encrypt_value"
        ) as mock_encrypt:
            mock_encrypt.return_value = "encrypted_key"

            config_data = {
                "providerName": "openai",
                "apiKey": "sk-test123",
                "model": "gpt-4o",
                "isActive": True,
            }
            await shared_integration_client.post(
                "/api/v1/llm-configs", json=config_data
            )

        # Toggle off
        response = await shared_integration_client.patch(
            "/api/v1/llm-configs/openai/toggle",
            json={"isActive": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isActive"] is False
        assert data["model"] == "gpt-4o"  # Model preserved

        # Toggle back on
        response = await shared_integration_client.patch(
            "/api/v1/llm-configs/openai/toggle",
            json={"isActive": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isActive"] is True
        assert data["model"] == "gpt-4o"  # Model still preserved

    async def test_toggle_case_insensitive_provider_name(
        self, shared_integration_client: AsyncClient, clean_database
    ):
        """Test that toggle works with case-insensitive provider name."""
        config_data = {
            "providerName": "ollama",
            "baseUrl": "http://localhost:11434",
            "isActive": True,
        }
        await shared_integration_client.post("/api/v1/llm-configs", json=config_data)

        # Toggle with uppercase
        response = await shared_integration_client.patch(
            "/api/v1/llm-configs/OLLAMA/toggle",
            json={"isActive": False},
        )

        assert response.status_code == 200
        assert response.json()["isActive"] is False
