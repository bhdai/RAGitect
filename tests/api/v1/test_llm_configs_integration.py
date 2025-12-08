"""Integration tests for LLM config API endpoints.

Tests verify actual database operations with real PostgreSQL test database.
Requires DATABASE_URL environment variable pointing to test database.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from main import app
from ragitect.services.database.connection import DatabaseManager

# Skip all tests if DATABASE_URL not set
pytestmark = pytest.mark.integration

# Ensure test database is used
TEST_DATABASE_URL = os.getenv("DATABASE_URL")
if not TEST_DATABASE_URL or "ragitect_test" not in TEST_DATABASE_URL:
    pytest.skip(
        "Integration tests require DATABASE_URL pointing to ragitect_test database",
        allow_module_level=True,
    )


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_test_database():
    """Initialize test database connection for integration tests."""
    db_manager = DatabaseManager.get_instance()
    await db_manager.initialize()
    yield
    await db_manager.close()


@pytest_asyncio.fixture
async def integration_client():
    """HTTP client for integration tests with real database."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest_asyncio.fixture(autouse=True)
async def cleanup_llm_configs():
    """Clean up llm_provider_configs table before and after each test."""
    from ragitect.services.database.connection import get_session

    # Clean before test
    async with get_session() as session:
        await session.execute(text("DELETE FROM llm_provider_configs"))
        await session.commit()

    yield

    # Clean after test
    async with get_session() as session:
        await session.execute(text("DELETE FROM llm_provider_configs"))
        await session.commit()


@pytest.mark.asyncio
class TestLLMConfigIntegration:
    """Integration tests for LLM configuration endpoints."""

    async def test_create_llm_config_ollama(self, integration_client):
        """Test creating Ollama configuration with real database."""
        config_data = {
            "providerName": "ollama",
            "baseUrl": "http://localhost:11434",
            "model": "llama3.2",
            "isActive": True,
        }

        response = await integration_client.post(
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

    async def test_create_llm_config_openai(self, integration_client):
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

            response = await integration_client.post(
                "/api/v1/llm-configs", json=config_data
            )

            assert response.status_code == 201
            data = response.json()
            assert data["providerName"] == "openai"
            assert data["model"] == "gpt-4"
            assert "apiKey" not in data  # Should not be exposed
            mock_encrypt.assert_called_once()

    async def test_update_existing_config(self, integration_client):
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
            response = await integration_client.post(
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
            response = await integration_client.post(
                "/api/v1/llm-configs", json=update_data
            )
            assert response.status_code == 201
            data = response.json()
            assert data["model"] == "claude-3-opus-20240229"
            assert data["isActive"] is False

    async def test_list_llm_configs(self, integration_client):
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
                await integration_client.post("/api/v1/llm-configs", json=config)

        # List configs
        response = await integration_client.get("/api/v1/llm-configs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["configs"]) == 2
        provider_names = {config["providerName"] for config in data["configs"]}
        assert provider_names == {"ollama", "openai"}

    async def test_get_specific_config(self, integration_client):
        """Test retrieving specific provider configuration."""
        # Create config
        config_data = {
            "providerName": "ollama",
            "baseUrl": "http://localhost:11434",
            "isActive": True,
        }
        await integration_client.post("/api/v1/llm-configs", json=config_data)

        # Get specific config
        response = await integration_client.get("/api/v1/llm-configs/ollama")
        assert response.status_code == 200
        data = response.json()
        assert data["providerName"] == "ollama"
        assert data["baseUrl"] == "http://localhost:11434"

    async def test_get_nonexistent_config(self, integration_client):
        """Test retrieving non-existent configuration returns 404."""
        response = await integration_client.get("/api/v1/llm-configs/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_delete_config(self, integration_client):
        """Test deleting configuration."""
        # Create config
        config_data = {
            "providerName": "ollama",
            "baseUrl": "http://localhost:11434",
            "isActive": True,
        }
        await integration_client.post("/api/v1/llm-configs", json=config_data)

        # Delete config
        response = await integration_client.delete("/api/v1/llm-configs/ollama")
        assert response.status_code == 204

        # Verify deletion
        response = await integration_client.get("/api/v1/llm-configs/ollama")
        assert response.status_code == 404

    async def test_delete_nonexistent_config(self, integration_client):
        """Test deleting non-existent configuration returns 404."""
        response = await integration_client.delete("/api/v1/llm-configs/nonexistent")
        assert response.status_code == 404

    async def test_validate_ollama_success(self, integration_client):
        """Test successful Ollama validation."""
        with patch(
            "ragitect.services.llm_config_service.validate_ollama_url"
        ) as mock_validate:
            mock_validate.return_value = (True, "Successfully connected to Ollama")

            validation_data = {
                "providerName": "ollama",
                "baseUrl": "http://localhost:11434",
            }

            response = await integration_client.post(
                "/api/v1/llm-configs/validate", json=validation_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert "success" in data["message"].lower()

    async def test_validate_ollama_failure(self, integration_client):
        """Test failed Ollama validation."""
        with patch(
            "ragitect.services.llm_config_service.validate_ollama_url"
        ) as mock_validate:
            mock_validate.return_value = (
                False,
                "Could not connect to http://invalid:11434",
            )

            validation_data = {
                "providerName": "ollama",
                "baseUrl": "http://invalid:11434",
            }

            response = await integration_client.post(
                "/api/v1/llm-configs/validate", json=validation_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert data["error"] is not None

    async def test_validate_openai_success(self, integration_client):
        """Test successful OpenAI API key validation."""
        with patch(
            "ragitect.services.llm_config_service.validate_api_key"
        ) as mock_validate:
            mock_validate.return_value = (True, "OpenAI API key is valid")

            validation_data = {
                "providerName": "openai",
                "apiKey": "sk-test123",
            }

            response = await integration_client.post(
                "/api/v1/llm-configs/validate", json=validation_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True

    async def test_validate_missing_required_field(self, integration_client):
        """Test validation with missing required field."""
        validation_data = {
            "providerName": "ollama",
            # Missing baseUrl
        }

        response = await integration_client.post(
            "/api/v1/llm-configs/validate", json=validation_data
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "required" in data["error"].lower()

    async def test_invalid_provider_name(self, integration_client):
        """Test creating config with invalid provider name."""
        config_data = {
            "providerName": "invalid_provider",
            "isActive": True,
        }

        response = await integration_client.post(
            "/api/v1/llm-configs", json=config_data
        )
        assert response.status_code == 422  # Validation error

    async def test_api_key_not_exposed_in_list(self, integration_client):
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
            await integration_client.post("/api/v1/llm-configs", json=config_data)

        # List configs
        response = await integration_client.get("/api/v1/llm-configs")
        assert response.status_code == 200
        data = response.json()

        # Verify no API key in response
        for config in data["configs"]:
            assert "apiKey" not in config
            assert "api_key" not in config
