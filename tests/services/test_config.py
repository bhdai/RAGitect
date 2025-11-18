"""Tests for config.py"""

import os
import pytest
from unittest.mock import patch

from ragitect.services.config import (
    LLMConfig,
    DocumentConfig,
    load_config_from_env,
    get_default_config,
)


class TestLLMConfig:
    """Test LLMConfig dataclass"""

    def test_default_values(self):
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.model == "llama3.1:8b"
        assert config.temperature == 0.7
        assert config.timeout == 60

    def test_custom_values(self):
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.5,
            api_key="test-key",
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.api_key == "test-key"


class TestDocumentConfig:
    """Test DocumentConfig dataclass"""

    def test_default_values(self):
        config = DocumentConfig()
        assert config.enable_docling is True
        assert config.enable_unstructure is False
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50


class TestGetDefaultConfig:
    """Test default configuration getter"""

    def test_returns_ollama_config(self):
        config = get_default_config()
        assert config.provider == "ollama"
        assert config.model == "llama3.1:8b"
        assert config.base_url == "http://localhost:11434"

    def test_returns_immutable_config(self):
        config1 = get_default_config()
        config2 = get_default_config()

        # Should return new instances
        assert config1 is not config2


class TestLoadConfigFromEnv:
    """Test environment variable loading"""

    @patch.dict(os.environ, {}, clear=True)
    def test_uses_defaults_when_no_env_vars(self):
        config = load_config_from_env()
        assert config.provider == "ollama"
        assert config.model == "llama3.1:8b"
        assert config.temperature == 0.7

    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4",
            "LLM_API_KEY": "sk-test123",
        },
        clear=True,
    )
    def test_loads_from_environment(self):
        config = load_config_from_env()
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "sk-test123"

    @patch.dict(os.environ, {"LLM_TEMPERATURE": "0.9"}, clear=True)
    def test_parses_temperature_float(self):
        config = load_config_from_env()
        assert config.temperature == 0.9

    @patch.dict(os.environ, {"LLM_TEMPERATURE": "invalid"}, clear=True)
    def test_handles_invalid_temperature(self):
        config = load_config_from_env()
        assert config.temperature == 0.7  # Falls back to default

    @patch.dict(os.environ, {"LLM_MAX_TOKENS": "1000"}, clear=True)
    def test_parses_max_tokens(self):
        config = load_config_from_env()
        assert config.max_tokens == 1000

    @patch.dict(os.environ, {"LLM_MAX_TOKENS": "not_a_number"}, clear=True)
    def test_handles_invalid_max_tokens(self):
        config = load_config_from_env()
        assert config.max_tokens is None  # Falls back to None

    @patch.dict(os.environ, {"LLM_TIMEOUT": "120"}, clear=True)
    def test_parses_timeout(self):
        config = load_config_from_env()
        assert config.timeout == 120

    @patch.dict(os.environ, {"LLM_TIMEOUT": "invalid"}, clear=True)
    def test_handles_invalid_timeout(self):
        config = load_config_from_env()
        assert config.timeout == 60  # Falls back to default

    @patch.dict(
        os.environ,
        {"LLM_PROVIDER": "ollama"},
        clear=True,
    )
    def test_sets_default_base_url_for_ollama(self):
        config = load_config_from_env()
        assert config.base_url == "http://localhost:11434"

    @patch.dict(
        os.environ,
        {"LLM_PROVIDER": "openai", "LLM_BASE_URL": ""},
        clear=True,
    )
    def test_no_default_base_url_for_non_ollama(self):
        config = load_config_from_env()
        assert config.base_url == ""

    @patch.dict(os.environ, {"LLM_CUSTOM_PROVIDER": "custom"}, clear=True)
    def test_loads_custom_provider(self):
        config = load_config_from_env()
        assert config.custom_provider == "custom"

    @patch.dict(
        os.environ,
        {"LLM_PROVIDER": "OpEnAi"},  # Mixed case
        clear=True,
    )
    def test_normalizes_provider_to_lowercase(self):
        config = load_config_from_env()
        assert config.provider == "openai"
