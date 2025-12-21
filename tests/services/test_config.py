"""Tests for config.py"""

import os
from unittest.mock import patch

from ragitect.services.config import (
    DocumentConfig,
    EmbeddingConfig,
    LLMConfig,
    get_default_config,
    load_config_from_env,
    load_document_config,
    load_embedding_config,
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
    """Test DocumentConfig dataclass with token-based chunking"""

    def test_default_values(self):
        """Test default values use token-based sizing"""
        config = DocumentConfig()
        assert config.enable_docling is True
        assert config.enable_unstructure is False
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.min_chunk_size == 64

    def test_loads_chunk_size_from_env(self):
        """Test that CHUNK_SIZE_TOKENS env var overrides default"""
        with patch.dict(os.environ, {"CHUNK_SIZE_TOKENS": "256"}):
            config = load_document_config()
            assert config.chunk_size == 256
            assert config.chunk_overlap == 50  # Should still use default

    def test_loads_chunk_overlap_from_env(self):
        """Test that CHUNK_OVERLAP_TOKENS env var overrides default"""
        with patch.dict(os.environ, {"CHUNK_OVERLAP_TOKENS": "100"}):
            config = load_document_config()
            assert config.chunk_size == 512  # Should still use default
            assert config.chunk_overlap == 100

    def test_loads_both_chunk_params_from_env(self):
        """Test that both chunk env vars can be set together"""
        with patch.dict(
            os.environ,
            {"CHUNK_SIZE_TOKENS": "1024", "CHUNK_OVERLAP_TOKENS": "100"},
        ):
            config = load_document_config()
            assert config.chunk_size == 1024
            assert config.chunk_overlap == 100

    def test_loads_min_chunk_size_from_env(self):
        """Test that MIN_CHUNK_SIZE_TOKENS env var overrides default"""
        with patch.dict(os.environ, {"MIN_CHUNK_SIZE_TOKENS": "32"}):
            config = load_document_config()
            assert config.min_chunk_size == 32


class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass with batch processing"""

    def test_default_values(self):
        """Test default values including batch_size"""
        config = EmbeddingConfig()
        assert config.provider == "ollama"
        assert config.model == "qwen3-embedding:0.6b"  # Updated default
        assert config.base_url == "http://localhost:11434"
        assert config.dimension == 768
        assert config.batch_size == 32

    def test_batch_size_configurable(self):
        """Test that batch_size can be customized"""
        config = EmbeddingConfig(batch_size=16)
        assert config.batch_size == 16

    def test_load_embedding_config_from_env(self):
        """Test that load_embedding_config reads from environment"""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-3-small",
                "EMBEDDING_BATCH_SIZE": "64",
            },
        ):
            config = load_embedding_config()
            assert config.provider == "openai"
            assert config.model == "text-embedding-3-small"
            assert config.batch_size == 64


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
