"""Pydantic schemas for LLM configuration API endpoints."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic.alias_generators import to_camel


class LLMProviderConfigCreate(BaseModel):
    """Request model for creating/updating LLM provider configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    provider_name: str = Field(
        ...,
        description="Provider name (ollama, openai, anthropic)",
        examples=["ollama", "openai", "anthropic"],
    )
    base_url: str | None = Field(
        None,
        description="Base URL for the provider (required for Ollama)",
        examples=["http://localhost:11434"],
    )
    api_key: SecretStr | None = Field(
        None,
        description="API key for cloud providers (required for OpenAI/Anthropic)",
    )
    model: str | None = Field(
        None,
        description="Model name to use with the provider",
        examples=["llama3.2", "gpt-4", "claude-3-5-sonnet-20241022"],
    )
    is_active: bool = Field(
        True,
        description="Whether this configuration is active",
    )

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, v: str) -> str:
        """Validate provider name is one of the supported providers."""
        allowed_providers = {"ollama", "openai", "anthropic"}
        if v.lower() not in allowed_providers:
            raise ValueError(
                f"Invalid provider_name. Must be one of: {', '.join(allowed_providers)}"
            )
        return v.lower()

    @field_validator("api_key")
    @classmethod
    def validate_api_key_format(cls, v: SecretStr | None, info) -> SecretStr | None:
        """Validate API key format for OpenAI and Anthropic."""
        if v is None:
            return v

        provider_name = info.data.get("provider_name", "").lower()
        api_key_str = v.get_secret_value()

        if provider_name == "openai" and not api_key_str.startswith("sk-"):
            raise ValueError("OpenAI API keys must start with 'sk-'")
        elif provider_name == "anthropic" and not api_key_str.startswith("sk-ant-"):
            raise ValueError("Anthropic API keys must start with 'sk-ant-'")

        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that required fields are present based on provider."""
        if self.provider_name == "ollama" and not self.base_url:
            raise ValueError("base_url is required for Ollama provider")
        elif self.provider_name in {"openai", "anthropic"} and not self.api_key:
            raise ValueError(
                f"api_key is required for {self.provider_name.title()} provider"
            )


class LLMProviderConfigResponse(BaseModel):
    """Response model for LLM provider configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        from_attributes=True,
    )

    id: str = Field(..., description="Configuration ID")
    provider_name: str = Field(..., description="Provider name")
    base_url: str | None = Field(None, description="Base URL (if applicable)")
    model: str | None = Field(None, description="Model name (if specified)")
    is_active: bool = Field(..., description="Whether configuration is active")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

    # Never expose api_key in responses


class LLMProviderConfigListResponse(BaseModel):
    """Response model for listing LLM provider configurations."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    configs: list[LLMProviderConfigResponse] = Field(
        ..., description="List of provider configurations"
    )
    total: int = Field(..., description="Total number of configurations")


class LLMProviderConfigValidate(BaseModel):
    """Request model for validating LLM provider configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    provider_name: str = Field(
        ..., description="Provider name to validate", examples=["ollama"]
    )
    base_url: str | None = Field(None, description="Base URL to validate (for Ollama)")
    api_key: SecretStr | None = Field(
        None, description="API key to validate (for OpenAI/Anthropic)"
    )
    model: str | None = Field(None, description="Model name to use for validation")


class LLMProviderConfigValidateResponse(BaseModel):
    """Response model for validation result."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    valid: bool = Field(..., description="Whether validation succeeded")
    message: str = Field(..., description="Validation message")
    error: str | None = Field(None, description="Error details if validation failed")


class LLMProviderToggleRequest(BaseModel):
    """Request model for toggling provider active state."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    is_active: bool = Field(
        ..., description="Whether to enable or disable the provider"
    )


# Embedding config schemas
class EmbeddingConfigCreate(BaseModel):
    """Request model for creating/updating embedding provider configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    provider_name: str = Field(
        ...,
        description="Embedding provider name (ollama, openai, vertex_ai, openai_compatible)",
        examples=["ollama", "openai", "vertex_ai"],
    )
    base_url: str | None = Field(
        None,
        description="Base URL for the provider (required for Ollama and OpenAI Compatible)",
        examples=["http://localhost:11434"],
    )
    api_key: SecretStr | None = Field(
        None,
        description="API key for cloud providers (required for OpenAI/Vertex AI)",
    )
    model: str | None = Field(
        None,
        description="Model name to use with the provider",
        examples=["nomic-embed-text", "text-embedding-3-small", "text-embedding-004"],
    )
    dimension: int | None = Field(
        None,
        description="Embedding dimension size",
        examples=[768, 1536, 3072],
    )
    is_active: bool = Field(
        True,
        description="Whether this configuration is active",
    )

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, v: str) -> str:
        """Validate provider name is one of the supported embedding providers."""
        allowed_providers = {"ollama", "openai", "vertex_ai", "openai_compatible"}
        if v.lower() not in allowed_providers:
            raise ValueError(
                f"Invalid provider_name. Must be one of: {', '.join(allowed_providers)}"
            )
        return v.lower()


class EmbeddingConfigResponse(BaseModel):
    """Response model for embedding provider configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        from_attributes=True,
    )

    id: str = Field(..., description="Configuration ID")
    provider_name: str = Field(..., description="Provider name")
    base_url: str | None = Field(None, description="Base URL (if applicable)")
    model: str | None = Field(None, description="Model name (if specified)")
    dimension: int | None = Field(None, description="Embedding dimension")
    is_active: bool = Field(..., description="Whether configuration is active")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class EmbeddingConfigListResponse(BaseModel):
    """Response model for listing embedding provider configurations."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    configs: list[EmbeddingConfigResponse] = Field(
        ..., description="List of embedding configurations"
    )
    total: int = Field(..., description="Total number of configurations")


class EmbeddingConfigValidate(BaseModel):
    """Request model for validating embedding provider configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    provider_name: str = Field(
        ..., description="Provider name to validate", examples=["ollama"]
    )
    base_url: str | None = Field(None, description="Base URL to validate (for Ollama)")
    api_key: SecretStr | None = Field(
        None, description="API key to validate (for OpenAI/Vertex AI)"
    )
    model: str | None = Field(None, description="Model name to use for validation")
