"""Service layer for LLM provider configuration management."""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.services.config import LLMConfig
from ragitect.services.database.models import EmbeddingProviderConfig, LLMProviderConfig
from ragitect.services.llm import validate_llm_config
from ragitect.utils.encryption import decrypt_value, encrypt_value


@dataclass
class LLMConfigDTO:
    """Read-only DTO for LLM configuration with decrypted API key.

    This is a data transfer object that provides a clean separation between
    the ORM model (which stores encrypted data) and the runtime representation
    (which has decrypted values for use). This prevents accidental write-back
    of decrypted values to the database.

    Attributes:
        id: Unique identifier
        provider_name: Provider name (ollama, openai, anthropic)
        config_data: Configuration data with decrypted api_key
        is_active: Whether this config is active
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    id: Any
    provider_name: str
    config_data: dict[str, Any]
    is_active: bool
    created_at: Any
    updated_at: Any

    # Convenience properties for common access patterns
    @property
    def model_name(self) -> str | None:
        """Get model name from config_data."""
        return self.config_data.get("model")

    @property
    def api_key(self) -> str | None:
        """Get decrypted API key from config_data."""
        return self.config_data.get("api_key")

    @property
    def base_url(self) -> str | None:
        """Get base URL from config_data."""
        return self.config_data.get("base_url")

    @classmethod
    def from_orm(cls, config: LLMProviderConfig) -> "LLMConfigDTO":
        """Create DTO from ORM model with decrypted API key.

        Args:
            config: LLMProviderConfig ORM model

        Returns:
            LLMConfigDTO with decrypted api_key in config_data
        """
        config_data = config.config_data.copy()
        if "api_key" in config_data and config_data["api_key"]:
            config_data["api_key"] = decrypt_value(config_data["api_key"])

        return cls(
            id=config.id,
            provider_name=config.provider_name,
            config_data=config_data,
            is_active=config.is_active,
            created_at=config.created_at,
            updated_at=config.updated_at,
        )


@dataclass
class EmbeddingConfigDTO:
    """Read-only DTO for embedding configuration with decrypted API key.

    Same pattern as LLMConfigDTO - separates read-model from persistence model.

    Attributes:
        id: Unique identifier
        provider_name: Provider name (ollama, openai, vertex_ai, openai_compatible)
        config_data: Configuration data with decrypted api_key
        is_active: Whether this config is active
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    id: Any
    provider_name: str
    config_data: dict[str, Any]
    is_active: bool
    created_at: Any
    updated_at: Any

    # Convenience properties for common access patterns
    @property
    def model_name(self) -> str | None:
        """Get model name from config_data."""
        return self.config_data.get("model")

    @property
    def api_key(self) -> str | None:
        """Get decrypted API key from config_data."""
        return self.config_data.get("api_key")

    @property
    def base_url(self) -> str | None:
        """Get base URL from config_data."""
        return self.config_data.get("base_url")

    @property
    def dimension(self) -> int:
        """Get embedding dimension from config_data."""
        return self.config_data.get("dimension", 768)

    @classmethod
    def from_orm(cls, config: EmbeddingProviderConfig) -> "EmbeddingConfigDTO":
        """Create DTO from ORM model with decrypted API key.

        Args:
            config: EmbeddingProviderConfig ORM model

        Returns:
            EmbeddingConfigDTO with decrypted api_key in config_data
        """
        config_data = config.config_data.copy()
        if "api_key" in config_data and config_data["api_key"]:
            config_data["api_key"] = decrypt_value(config_data["api_key"])

        return cls(
            id=config.id,
            provider_name=config.provider_name,
            config_data=config_data,
            is_active=config.is_active,
            created_at=config.created_at,
            updated_at=config.updated_at,
        )


async def save_config(
    session: AsyncSession, provider_name: str, config_data: dict[str, Any]
) -> LLMProviderConfig:
    """Save or update LLM provider configuration.

    Args:
        session: Database session
        provider_name: Provider name (ollama, openai, anthropic)
        config_data: Configuration data including api_key, base_url, model, etc.

    Returns:
        LLMProviderConfig: Saved configuration

    Raises:
        ValueError: If provider_name is invalid or required fields are missing
    """
    # Validate provider_name
    valid_providers = {"ollama", "openai", "anthropic"}
    if provider_name.lower() not in valid_providers:
        raise ValueError(
            f"Invalid provider_name. Must be one of: {', '.join(valid_providers)}"
        )

    # Encrypt API key if present
    encrypted_data = config_data.copy()
    if "api_key" in encrypted_data and encrypted_data["api_key"]:
        encrypted_data["api_key"] = encrypt_value(encrypted_data["api_key"])

    # Check if config already exists
    result = await session.execute(
        select(LLMProviderConfig).where(
            LLMProviderConfig.provider_name == provider_name.lower()
        )
    )
    existing_config = result.scalar_one_or_none()

    if existing_config:
        # Update existing configuration
        existing_config.config_data = encrypted_data
        existing_config.is_active = config_data.get("is_active", True)
        await session.flush()
        await session.refresh(existing_config)
        return existing_config
    else:
        # Create new configuration
        new_config = LLMProviderConfig(
            provider_name=provider_name.lower(),
            config_data=encrypted_data,
            is_active=config_data.get("is_active", True),
        )
        session.add(new_config)
        await session.flush()
        await session.refresh(new_config)
        return new_config


async def get_config(session: AsyncSession, provider_name: str) -> LLMConfigDTO | None:
    """Retrieve and decrypt LLM provider configuration.

    Returns a read-only DTO with decrypted API key, separate from ORM model
    to prevent accidental write-back of decrypted values.

    Args:
        session: Database session
        provider_name: Provider name

    Returns:
        LLMConfigDTO | None: Configuration DTO with decrypted API key, or None
    """
    result = await session.execute(
        select(LLMProviderConfig).where(
            LLMProviderConfig.provider_name == provider_name.lower()
        )
    )
    config = result.scalar_one_or_none()

    return LLMConfigDTO.from_orm(config) if config else None


async def get_all_configs(session: AsyncSession) -> list[LLMProviderConfig]:
    """Retrieve all LLM provider configurations.

    Args:
        session: Database session

    Returns:
        list[LLMProviderConfig]: List of all configurations (with API keys masked)
    """
    result = await session.execute(select(LLMProviderConfig))
    configs = result.scalars().all()

    # Don't decrypt API keys when listing - they should not be exposed
    return list(configs)


async def get_active_config(session: AsyncSession) -> LLMConfigDTO | None:
    """Retrieve the first active LLM provider configuration with decrypted API key.

    Returns a read-only DTO with decrypted API key, separate from ORM model
    to prevent accidental write-back of decrypted values.

    Args:
        session: Database session

    Returns:
        LLMConfigDTO | None: Active configuration DTO with decrypted API key, or None
    """
    result = await session.execute(
        select(LLMProviderConfig).where(LLMProviderConfig.is_active.is_(True))
    )
    config = result.scalars().first()

    return LLMConfigDTO.from_orm(config) if config else None


async def delete_config(session: AsyncSession, provider_name: str) -> bool:
    """Delete LLM provider configuration.

    Args:
        session: Database session
        provider_name: Provider name

    Returns:
        bool: True if deleted, False if not found
    """
    result = await session.execute(
        select(LLMProviderConfig).where(
            LLMProviderConfig.provider_name == provider_name.lower()
        )
    )
    config = result.scalar_one_or_none()

    if config:
        await session.delete(config)
        await session.flush()
        return True
    return False


async def validate_ollama_url(base_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Validate Ollama base URL by sending test request to health endpoint.

    Args:
        base_url: Ollama base URL
        timeout: Request timeout in seconds

    Returns:
        tuple[bool, str]: (is_valid, message)
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
            if response.status_code == 200:
                return True, "Successfully connected to Ollama"
            else:
                return False, f"Ollama returned status code: {response.status_code}"
    except httpx.TimeoutException:
        return False, f"Connection timed out after {timeout} seconds"
    except httpx.ConnectError:
        return False, f"Could not connect to {base_url}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def validate_api_key(
    provider_name: str, api_key: str, model: str | None = None, timeout: float = 5.0
) -> tuple[bool, str]:
    """Validate API key for OpenAI or Anthropic by making test API call.

    Args:
        provider_name: Provider name (openai, anthropic)
        api_key: API key to validate
        model: Model name to use for test call
        timeout: Request timeout in seconds

    Returns:
        tuple[bool, str]: (is_valid, message)
    """
    # Use existing validate_llm_config function
    config = LLMConfig(
        provider=provider_name,
        model=model
        or ("gpt-4" if provider_name == "openai" else "claude-3-5-sonnet-20241022"),
        api_key=api_key,
        temperature=0.7,
    )

    try:
        is_valid, error_msg = await asyncio.wait_for(
            validate_llm_config(config), timeout=timeout
        )
        if is_valid:
            return True, f"{provider_name.title()} API key is valid"
        else:
            return (
                False,
                error_msg or f"{provider_name.title()} API key validation failed",
            )
    except asyncio.TimeoutError:
        return False, f"Validation timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# Embedding config service methods
async def save_embedding_config(
    session: AsyncSession, provider_name: str, config_data: dict[str, Any]
) -> EmbeddingProviderConfig:
    """Save or update embedding provider configuration.

    Args:
        session: Database session
        provider_name: Provider name (ollama, openai, vertex_ai, openai_compatible)
        config_data: Configuration data including api_key, base_url, model, dimension, etc.

    Returns:
        EmbeddingProviderConfig: Saved configuration

    Raises:
        ValueError: If provider_name is invalid or required fields are missing
    """
    # Validate provider_name
    valid_providers = {"ollama", "openai", "vertex_ai", "openai_compatible"}
    if provider_name.lower() not in valid_providers:
        raise ValueError(
            f"Invalid provider_name. Must be one of: {', '.join(valid_providers)}"
        )

    # Encrypt API key if present
    encrypted_data = config_data.copy()
    if "api_key" in encrypted_data and encrypted_data["api_key"]:
        encrypted_data["api_key"] = encrypt_value(encrypted_data["api_key"])

    # Check if config already exists
    result = await session.execute(
        select(EmbeddingProviderConfig).where(
            EmbeddingProviderConfig.provider_name == provider_name.lower()
        )
    )
    existing_config = result.scalar_one_or_none()

    if existing_config:
        # Update existing configuration
        existing_config.config_data = encrypted_data
        existing_config.is_active = config_data.get("is_active", True)
        await session.flush()
        await session.refresh(existing_config)
        return existing_config
    else:
        # Create new configuration
        new_config = EmbeddingProviderConfig(
            provider_name=provider_name.lower(),
            config_data=encrypted_data,
            is_active=config_data.get("is_active", True),
        )
        session.add(new_config)
        await session.flush()
        await session.refresh(new_config)
        return new_config


async def get_embedding_config(
    session: AsyncSession, provider_name: str
) -> EmbeddingConfigDTO | None:
    """Retrieve and decrypt embedding provider configuration.

    Returns a read-only DTO with decrypted API key, separate from ORM model
    to prevent accidental write-back of decrypted values.

    Args:
        session: Database session
        provider_name: Provider name

    Returns:
        EmbeddingConfigDTO | None: Configuration DTO with decrypted API key, or None
    """
    result = await session.execute(
        select(EmbeddingProviderConfig).where(
            EmbeddingProviderConfig.provider_name == provider_name.lower()
        )
    )
    config = result.scalar_one_or_none()

    return EmbeddingConfigDTO.from_orm(config) if config else None


async def get_all_embedding_configs(
    session: AsyncSession,
) -> list[EmbeddingProviderConfig]:
    """Retrieve all embedding provider configurations.

    Args:
        session: Database session

    Returns:
        list[EmbeddingProviderConfig]: List of all configurations (with API keys masked)
    """
    result = await session.execute(select(EmbeddingProviderConfig))
    configs = result.scalars().all()

    # Don't decrypt API keys when listing - they should not be exposed
    return list(configs)


async def get_active_embedding_config(
    session: AsyncSession,
) -> EmbeddingConfigDTO | None:
    """Retrieve the active embedding provider configuration with decrypted API key.

    Returns a read-only DTO with decrypted API key, separate from ORM model
    to prevent accidental write-back of decrypted values.

    Args:
        session: Database session

    Returns:
        EmbeddingConfigDTO | None: Active configuration DTO with decrypted API key, or None
    """
    result = await session.execute(
        select(EmbeddingProviderConfig).where(
            EmbeddingProviderConfig.is_active.is_(True)
        )
    )
    config = result.scalars().first()

    return EmbeddingConfigDTO.from_orm(config) if config else None


async def delete_embedding_config(session: AsyncSession, provider_name: str) -> bool:
    """Delete embedding provider configuration.

    Args:
        session: Database session
        provider_name: Provider name

    Returns:
        bool: True if deleted, False if not found
    """
    result = await session.execute(
        select(EmbeddingProviderConfig).where(
            EmbeddingProviderConfig.provider_name == provider_name.lower()
        )
    )
    config = result.scalar_one_or_none()

    if config:
        await session.delete(config)
        await session.flush()
        return True
    return False


async def validate_embedding_config(
    provider_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """Validate embedding provider configuration.

    Args:
        provider_name: Provider name (ollama, openai, vertex_ai, openai_compatible)
        base_url: Base URL (for Ollama/OpenAI compatible)
        api_key: API key (for OpenAI/Vertex AI)
        model: Model name to use for validation
        timeout: Request timeout in seconds

    Returns:
        tuple[bool, str]: (is_valid, message)
    """
    try:
        if provider_name in {"ollama", "openai_compatible"}:
            if not base_url:
                return False, f"base_url is required for {provider_name}"

            # For Ollama, test the base URL
            if provider_name == "ollama":
                return await validate_ollama_url(base_url, timeout)
            else:
                # For OpenAI compatible, just check connectivity
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(f"{base_url.rstrip('/')}/")
                    if response.status_code in {200, 404}:
                        return True, "Successfully connected to endpoint"
                    return (
                        False,
                        f"Endpoint returned status code: {response.status_code}",
                    )

        elif provider_name == "openai":
            if not api_key:
                return False, "api_key is required for OpenAI"

            # Test with a simple embeddings call
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "input": "test",
                        "model": model or "text-embedding-3-small",
                    },
                )
                if response.status_code == 200:
                    return True, "OpenAI API key is valid"
                else:
                    return False, f"OpenAI API returned error: {response.text}"

        elif provider_name == "vertex_ai":
            # Vertex AI validation would require Google Cloud SDK setup
            # For now, just validate that api_key is provided
            if not api_key:
                return False, "api_key is required for Vertex AI"
            return (
                True,
                "Vertex AI configuration accepted (full validation requires Google Cloud SDK)",
            )

        else:
            return False, f"Unsupported provider: {provider_name}"

    except httpx.TimeoutException:
        return False, f"Connection timed out after {timeout} seconds"
    except httpx.ConnectError as e:
        return False, f"Could not connect: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"
