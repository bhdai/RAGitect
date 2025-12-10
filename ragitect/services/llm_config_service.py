"""Service layer for LLM provider configuration management."""

import asyncio
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.services.config import LLMConfig
from ragitect.services.database.models import LLMProviderConfig
from ragitect.services.llm import validate_llm_config
from ragitect.utils.encryption import decrypt_value, encrypt_value


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


async def get_config(
    session: AsyncSession, provider_name: str
) -> LLMProviderConfig | None:
    """Retrieve and decrypt LLM provider configuration.

    Args:
        session: Database session
        provider_name: Provider name

    Returns:
        LLMProviderConfig | None: Configuration with decrypted API key, or None
    """
    result = await session.execute(
        select(LLMProviderConfig).where(
            LLMProviderConfig.provider_name == provider_name.lower()
        )
    )
    config = result.scalar_one_or_none()

    if config and "api_key" in config.config_data and config.config_data["api_key"]:
        # Decrypt API key for use
        config_data_copy = config.config_data.copy()
        config_data_copy["api_key"] = decrypt_value(config.config_data["api_key"])
        config.config_data = config_data_copy

    return config


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


async def get_active_config(session: AsyncSession) -> LLMProviderConfig | None:
    """Retrieve the first active LLM provider configuration with decrypted API key.

    Args:
        session: Database session

    Returns:
        LLMProviderConfig | None: Active configuration with decrypted API key, or None
    """
    result = await session.execute(
        select(LLMProviderConfig).where(LLMProviderConfig.is_active.is_(True))
    )
    config = result.scalars().first()

    if config and "api_key" in config.config_data and config.config_data["api_key"]:
        # Decrypt API key for use
        config_data_copy = config.config_data.copy()
        config_data_copy["api_key"] = decrypt_value(config.config_data["api_key"])
        config.config_data = config_data_copy

    return config


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
