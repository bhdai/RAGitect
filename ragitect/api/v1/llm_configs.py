"""API router for LLM provider configuration endpoints."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.api.schemas.llm_config import (
    LLMProviderConfigCreate,
    LLMProviderConfigListResponse,
    LLMProviderConfigResponse,
    LLMProviderConfigValidate,
    LLMProviderConfigValidateResponse,
)
from ragitect.services.database.connection import get_async_session
from ragitect.services.llm_config_service import (
    delete_config,
    get_all_configs,
    get_config,
    save_config,
    validate_api_key,
    validate_ollama_url,
)

router = APIRouter(prefix="/llm-configs", tags=["LLM Configuration"])


@router.post(
    "",
    response_model=LLMProviderConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save LLM provider configuration",
    description="Create or update LLM provider configuration with encrypted API keys",
)
async def create_llm_config(
    config_data: LLMProviderConfigCreate,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> LLMProviderConfigResponse:
    """Save or update LLM provider configuration."""
    # Prepare config data dictionary
    config_dict = {
        "is_active": config_data.is_active,
    }

    if config_data.base_url:
        config_dict["base_url"] = config_data.base_url

    if config_data.api_key:
        config_dict["api_key"] = config_data.api_key.get_secret_value()

    if config_data.model:
        config_dict["model"] = config_data.model

    # Save configuration
    saved_config = await save_config(session, config_data.provider_name, config_dict)

    # Prepare response (never expose API key)
    return LLMProviderConfigResponse(
        id=str(saved_config.id),
        provider_name=saved_config.provider_name,
        base_url=saved_config.config_data.get("base_url"),
        model=saved_config.config_data.get("model"),
        is_active=saved_config.is_active,
        created_at=saved_config.created_at.isoformat(),
        updated_at=saved_config.updated_at.isoformat(),
    )


@router.get(
    "",
    response_model=LLMProviderConfigListResponse,
    summary="List all LLM provider configurations",
    description="Retrieve all saved LLM provider configurations (API keys masked)",
)
async def list_llm_configs(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> LLMProviderConfigListResponse:
    """List all LLM provider configurations."""
    configs = await get_all_configs(session)

    config_responses = [
        LLMProviderConfigResponse(
            id=str(config.id),
            provider_name=config.provider_name,
            base_url=config.config_data.get("base_url"),
            model=config.config_data.get("model"),
            is_active=config.is_active,
            created_at=config.created_at.isoformat(),
            updated_at=config.updated_at.isoformat(),
        )
        for config in configs
    ]

    return LLMProviderConfigListResponse(
        configs=config_responses,
        total=len(config_responses),
    )


@router.get(
    "/{provider_name}",
    response_model=LLMProviderConfigResponse,
    summary="Get specific provider configuration",
    description="Retrieve configuration for a specific LLM provider (API key masked)",
)
async def get_llm_config(
    provider_name: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> LLMProviderConfigResponse:
    """Get configuration for specific provider."""
    config = await get_config(session, provider_name)

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration for provider '{provider_name}' not found",
        )

    return LLMProviderConfigResponse(
        id=str(config.id),
        provider_name=config.provider_name,
        base_url=config.config_data.get("base_url"),
        model=config.config_data.get("model"),
        is_active=config.is_active,
        created_at=config.created_at.isoformat(),
        updated_at=config.updated_at.isoformat(),
    )


@router.delete(
    "/{provider_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete provider configuration",
    description="Remove configuration for a specific LLM provider",
)
async def delete_llm_config(
    provider_name: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> None:
    """Delete configuration for specific provider."""
    deleted = await delete_config(session, provider_name)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration for provider '{provider_name}' not found",
        )


@router.post(
    "/validate",
    response_model=LLMProviderConfigValidateResponse,
    summary="Validate LLM provider configuration",
    description="Test connectivity and validate API keys for LLM providers",
)
async def validate_llm_provider_config(
    validation_data: LLMProviderConfigValidate,
) -> LLMProviderConfigValidateResponse:
    """Validate LLM provider configuration."""
    provider = validation_data.provider_name.lower()

    try:
        if provider == "ollama":
            if not validation_data.base_url:
                return LLMProviderConfigValidateResponse(
                    valid=False,
                    message="Validation failed",
                    error="base_url is required for Ollama provider",
                )

            is_valid, message = await validate_ollama_url(validation_data.base_url)
            return LLMProviderConfigValidateResponse(
                valid=is_valid,
                message=message,
                error=None if is_valid else message,
            )

        elif provider in {"openai", "anthropic"}:
            if not validation_data.api_key:
                return LLMProviderConfigValidateResponse(
                    valid=False,
                    message="Validation failed",
                    error=f"api_key is required for {provider.title()} provider",
                )

            is_valid, message = await validate_api_key(
                provider,
                validation_data.api_key.get_secret_value(),
                validation_data.model,
            )
            return LLMProviderConfigValidateResponse(
                valid=is_valid,
                message=message,
                error=None if is_valid else message,
            )

        else:
            return LLMProviderConfigValidateResponse(
                valid=False,
                message="Validation failed",
                error=f"Unsupported provider: {provider}",
            )

    except Exception as e:
        return LLMProviderConfigValidateResponse(
            valid=False,
            message="Validation error",
            error=str(e),
        )
