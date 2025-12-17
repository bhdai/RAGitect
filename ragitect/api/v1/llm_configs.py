"""API router for LLM provider configuration endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragitect.api.schemas.llm_config import (
    EmbeddingConfigCreate,
    EmbeddingConfigListResponse,
    EmbeddingConfigResponse,
    EmbeddingConfigValidate,
    LLMProviderConfigCreate,
    LLMProviderConfigListResponse,
    LLMProviderConfigResponse,
    LLMProviderConfigUpdate,
    LLMProviderConfigValidate,
    LLMProviderConfigValidateResponse,
    LLMProviderToggleRequest,
)
from ragitect.services.database.connection import get_async_session
from ragitect.services.database.models import LLMProviderConfig
from ragitect.services.llm_config_service import (
    delete_config,
    get_all_configs,
    get_all_embedding_configs,
    get_config,
    save_config,
    save_embedding_config,
    validate_api_key,
    validate_embedding_config,
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
    config_dict: dict[str, Any] = {
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


# Embedding config endpoints
@router.get(
    "/embedding-configs",
    response_model=EmbeddingConfigListResponse,
    summary="List all embedding provider configurations",
    description="Retrieve all saved embedding provider configurations (API keys masked)",
)
async def list_embedding_configs(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> EmbeddingConfigListResponse:
    """List all embedding provider configurations."""
    configs = await get_all_embedding_configs(session)

    config_responses = [
        EmbeddingConfigResponse(
            id=str(config.id),
            provider_name=config.provider_name,
            base_url=config.config_data.get("base_url"),
            model=config.config_data.get("model"),
            dimension=config.config_data.get("dimension"),
            is_active=config.is_active,
            created_at=config.created_at.isoformat(),
            updated_at=config.updated_at.isoformat(),
        )
        for config in configs
    ]

    return EmbeddingConfigListResponse(
        configs=config_responses,
        total=len(config_responses),
    )


@router.post(
    "/embedding-configs",
    response_model=EmbeddingConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save embedding provider configuration",
    description="Create or update embedding provider configuration",
)
async def create_embedding_config(
    config_data: EmbeddingConfigCreate,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> EmbeddingConfigResponse:
    """Save or update embedding provider configuration."""
    # Prepare config data dictionary
    config_dict: dict[str, Any] = {
        "is_active": config_data.is_active,
    }

    if config_data.base_url:
        config_dict["base_url"] = config_data.base_url

    if config_data.api_key:
        config_dict["api_key"] = config_data.api_key.get_secret_value()

    if config_data.model:
        config_dict["model"] = config_data.model

    if config_data.dimension:
        config_dict["dimension"] = config_data.dimension

    # Save configuration
    saved_config = await save_embedding_config(
        session, config_data.provider_name, config_dict
    )

    # Prepare response (never expose API key)
    return EmbeddingConfigResponse(
        id=str(saved_config.id),
        provider_name=saved_config.provider_name,
        base_url=saved_config.config_data.get("base_url"),
        model=saved_config.config_data.get("model"),
        dimension=saved_config.config_data.get("dimension"),
        is_active=saved_config.is_active,
        created_at=saved_config.created_at.isoformat(),
        updated_at=saved_config.updated_at.isoformat(),
    )


@router.post(
    "/embedding-configs/validate",
    response_model=LLMProviderConfigValidateResponse,
    summary="Validate embedding provider configuration",
    description="Test connectivity and validate API keys for embedding providers",
)
async def validate_embedding_provider_config(
    validation_data: EmbeddingConfigValidate,
) -> LLMProviderConfigValidateResponse:
    """Validate embedding provider configuration."""
    provider = validation_data.provider_name.lower()

    try:
        if provider in {"ollama", "openai_compatible"}:
            if not validation_data.base_url:
                return LLMProviderConfigValidateResponse(
                    valid=False,
                    message="Validation failed",
                    error=f"base_url is required for {provider} provider",
                )

            is_valid, message = await validate_embedding_config(
                provider,
                base_url=validation_data.base_url,
                model=validation_data.model,
            )
            return LLMProviderConfigValidateResponse(
                valid=is_valid,
                message=message,
                error=None if is_valid else message,
            )

        elif provider in {"openai", "vertex_ai"}:
            if not validation_data.api_key:
                return LLMProviderConfigValidateResponse(
                    valid=False,
                    message="Validation failed",
                    error=f"api_key is required for {provider} provider",
                )

            is_valid, message = await validate_embedding_config(
                provider,
                api_key=validation_data.api_key.get_secret_value(),
                model=validation_data.model,
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


@router.patch(
    "/{provider_name}/toggle",
    response_model=LLMProviderConfigResponse,
    summary="Toggle provider active state",
    description="Enable or disable a saved provider without requiring API key re-entry",
)
async def toggle_llm_config(
    provider_name: str,
    request: LLMProviderToggleRequest,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> LLMProviderConfigResponse:
    """Toggle provider active state without requiring API key.

    This allows users to enable/disable a saved provider configuration
    without needing to re-enter their API key.
    """
    result = await session.execute(
        select(LLMProviderConfig).where(
            LLMProviderConfig.provider_name == provider_name.lower()
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider {provider_name} not configured",
        )

    config.is_active = request.is_active
    await session.flush()
    await session.refresh(config)

    return LLMProviderConfigResponse(
        id=str(config.id),
        provider_name=config.provider_name,
        base_url=config.config_data.get("base_url"),
        model=config.config_data.get("model"),
        is_active=config.is_active,
        created_at=config.created_at.isoformat(),
        updated_at=config.updated_at.isoformat(),
    )


@router.patch(
    "/{provider_name}",
    response_model=LLMProviderConfigResponse,
    summary="Update provider configuration",
    description="Partially update a saved provider configuration. API key is optional - "
    "existing key is preserved if not provided.",
)
async def update_llm_config(
    provider_name: str,
    config_data: LLMProviderConfigUpdate,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> LLMProviderConfigResponse:
    """Update existing provider configuration with partial data.

    This allows users to update model or other settings without re-entering
    their API key. The existing API key is preserved if not provided.
    """
    # Check if config exists
    existing = await get_config(session, provider_name)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider {provider_name} not configured. Use POST to create.",
        )

    # Build update dict with only provided values
    update_dict: dict[str, Any] = {}

    if config_data.base_url is not None:
        update_dict["base_url"] = config_data.base_url

    if config_data.api_key is not None:
        update_dict["api_key"] = config_data.api_key.get_secret_value()

    if config_data.model is not None:
        update_dict["model"] = config_data.model

    if config_data.is_active is not None:
        update_dict["is_active"] = config_data.is_active

    # Save with merge behavior
    saved_config = await save_config(session, provider_name, update_dict)

    return LLMProviderConfigResponse(
        id=str(saved_config.id),
        provider_name=saved_config.provider_name,
        base_url=saved_config.config_data.get("base_url"),
        model=saved_config.config_data.get("model"),
        is_active=saved_config.is_active,
        created_at=saved_config.created_at.isoformat(),
        updated_at=saved_config.updated_at.isoformat(),
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
