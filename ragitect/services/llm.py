"""LLM service for RAGitect.

Provides LLM client creation, validation, and response generation
with proper error handling and structured logging.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator, Sequence
from enum import Enum
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_litellm import ChatLiteLLM

from ragitect.services.config import LLMConfig, get_default_config

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


# =============================================================================
# Private Helpers (DRY extraction)
# =============================================================================


def _build_litellm_kwargs(config: LLMConfig) -> dict:
    """Build kwargs for ChatLiteLLM instantiation.

    Centralizes model string and kwargs building to avoid duplication
    between create_llm() and validate_llm_config().

    Args:
        config: LLM configuration object

    Returns:
        dict: Kwargs ready for ChatLiteLLM constructor
    """
    if config.custom_provider:
        model_str = f"{config.custom_provider}/{config.model}"
    else:
        model_str = f"{config.provider.lower()}/{config.model}"

    kwargs: dict = {
        "model": model_str,
        "temperature": config.temperature,
        "api_key": config.api_key,
        "max_tokens": config.max_tokens,
        "request_timeout": config.timeout,
    }

    if config.base_url:
        kwargs["api_base"] = config.base_url

    return kwargs


def _log_llm_metrics(operation: str, latency_ms: float, success: bool, **extra) -> None:
    """Log LLM operation metrics in structured JSON format.

    Follows Story 2.5 pattern for structured logging.

    Args:
        operation: Type of LLM operation (validate, generate, stream)
        latency_ms: Operation duration in milliseconds
        success: Whether the operation succeeded
        **extra: Additional context fields
    """
    metrics = {
        "operation": operation,
        "latency_ms": round(latency_ms, 2),
        "success": success,
        **extra,
    }
    logger.info(f"LLM_METRICS: {json.dumps(metrics)}")


# =============================================================================
# Public API - LLM Creation and Validation
# =============================================================================


async def validate_llm_config(config: LLMConfig) -> tuple[bool, str]:
    """Validate LLM model configuration by making a test call.

    Args:
        config: LLM configuration to validate

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    start_time = time.time()

    try:
        kwargs = _build_litellm_kwargs(config)
        model_str = kwargs["model"]

        logger.info(
            f"Validating LLM config: provider={config.provider}, model={config.model}"
        )

        llm = ChatLiteLLM(**kwargs)
        test_message = HumanMessage(content="Hello")
        response = await llm.ainvoke([test_message])

        latency_ms = (time.time() - start_time) * 1000

        if response and response.content:
            _log_llm_metrics("validate", latency_ms, True, model=model_str)
            logger.info(f"Successfully validated LLM: {model_str}")
            return True, ""
        else:
            _log_llm_metrics(
                "validate", latency_ms, False, model=model_str, reason="empty_response"
            )
            return False, "LLM returned empty response during validation"

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        error_msg = f"Failed to validate LLM configuration: {e!s}"
        _log_llm_metrics("validate", latency_ms, False, error=str(e))
        logger.error(error_msg)
        return False, error_msg


async def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create LLM model from configuration.

    Args:
        config: LLM configuration

    Returns:
        BaseChatModel: LangChain-compatible chat model instance
    """
    kwargs = _build_litellm_kwargs(config)
    logger.info(f"Creating LLM: {kwargs['model']}")
    return ChatLiteLLM(**kwargs)


# =============================================================================
# Public API - Response Generation
# =============================================================================


async def generate_response(
    llm_model: BaseChatModel,
    messages: Sequence[BaseMessage],
) -> str:
    """Generate response from LLM.

    Args:
        llm_model: LangChain chat model instance
        messages: Sequence of messages to send

    Returns:
        str: Generated response content
    """
    start_time = time.time()
    logger.debug(f"Generating response with {len(messages)} messages")

    try:
        ai_msg = await llm_model.ainvoke(messages)
        latency_ms = (time.time() - start_time) * 1000

        content = str(ai_msg.content) if ai_msg.content else ""
        _log_llm_metrics("generate", latency_ms, True, message_count=len(messages))

        return content

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        _log_llm_metrics("generate", latency_ms, False, error=str(e))
        raise


async def generate_response_stream(
    llm_model: BaseChatModel,
    messages: Sequence[BaseMessage],
) -> AsyncGenerator[str, None]:
    """Generate streaming response from LLM.

    Args:
        llm_model: LangChain chat model instance
        messages: Sequence of messages to send

    Yields:
        str: Response content chunks

    Note:
        Includes error handling and logging (fixed from legacy version).
    """
    start_time = time.time()
    chunk_count = 0

    logger.info(f"Starting streaming response with {len(messages)} messages")

    try:
        async for chunk in llm_model.astream(messages):
            chunk_count += 1
            content = str(chunk.content) if chunk.content else ""
            if content:
                yield content

        latency_ms = (time.time() - start_time) * 1000
        _log_llm_metrics("stream", latency_ms, True, chunks=chunk_count)
        logger.info(f"Streaming complete: {chunk_count} chunks")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        _log_llm_metrics("stream", latency_ms, False, chunks=chunk_count, error=str(e))
        logger.error(f"Streaming failed after {chunk_count} chunks: {e}")
        raise


async def generate_response_with_prompt(
    llm_model: BaseChatModel,
    prompt: str,
) -> str:
    """Generate response from simple prompt string.

    Convenience wrapper that converts prompt to HumanMessage.

    Args:
        llm_model: LangChain chat model instance
        prompt: Simple prompt string

    Returns:
        str: Generated response content
    """
    messages = [HumanMessage(content=prompt)]
    return await generate_response(llm_model, messages)


# =============================================================================
# Database Integration
# =============================================================================


async def get_active_llm_config(session: "AsyncSession") -> LLMConfig:
    """Load active LLM configuration from database.

    Falls back to default Ollama config if no active config exists.

    Args:
        session: Async database session

    Returns:
        LLMConfig: Configuration from DB or default
    """
    # Import here to avoid circular dependency
    # TODO: Consider restructuring to eliminate this
    from ragitect.services.llm_config_service import get_active_config

    try:
        config = await get_active_config(session)

        if config:
            provider = config.provider_name
            config_data = config.config_data

            logger.info(f"Using active LLM config: {provider}")

            return LLMConfig(
                provider=provider,
                model=config_data.get("model", "llama3.1:8b"),
                temperature=config_data.get("temperature", 0.7),
                base_url=config_data.get("base_url"),
                api_key=config_data.get("api_key"),
                max_tokens=config_data.get("max_tokens"),
                timeout=config_data.get("timeout", 60),
            )

        logger.info("No active LLM config found, using default")
        return get_default_config()

    except Exception as e:
        logger.warning(f"Failed to load LLM config from DB: {e}, using default")
        return get_default_config()


async def create_llm_from_db(session: "AsyncSession") -> BaseChatModel:
    """Create LLM model using active database configuration.

    Args:
        session: Async database session

    Returns:
        BaseChatModel: Configured LangChain chat model
    """
    config = await get_active_llm_config(session)
    return await create_llm(config)
