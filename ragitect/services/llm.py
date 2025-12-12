import logging
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
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


async def validate_llm_config(config: LLMConfig) -> tuple[bool, str]:
    """Validate llm model based on config (async)

    Args:
        config: LLM configuration
    Returns:
        tuple[bool, str]: is valid, error message
    """
    try:
        if config.custom_provider:
            model_str = f"{config.custom_provider}/{config.model}"
        else:
            provider_name = config.provider.lower()
            logger.info(
                f"Creaeting LLM with provider={provider_name}, model={config.model}"
            )

            model_str = f"{provider_name}/{config.model}"

        llm_kwargs = {
            "model": model_str,
            "temperature": config.temperature,
            "api_key": config.api_key,
            "max_tokens": config.max_tokens,
            "request_timeout": config.timeout,
        }

        if config.base_url:
            llm_kwargs["api_base"] = config.base_url

        llm = ChatLiteLLM(**llm_kwargs)  # pyright: ignore[reportArgumentType]

        # make a test call
        test_message = HumanMessage(content="Hello there")
        response = await llm.ainvoke([test_message])
        if response and response.content:
            logger.info(f"Successfully validated LLM model: {model_str}")
            return True, ""
        else:
            logger.warning(f"LLM config validation return empty response: {model_str}")
            return False, "LLM returned empty response during validation"
    except Exception as e:
        error_msg = f"Failed to validate LLM configuration: {e!s}"
        logger.error(error_msg)
        return False, error_msg


async def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create llm model based on config (async)

    Args:
        config: LLM configuration
    Returns:
        BaseChatModel: langchain compatible chat model instance
    """
    # Future: await config loading from DB here (Story 1.4)
    if config.custom_provider:
        model_str = f"{config.custom_provider}/{config.model}"
    else:
        provider_name = config.provider.lower()
        logger.info(
            f"Creaeting LLM with provider={provider_name}, model={config.model}"
        )

        model_str = f"{provider_name}/{config.model}"

    llm_kwargs = {
        "model": model_str,
        "temperature": config.temperature,
        "api_key": config.api_key,
        "max_tokens": config.max_tokens,
        "request_timeout": config.timeout,
    }

    if config.base_url:
        llm_kwargs["api_base"] = config.base_url

    llm = ChatLiteLLM(**llm_kwargs)  # pyright: ignore[reportArgumentType]
    return llm


async def generate_response(
    llm_model: BaseChatModel, messages: Sequence[BaseMessage]
) -> str:
    """Generate response from llm (async)

    Args:
        llm_model: llm model
        messages: list of langchain message objects

    Returns:
        response string
    """
    logger.debug(f"Sending {len(messages)} messages to LLM...")
    ai_msg = await llm_model.ainvoke(messages)
    return ai_msg.content  # pyright: ignore[reportReturnType]


async def generate_response_stream(
    llm_model: BaseChatModel, messages: Sequence[BaseMessage]
) -> AsyncGenerator[str, None]:
    """Generate streaming response from llm (async generator)

    Args:
        llm_model: llm model
        messages: list of langchain message objects

    Yields:
        response string chunks
    """
    print("Streaming response from LLM...")
    async for chunk in llm_model.astream(messages):
        yield chunk.content  # pyright: ignore[reportReturnType]


async def generate_response_with_prompt(llm_model: BaseChatModel, prompt: str) -> str:
    """Generate response from LLM using simple prompt string (async)

    Convenience function for backward compatibility with existing code.
    Converts string prompt to HumanMessage internally.

    Args:
        llm_model: LangChain chat model instance
        prompt: Simple prompt string

    Returns:
        str: Generated response content
    """
    messages = [HumanMessage(content=prompt)]
    return await generate_response(llm_model, messages)


async def get_active_llm_config(session: "AsyncSession") -> LLMConfig:
    """Load active LLM configuration from database.

    Queries the database for an active LLM provider configuration and
    constructs an LLMConfig from it. Falls back to default Ollama config
    if no active configuration exists.

    Args:
        session: Async database session

    Returns:
        LLMConfig: Configuration loaded from DB or default config

    Note:
        This function is the bridge between stored configs and LLM creation.
    """
    # Import here to avoid circular dependency
    from ragitect.services.llm_config_service import get_active_config

    try:
        config = await get_active_config(session)

        if config:
            provider = config.provider_name
            config_data = config.config_data

            logger.info(f"Using active LLM provider configuration: {provider}")

            return LLMConfig(
                provider=provider,
                model=config_data.get("model", "llama3.1:8b"),
                temperature=config_data.get("temperature", 0.7),
                base_url=config_data.get("base_url"),
                api_key=config_data.get("api_key"),
                max_tokens=config_data.get("max_tokens"),
                timeout=config_data.get("timeout", 60),
            )

        logger.info("No active LLM configuration found, using default Ollama config")
        return get_default_config()

    except Exception as e:
        logger.warning(
            f"Failed to load LLM config from database: {e}, using default config"
        )
        return get_default_config()


async def create_llm_from_db(session: "AsyncSession") -> BaseChatModel:
    """Create LLM model using active configuration from database.

    Loads the active LLM provider configuration from the database and
    creates a LangChain-compatible chat model. Falls back to default
    Ollama configuration if no active config exists.

    Args:
        session: Async database session

    Returns:
        BaseChatModel: Configured LangChain chat model instance

    Note:
        an LLM configured based on user preferences stored in the database.
    """
    config = await get_active_llm_config(session)
    return await create_llm(config)
