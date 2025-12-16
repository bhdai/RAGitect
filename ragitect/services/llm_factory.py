"""LLM factory functions for database integration.

This module provides functions that create LLM instances from database
configurations. It exists as a separate module to avoid circular imports
between llm.py and llm_config_service.py:

- llm.py defines core LLM functions (create_llm, validate_llm_config, etc.)
- llm_config_service.py imports validate_llm_config from llm.py
- This module imports from both without creating a cycle

Import chain:
  llm_config_service.py -> llm.py (for validate_llm_config)
  llm_factory.py -> llm.py (for create_llm, LLMConfig)
  llm_factory.py -> llm_config_service.py (for get_active_config)

Story 3.0: Streaming Infrastructure - Circular import fix
"""

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from ragitect.services.config import LLMConfig, get_default_config
from ragitect.services.llm import create_llm
from ragitect.services.llm_config_service import get_active_config

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def get_active_llm_config(session: "AsyncSession") -> LLMConfig:
    """Load active LLM configuration from database.

    Falls back to default Ollama config if no active config exists.

    Args:
        session: Async database session

    Returns:
        LLMConfig: Configuration from DB or default
    """
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
        import traceback

        logger.warning(
            f"Failed to load LLM config from DB: {type(e).__name__}: {e!r}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
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
