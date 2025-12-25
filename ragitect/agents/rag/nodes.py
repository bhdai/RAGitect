"""LangGraph nodes for RAG pipeline.

This module implements the node functions used in the RAG StateGraph.
Each node takes state and returns partial state updates.
"""

from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

from ragitect.agents.rag.schemas import SearchStrategy
from ragitect.prompts.strategy_prompts import (
    build_strategy_prompt,
    format_chat_history_for_strategy,
)

if TYPE_CHECKING:
    from ragitect.agents.rag.state import RAGState


# =============================================================================
# Node Functions
# =============================================================================


async def generate_strategy(
    state: "RAGState",
    *,
    llm: Any = None,
) -> dict:
    """Generate search strategy from user query.

    Analyzes the user's query along with chat history context to produce
    a search strategy with 1-5 targeted search terms for parallel retrieval.

    The LLM resolves pronouns and ambiguous references using conversation
    history, then decomposes complex queries into distinct search aspects.

    Args:
        state: Current RAGState containing messages and original_query
        llm: LLM instance (injected for testing). If None, creates default.

    Returns:
        Dict with 'strategy' (SearchStrategy) and 'llm_calls' (int)
    """
    if llm is None:
        # Default LLM - will be configured via dependency injection in production
        llm = ChatLiteLLM(model="ollama/llama3.1:8b")

    # Build prompt with chat history for pronoun resolution
    query = state["original_query"]
    messages = state.get("messages", [])

    formatted_history = format_chat_history_for_strategy(messages)
    prompt = build_strategy_prompt(query, formatted_history)

    # Use structured output for type-safe strategy generation
    strategy_llm = llm.with_structured_output(SearchStrategy)

    # Generate strategy
    strategy = await strategy_llm.ainvoke([HumanMessage(content=prompt)])

    return {
        "strategy": strategy,
        "llm_calls": 1,
    }
