import json
import logging
import time
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ragitect.services.llm import generate_response

# from llm import generate_response, create_llm

logger = logging.getLogger(__name__)


def _should_reformulate(user_query: str, chat_history: list[dict[str, str]]) -> bool:
    """Determine if query reformulation is necessary

    Args:
        user_query: the current user query
        chat_history: list of previous message

    Returns:
        bool: True if the reformulation should be perform
    """
    if not chat_history:
        logger.debug("Skipping reformulation: empty chat history")
        return False

    # very short query like a single word
    if len(user_query.split()) <= 1:
        logger.debug("Skipping reformulation: user query too short")
        return False

    # query is already very long and specific
    if len(user_query.split()) > 50:
        logger.debug("Skipping reformulation: query already very specific")
        return False

    return True


def _build_reformulation_prompt(user_query: str, formatted_history: str) -> str:
    """Build a simplified prompt for query reformulation

    Phase 1 Hotfix: Reduced from ~2350 chars to ~400 chars (83% reduction)
    Removed few-shot examples that were causing LLM confusion.

    Args:
        user_query: the current user query to reformulate
        formatted_history: XML formatted chat history string

    Returns:
        str: complete prompt for the LLM
    """
    # Simplified prompt - direct instructions only, no examples
    prompt = f"""Reformulate this query to be self-contained for semantic search.

Rules:
1. Replace pronouns (it/that/this) with their referents from history
2. Add essential context to make the query standalone
3. Keep it concise (1-2 sentences max)
4. Output ONLY the reformulated query

History:
{formatted_history}

Query: {user_query}
Reformulated:"""

    return prompt


def _extract_reformulated_query(llm_response: str) -> str:
    response = llm_response.strip()

    # remove common prefixes
    prefixes_to_remove = [
        "Reformulated Query:",
        "Reformulated:",
        "Query:",
        "Here's the reformulated query:",
        "The reformulated query is:",
    ]

    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix) :].strip()
            break

    # remove quotes if the entire response is wrapped in them
    if response.startswith(('"', "'")) and response.endswith(('"', "'")):
        response = response[1:-1].strip()

    return response


def _validate_reformulated_query(query: str) -> bool:
    """Validate that LLM output is a query, not an answer

    Uses heuristic-based detection to catch common patterns where the LLM
    generates an answer instead of reformulating the query.

    Phase 1 Hotfix: Added to prevent answer generation bug.

    Args:
        query: The reformulated query string to validate

    Returns:
        bool: True if valid query, False if appears to be an answer

    Examples of INVALID outputs (returns False):
        - "FastAPI is a modern Python web framework for building APIs."
        - "You should use async functions because they are faster."
        - "This means that decorators modify function behavior."

    Examples of VALID outputs (returns True):
        - "What is FastAPI?"
        - "How do I use async functions in Python?"
        - "Explain Python decorators"
    """
    if not query or not query.strip():
        return False

    query_lower = query.lower().strip()

    # Pattern 1: Explanation markers (strong signal of answer)
    answer_markers = [
        "because",
        "this means",
        "it is",
        "they are",
        "it works by",
        "you should",
        "you can",
        "you need to",
        "this is",
        "that is",
    ]
    for marker in answer_markers:
        if marker in query_lower:
            logger.warning(
                f"Validation failed: answer marker '{marker}' detected in output"
            )
            return False

    # Pattern 2: Definitional structure (subject + "is a" + definition)
    # Example: "FastAPI is a modern framework..."
    if " is a " in query_lower or " are " in query_lower:
        # Check if it's a statement form (no question mark, ends with period)
        if "?" not in query and query.endswith("."):
            logger.warning(
                "Validation failed: definitional structure detected (is a/are + period)"
            )
            return False

    # Pattern 3: Long statement without question markers
    # Heuristic: >80 chars, no question words, ends with period = likely answer
    question_words = ["what", "how", "why", "when", "where", "which", "who", "explain"]
    has_question_word = any(word in query_lower for word in question_words)

    if len(query) > 80 and not has_question_word and query.endswith("."):
        logger.warning(
            "Validation failed: long statement without question markers (>80 chars, no question words, ends with period)"
        )
        return False

    # Pattern 4: Starts with capital statement (not question form)
    # Example: "Decorators are functions..." vs "What are decorators?"
    if query.endswith(".") and not has_question_word and query[0].isupper():
        # Check if it looks like a statement sentence
        words = query.split()
        if len(words) > 5:  # Long enough to be a statement
            logger.warning(
                "Validation failed: statement form detected (capital start, >5 words, ends with period, no question words)"
            )
            return False

    # Passed all validation checks
    return True


async def reformulate_query_with_chat_history(
    llm_model: BaseChatModel,
    user_query: str,
    chat_history: list[dict[str, str]],
) -> str:
    """Reformulate user query using chat history context for better retrieval (async)

    This function uses an LLM to analyze the conversation history and transform
    the current user query into an optimized query search history that capture
    the full intent, including the context from previous exchanges

    Phase 1 Hotfix: Added validation and metrics logging.

    Args:
        llm_model: the LLM model instance
        user_query: the current user query string
        chat_history: list of previous messages

    Returns:
        str: the reformulated query string
    """
    start_time = time.time()
    logger.info(f"Reformulating query: {user_query}")

    if not _should_reformulate(user_query, chat_history):
        # Log metrics for skipped reformulation
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_length": len(user_query),
            "reformulated": False,
            "reason": "skipped_by_should_reformulate",
            "latency_ms": (time.time() - start_time) * 1000,
        }
        logger.info(f"QueryMetrics: {json.dumps(metrics)}")
        logger.info("Reformulation skipped - returning original query")
        return user_query

    try:
        # limit the history to last 10 messages for token efficiency
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        formatted_history = format_chat_history(recent_history)

        prompt = _build_reformulation_prompt(user_query, formatted_history)
        prompt_length = len(prompt)
        estimated_tokens = prompt_length // 4  # rough estimate

        logger.debug(f"Prompt length: {prompt_length} characters")
        logger.debug(f"Estimated tokens: {estimated_tokens}")

        # call the llm
        logger.debug("Calling LLM for reformulation...")
        llm_start = time.time()
        human_message = HumanMessage(content=prompt)
        llm_response = await generate_response(llm_model, messages=[human_message])
        llm_latency = (time.time() - llm_start) * 1000

        reformulated = _extract_reformulated_query(llm_response)
        logger.debug(f"Extracted query: '{reformulated}'")

        # Phase 1: Validate output to catch answer generation
        validation_passed = _validate_reformulated_query(reformulated)

        if not reformulated or not reformulated.strip() or not validation_passed:
            if not reformulated or not reformulated.strip():
                reason = "empty_response"
                logger.warning("LLM returned empty reformulated query - using original")
            else:
                reason = "validation_failed"
                logger.warning(
                    f"Validation failed for reformulated query: '{reformulated}' - using original"
                )

            # Log metrics for failed validation
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_length": len(user_query),
                "reformulated": False,
                "reason": reason,
                "latency_ms": (time.time() - start_time) * 1000,
                "llm_latency_ms": llm_latency,
                "prompt_length": prompt_length,
                "estimated_tokens": estimated_tokens,
                "validation_passed": validation_passed,
            }
            logger.info(f"QueryMetrics: {json.dumps(metrics)}")
            return user_query

        # Success - log metrics
        total_latency = (time.time() - start_time) * 1000
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_length": len(user_query),
            "reformulated": True,
            "latency_ms": total_latency,
            "llm_latency_ms": llm_latency,
            "prompt_length": prompt_length,
            "estimated_tokens": estimated_tokens,
            "validation_passed": True,
            "output_length": len(reformulated),
        }
        logger.info(f"QueryMetrics: {json.dumps(metrics)}")
        logger.info(f"Successfully reformulated: '{user_query}' -> '{reformulated}'")
        return reformulated

    except Exception as e:
        # fail-safe with metrics
        total_latency = (time.time() - start_time) * 1000
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_length": len(user_query),
            "reformulated": False,
            "reason": "exception",
            "error": str(e),
            "latency_ms": total_latency,
        }
        logger.info(f"QueryMetrics: {json.dumps(metrics)}")
        logger.error(f"Error during query reformulation: {e}", exc_info=True)
        logger.error("Falling back to original query")
        return user_query


def format_chat_history(chat_history: list[dict[str, str]]) -> str:
    """format chat history into a structure string for LLM context

    Convert the chat history list into a readable format that LLM can understand.
    Uses XML-like tags for clear structure

    Args:
        chat_history: list of messages dictionaries with 'role' and 'context' keys

    Returns:
        str: formated chat history string

    Example:
        >>> history = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> print(format_chat_history(history))
        <chat_history>
        <message role="user">Hello</message>
        <message role="assistant">Hi there!</message>
        </chat_history>
    """
    if not chat_history:
        return "<chat_history>\n</chat_history>"

    lines = ["<chat_history>"]
    for idx, message in enumerate(chat_history):
        if "role" not in message:
            raise ValueError(f"Message at index {idx} missing 'role' key")

        if "content" not in message:
            raise ValueError(f"Message at index {idx} missing 'content' key")

        role = message["role"]
        content = message["content"]

        lines.append(f'<message role="{role}">{content}</message>')

    lines.append("</chat_history>")

    return "\n".join(lines)
