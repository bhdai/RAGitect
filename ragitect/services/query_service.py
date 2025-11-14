import logging

from langchain_ollama.chat_models import ChatOllama

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
    """Build a few shot prompt for query reformation

    Args:
        user_query: the current user query to reforumlate
        formatted_history: XML formatted chat history string

    Returns:
        str: complete prompt for the LLM
    """
    prompt = f"""You are a query reformulation assistant. Your task is to take a user's current query and the conversation history, then reformulate the query to be self-contained and optimized for semantic search.

**Rules:**
1. Resolve all pronouns (it, that, this, they, etc.) to their actual referents from the history
2. Include necessary context from previous messages to make the query standalone
3. Keep the reformulated query concise (1-2 sentences maximum)
4. Preserve the user's original intent and question type
5. Output ONLY the reformulated query with no preamble or explanation

**Examples:**

Example 1:
History:
<chat_history>
<message role="user">What is FastAPI?</message>
<message role="assistant">FastAPI is a modern Python web framework for building APIs with automatic validation and documentation.</message>
</chat_history>

Current Query: How do I install it?
Reformulated Query: How do I install FastAPI?

---

Example 2:
History:
<chat_history>
<message role="user">Explain Python decorators</message>
<message role="assistant">Decorators are functions that modify the behavior of other functions. They use the @syntax.</message>
<message role="user">That's helpful</message>
</chat_history>

Current Query: Show me an example
Reformulated Query: Show me an example of Python decorators

---

Example 3:
History:
<chat_history>
<message role="user">What's the difference between async and sync in Python?</message>
<message role="assistant">Async functions use async/await for concurrent operations, while sync functions execute sequentially.</message>
<message role="user">Which one is faster?</message>
<message role="assistant">Async is faster for I/O-bound tasks like API calls or database queries.</message>
</chat_history>

Current Query: When should I use it?
Reformulated Query: When should I use async functions in Python instead of synchronous functions?

---

**Now reformulate this query:**

History:
{formatted_history}

Current Query: {user_query}
Reformulated Query:"""

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


def reformulate_query_with_chat_history(
    llm_model: ChatOllama,
    user_query: str,
    chat_history: list[dict[str, str]],
) -> str:
    """Reformulate user query using chat history context for better retrieval

    this function uses an LLM to analyze the conversation history and transform
    the current user query into a optimized query search history that capture
    the full intent, including the context from previous exchanges

    Args:
        llm_model: the LLM model instance
        user_query: the current user query string
        chat_history: list of previous messages

    Returns:
        str: the reformulated query string
    """
    logger.info(f"Reformulating query: {user_query}")

    if not _should_reformulate(user_query, chat_history):
        logger.info("Reformulation skipped - returning original query")
        return user_query

    try:
        # limit the history to last 10 messages for token efficiency
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        formatted_history = format_chat_history(recent_history)

        prompt = _build_reformulation_prompt(user_query, formatted_history)
        logger.debug(f"Prompt length: {len(prompt)} characters")

        # clal the llm
        logger.debug("Calling LLM for reformulation...")
        llm_response = generate_response(llm_model, prompt)

        reformulated = _extract_reformulated_query(llm_response)
        logger.debug(f"Exacted query: '{reformulated}'")

        if not reformulated and not reformulated.strip():
            logger.warning("LLM returned empty reformulated query - using original")
            return user_query

        logger.info(f"Successfully reforumatted: '{user_query}' -> '{reformulated}'")
        return reformulated
    except Exception as e:
        # fail-safe
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
