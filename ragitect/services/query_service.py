import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ragitect.services.llm import generate_response

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Schema for Structured LLM Output
# =============================================================================


class QueryReformulationResponse(BaseModel):
    """Structured response for query reformulation.

    JSON schema for LLM to return structured output.
    The 'reasoning' field captures LLM explanation (discarded in output).
    """

    reasoning: str
    reformulated_query: str
    was_modified: bool


# =============================================================================
# Response Parsing
# =============================================================================


def _parse_reformulation_response(response: str, original_query: str) -> str:
    """Parse JSON response with fallback to regex cleanup.

    Primary: JSON parsing using Pydantic validation
    Fallback: Regex cleanup if LLM returns plain text

    Args:
        response: Raw LLM response string
        original_query: Original query for fallback

    Returns:
        str: Cleaned reformulated query
    """
    # Handle empty/whitespace-only responses
    if not response or not response.strip():
        logger.warning("Empty response - returning original query")
        return original_query

    # Strategy: Find JSON object by locating balanced braces
    def extract_json_object(text: str) -> str | None:
        """Extract first complete JSON object from text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    # Try to extract and validate JSON using Pydantic
    json_str = extract_json_object(response)
    if json_str:
        try:
            parsed = QueryReformulationResponse.model_validate_json(json_str)
            query = parsed.reformulated_query.strip()
            if query:
                logger.debug(
                    f"JSON parse successful, was_modified={parsed.was_modified}"
                )
                return query
            else:
                logger.warning("Pydantic parsed but reformulated_query empty")
        except Exception as e:
            logger.debug(f"Pydantic validation failed ({e}), using fallback")

    # Fallback: regex cleanup for plain text responses
    cleaned = response.strip()

    # Remove parenthetical explanations at the end
    cleaned = re.sub(r"\n+\s*\([^)]+\)\s*$", "", cleaned, flags=re.DOTALL)

    # Remove leading/trailing quotes
    cleaned = cleaned.strip().strip("\"'")

    # Remove common prefixes
    cleaned = re.sub(
        r"^(Output|Query|Reformulated|Here is|The reformulated query is):\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.strip()

    if cleaned:
        logger.debug("Fallback regex cleanup successful")
        return cleaned

    logger.warning("All parsing failed, returning original query")
    return original_query


def _classify_query_complexity(
    user_query: str,
    chat_history: list[dict[str, str]],
) -> str:
    """Classify query as 'simple', 'ambiguous', or 'complex'.

    Uses heuristic-based classification to determine optimal processing path:
    - 'simple': Direct search, no reformulation needed
    - 'ambiguous': Contains pronouns or context refs, needs reformulation
    - 'complex': Requires decomposition into sub-queries (comparison queries)

    Args:
        user_query: The current user query string
        chat_history: List of previous messages with 'role' and 'content' keys

    Returns:
        str: One of 'simple', 'ambiguous', or 'complex'

    Examples:
        >>> _classify_query_complexity("What is Python?", [])
        'simple'
        >>> _classify_query_complexity("How do I install it?", [{"role": "user", "content": "Tell me about FastAPI"}])
        'ambiguous'
        >>> _classify_query_complexity("Compare FastAPI vs Flask", [])
        'complex'
    """
    query_lower = user_query.lower()

    # Check for complex patterns FIRST (highest priority)
    # Complex queries require decomposition into sub-queries
    complex_patterns = [
        "compare",
        "difference between",
        " vs ",
        " versus ",
        "both",
        "all of",
        "each of",
    ]
    is_complex = any(pattern in query_lower for pattern in complex_patterns)

    if is_complex:
        logger.debug("Query classified as complex")
        return "complex"

    # No chat history = simple query (no context to resolve)
    if not chat_history:
        logger.debug("Query classified as simple (no history)")
        return "simple"

    # Check for pronouns that need resolution (ambiguous)
    # Normalize query for word boundary detection
    query_normalized = re.sub(
        r"[^\w\s]", " ", query_lower
    )  # Replace punctuation with spaces
    query_words = set(query_normalized.split())
    pronouns = {"it", "that", "this", "those", "these", "they", "them"}
    has_pronouns = bool(query_words & pronouns)

    # Check for context references (ambiguous)
    context_refs = ["the previous", "earlier", "before", "above", "again"]
    has_context_ref = any(ref in query_lower for ref in context_refs)

    if has_pronouns or has_context_ref:
        logger.debug("Query classified as ambiguous")
        return "ambiguous"

    # Default to simple if no ambiguous markers found
    logger.debug("Query classified as simple")
    return "simple"


async def adaptive_query_processing(
    llm_model: BaseChatModel,
    user_query: str,
    chat_history: list[dict[str, str]],
) -> str:
    """Process query adaptively based on complexity classification.

    Routes queries to appropriate processing path:
    - 'simple': Return original query (skip reformulation)
    - 'ambiguous': Reformulate with chat history context
    - 'complex': Reformulate (future: decompose into sub-queries)

    Args:
        llm_model: The LLM model instance for reformulation
        user_query: The current user query string
        chat_history: List of previous messages with 'role' and 'content' keys

    Returns:
        str: The processed query (original or reformulated)
    """
    complexity = _classify_query_complexity(user_query, chat_history)
    logger.info(f"Query classified as: {complexity}")

    if complexity == "simple":
        logger.info("Using original query (simple)")
        return user_query

    elif complexity == "ambiguous":
        logger.info("Reformulating query (ambiguous)")
        return await reformulate_query_with_chat_history(
            llm_model, user_query, chat_history
        )

    else:  # complex
        logger.info("Complex query detected - using reformulation for now")
        return await reformulate_query_with_chat_history(
            llm_model, user_query, chat_history
        )


async def _grade_retrieval_relevance(
    llm_model: BaseChatModel,
    query: str,
    retrieved_docs: list[str],
) -> bool:
    """Grade if retrieved documents are relevant to the query.

    Uses LLM to assess whether the top retrieved document contains
    information relevant to answering the user's query.

    Implements fail-open pattern: returns True on any error to avoid
    blocking the user's query due to grading failures.

    Args:
        llm_model: The LLM model instance for grading
        query: The user's query string
        retrieved_docs: List of retrieved document content strings

    Returns:
        bool: True if relevant (or on error), False if irrelevant
    """
    # Empty docs = fail open (assume relevant to proceed)
    if not retrieved_docs:
        logger.warning("No documents to grade - failing open")
        return True

    # Take first doc (top result) - first 500 chars
    doc_sample = retrieved_docs[0][:500]

    prompt = f"""You are a relevance grader. Assess if the retrieved document contains information related to the user's query.

Query: {query}

Retrieved Document:
{doc_sample}

Is this document relevant to answering the query?
Answer only 'yes' or 'no'.

Relevant:"""

    try:
        response = await generate_response(
            llm_model, messages=[HumanMessage(content=prompt)]
        )

        grade = response.strip().lower()
        is_relevant = grade == "yes" or grade.startswith("yes")

        logger.info(f"Relevance grade: {grade}")
        return is_relevant

    except Exception as e:
        logger.error(f"Grading failed: {e}")
        return True  # Fail open - assume relevant


def log_query_metrics(metadata: dict) -> None:
    """Log query metrics in structured JSON format for analysis.

    Sanitizes PII by excluding raw query text from logs.
    Only logs non-sensitive metrics: classification, timing, flags.

    Args:
        metadata: Dictionary containing query processing metrics
    """
    # Sanitize: exclude raw query text (PII/privacy concern)
    safe_metadata = {
        "classification": metadata.get("classification"),
        "used_reformulation": metadata.get("used_reformulation"),
        "grade": metadata.get("grade"),
        "latency_ms": metadata.get("latency_ms"),
        "original_query_length": len(metadata.get("original_query", "")),
        "final_query_length": len(metadata.get("final_query", "")),
    }
    logger.info(f"QUERY_METRICS: {json.dumps(safe_metadata)}")


async def query_with_iterative_fallback(
    llm_model: BaseChatModel,
    user_query: str,
    chat_history: list[dict[str, str]],
    vector_search_fn: Callable,
) -> tuple[list[str], dict]:
    """Query with automatic fallback to reformulation if needed.

    Implements the router pattern with iterative fallback:
    1. Classify query complexity
    2. For simple queries: try direct search, check relevance, fallback if poor
    3. For ambiguous/complex: reformulate directly

    Args:
        llm_model: The LLM model instance for reformulation and grading
        user_query: The user's original query string
        chat_history: List of previous messages with 'role' and 'content' keys
        vector_search_fn: Async callable that performs vector search

    Returns:
        tuple: (retrieved_docs: list[str], metadata: dict)
        metadata keys: used_reformulation, original_query, final_query,
                      classification, grade, latency_ms
    """
    metadata = {
        "used_reformulation": False,
        "original_query": user_query,
        "final_query": user_query,
        "classification": None,
        "grade": None,
        "latency_ms": 0,
    }

    start_time = time.time()

    # Step 1: Classify query
    classification = _classify_query_complexity(user_query, chat_history)
    metadata["classification"] = classification

    # Step 2: For simple queries, try direct search first
    if classification == "simple":
        logger.info("Attempt 1: Direct search (simple query)")
        results = await vector_search_fn(user_query)

        # Grade the results
        is_relevant = await _grade_retrieval_relevance(llm_model, user_query, results)
        metadata["grade"] = "yes" if is_relevant else "no"

        if is_relevant:
            logger.info("Direct search successful - good relevance")
            metadata["latency_ms"] = (time.time() - start_time) * 1000
            log_query_metrics(metadata)
            return results, metadata

        # Fallback to reformulation
        logger.info("Attempt 2: Reformulation triggered by low relevance")
        reformulated_query = await reformulate_query_with_chat_history(
            llm_model, user_query, chat_history
        )

        metadata["used_reformulation"] = True
        metadata["final_query"] = reformulated_query

        results = await vector_search_fn(reformulated_query)
        metadata["latency_ms"] = (time.time() - start_time) * 1000
        log_query_metrics(metadata)

        return results, metadata

    else:  # ambiguous or complex - reformulate first
        logger.info("Direct reformulation (ambiguous/complex query)")
        reformulated_query = await reformulate_query_with_chat_history(
            llm_model, user_query, chat_history
        )

        metadata["used_reformulation"] = True
        metadata["final_query"] = reformulated_query

        results = await vector_search_fn(reformulated_query)
        metadata["latency_ms"] = (time.time() - start_time) * 1000
        log_query_metrics(metadata)

        return results, metadata


def _build_reformulation_prompt(user_query: str, formatted_history: str) -> str:
    """Build a guarded prompt for query reformulation with JSON output.

    Uses research-backed guardrails to prevent over-reformulation:
    - Returns query unchanged if already self-contained
    - Only replaces pronouns that genuinely need history context
    - Never adds information not explicitly in history
    - Requests JSON structured output for reliable parsing

    Args:
        user_query: the current user query to reformulate
        formatted_history: XML formatted chat history string

    Returns:
        str: complete prompt for the LLM
    """
    prompt = f"""You are a query preprocessor for semantic search. Your task is to make queries self-contained when needed.

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{"reasoning": "brief explanation", "reformulated_query": "the query", "was_modified": true/false}}

CRITICAL RULES:
1. If the query is ALREADY SELF-CONTAINED, return it UNCHANGED with was_modified=false
   - Pronouns in the SAME SENTENCE as their referent do NOT need reformulation
   - "What is Quickshell and how do I install it?" → "it" refers to "Quickshell" in the same sentence → UNCHANGED

2. ONLY replace pronouns if they refer to something from PREVIOUS CONVERSATION TURNS that cannot be understood from the current query alone

3. NEVER add information that is not explicitly stated in the conversation history
4. NEVER invent or assume context - only use what is clearly present
5. Keep reformulated queries concise (1-2 sentences max)

<example>
History: (empty)
Query: "What is Quickshell and how do I install it on Ubuntu?"
Output: {{"reasoning": "Query is self-contained. The pronoun 'it' refers to 'Quickshell' within the same sentence.", "reformulated_query": "What is Quickshell and how do I install it on Ubuntu?", "was_modified": false}}
</example>

<example>
History: User asked about FastAPI
Query: "How do I install it?"
Output: {{"reasoning": "The pronoun 'it' refers to FastAPI from conversation history.", "reformulated_query": "How do I install FastAPI?", "was_modified": true}}
</example>

<example>
History: (empty)
Query: "Tell me about Python and its features"
Output: {{"reasoning": "Query is self-contained. The pronoun 'its' refers to 'Python' within the same sentence.", "reformulated_query": "Tell me about Python and its features", "was_modified": false}}
</example>

<example>
History: User discussed PostgreSQL setup
Query: "Is it faster than MySQL?"
Output: {{"reasoning": "The pronoun 'it' refers to PostgreSQL from conversation history.", "reformulated_query": "Is PostgreSQL faster than MySQL?", "was_modified": true}}
</example>

<example>
History: (empty)
Query: "Compare React and Vue"
Output: {{"reasoning": "Query is self-contained with no ambiguous references.", "reformulated_query": "Compare React and Vue", "was_modified": false}}
</example>

Conversation History:
{formatted_history}

Current Query: {user_query}

Output:"""

    return prompt


async def reformulate_query_with_chat_history(
    llm_model: BaseChatModel,
    user_query: str,
    chat_history: list[dict[str, str]],
) -> str:
    """Reformulate user query using chat history context for better retrieval (async)

    This function uses an LLM to analyze the conversation history and transform
    the current user query into an optimized query for semantic search that captures
    the full intent, including context from previous exchanges.

    Note: Callers should use _classify_query_complexity() or adaptive_query_processing()
    to decide WHEN to call this function. This function always attempts reformulation.

    Args:
        llm_model: the LLM model instance
        user_query: the current user query string
        chat_history: list of previous messages

    Returns:
        str: the reformulated query string (or original on failure)
    """
    start_time = time.time()
    logger.info("Starting query reformulation")

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

        # Parse structured JSON response with fallback
        reformulated = _parse_reformulation_response(llm_response, user_query)
        logger.debug(f"Reformulated query length: {len(reformulated)} chars")

        if not reformulated:
            logger.warning("LLM returned empty reformulated query - using original")
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_length": len(user_query),
                "reformulated": False,
                "reason": "empty_response",
                "latency_ms": (time.time() - start_time) * 1000,
                "llm_latency_ms": llm_latency,
                "prompt_length": prompt_length,
                "estimated_tokens": estimated_tokens,
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
            "output_length": len(reformulated),
        }
        logger.info(f"QueryMetrics: {json.dumps(metrics)}")
        logger.info("Successfully reformulated query")
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
