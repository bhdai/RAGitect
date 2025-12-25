"""Strategy generation prompts for RAG agent query decomposition.

Components:
- Search strategy generation: Decompose queries into parallel search terms
- Pronoun resolution: Context-aware reference resolution from chat history
"""

from langchain_core.messages import HumanMessage

# =============================================================================
# Strategy Generation Prompt Components
# =============================================================================

STRATEGY_GENERATION_INSTRUCTIONS = """
You are a search strategy generator for a RAG (Retrieval-Augmented Generation) system.
Your task is to analyze the user's query and generate an optimal search strategy with targeted search terms.
"""

STRATEGY_OUTPUT_FORMAT = """
OUTPUT FORMAT:
You MUST return a JSON object with this EXACT structure:

{
  "reasoning": "Brief explanation of your strategy approach",
  "searches": [
    {
      "term": "Optimized search query for semantic similarity",
      "reasoning": "Why this search term will find relevant documents"
    }
  ]
}

CRITICAL OUTPUT RULES:
1. Return ONLY valid JSON - no markdown code blocks, no preamble, no explanation outside JSON
2. "reasoning" field (top-level): String explaining overall strategy (1-2 sentences)
3. "searches" field: Array of 1-5 search objects
4. Each search object MUST have both "term" and "reasoning" as strings
5. Search terms should be 3-10 words, optimized for semantic similarity search
6. Do NOT include special characters, quotes, or formatting in search terms
"""

STRATEGY_GUIDELINES = """
STRATEGY GUIDELINES:
1. For SIMPLE queries (direct questions, single topic):
   - Generate exactly 1 search term
   - The search term should capture the core intent

2. For COMPLEX queries (multi-part, comparing concepts, asking about relationships):
   - Generate 2-5 search terms
   - Each term targets a DISTINCT aspect of the query
   - Terms should be complementary, not redundant

3. Search term optimization:
   - Use natural language phrases (not keywords)
   - Include the main subject/entity
   - Add relevant context words
   - Avoid stopwords and filler words

4. NEVER generate more than 5 searches (system will enforce limit anyway)
"""

STRATEGY_PRONOUN_RULES = """
PRONOUN RESOLUTION:
When the user's query contains pronouns (it, they, this, that, them, its, their):

1. Look at the conversation history to identify what the pronoun refers to
2. Replace ambiguous pronouns with explicit terms in your search queries
3. Include enough context so the search is self-contained

Examples of pronoun resolution:
- History: "Tell me about FastAPI" → Query: "How do I install it?" → Search term: "FastAPI installation"
- History: "Explain SQLAlchemy models" → Query: "How do I test them?" → Search term: "testing SQLAlchemy models"
- No history + Query: "What is Python and its features" → 'its' refers to Python in same sentence → Keep as-is

If no history exists, interpret pronouns based on query context alone.
"""

STRATEGY_EXAMPLES = """
<example type="simple">
Query: "How do I install FastAPI?"
Conversation History: (none)

Output:
{
  "reasoning": "Simple direct question about FastAPI installation - single search needed",
  "searches": [
    {"term": "FastAPI installation guide setup", "reasoning": "Targets installation documentation"}
  ]
}
</example>

<example type="complex">
Query: "Compare authentication methods in FastAPI and explain OAuth2 vs JWT differences"
Conversation History: (none)

Output:
{
  "reasoning": "Complex multi-part query covering auth comparison and two specific protocols",
  "searches": [
    {"term": "FastAPI authentication methods overview", "reasoning": "General auth patterns"},
    {"term": "FastAPI OAuth2 implementation tutorial", "reasoning": "OAuth2 specifics"},
    {"term": "FastAPI JWT token authentication security", "reasoning": "JWT implementation details"}
  ]
}
</example>

<example type="pronoun_resolution">
Query: "How do I test them?"
Conversation History:
User: Tell me about SQLAlchemy models
Assistant: SQLAlchemy models are defined using declarative base...

Output:
{
  "reasoning": "Resolved 'them' to SQLAlchemy models from conversation history",
  "searches": [
    {"term": "testing SQLAlchemy models pytest", "reasoning": "Resolved pronoun, added testing context"}
  ]
}
</example>

<example type="multi_aspect">
Query: "What are the performance implications of using async vs sync in database queries?"
Conversation History: (none)

Output:
{
  "reasoning": "Query asks about performance comparison requiring multiple perspectives",
  "searches": [
    {"term": "async database queries performance benefits", "reasoning": "Async advantages"},
    {"term": "sync vs async database connection overhead", "reasoning": "Direct comparison"},
    {"term": "asyncio database driver benchmarks", "reasoning": "Concrete performance data"}
  ]
}
</example>
"""


def build_strategy_prompt(query: str, formatted_history: str) -> str:
    """Build complete strategy generation prompt.

    Constructs a comprehensive prompt for the LLM to generate a search
    strategy with properly formatted output.

    Args:
        query: The user's current query to analyze
        formatted_history: Pre-formatted string of recent chat history

    Returns:
        Complete prompt string for LLM with structured output
    """
    return (
        STRATEGY_GENERATION_INSTRUCTIONS
        + "\n"
        + STRATEGY_OUTPUT_FORMAT
        + "\n"
        + STRATEGY_GUIDELINES
        + "\n"
        + STRATEGY_PRONOUN_RULES
        + "\n"
        + STRATEGY_EXAMPLES
        + f"\n\nConversation History:\n{formatted_history}"
        + f"\n\nCurrent Query: {query}"
        + "\n\nGenerate the search strategy JSON:"
    )


def format_chat_history_for_strategy(messages: list) -> str:
    """Format chat history for strategy generation prompt.

    Extracts and formats the most recent messages for context,
    truncating long messages to prevent prompt bloat.

    Args:
        messages: List of message objects (HumanMessage, AIMessage, etc.)

    Returns:
        Formatted string representation of recent history
    """
    if not messages:
        return "(No conversation history)"

    # Take last 5 messages for context (balance between context and prompt size)
    recent = messages[-5:]

    formatted_lines = []
    for msg in recent:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = getattr(msg, "content", str(msg))
        # Truncate long messages to prevent prompt bloat
        if len(content) > 200:
            content = content[:200] + "..."
        formatted_lines.append(f"{role}: {content}")

    return "\n".join(formatted_lines)
