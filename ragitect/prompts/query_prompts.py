"""Query processing prompts for reformulation and relevance grading.

Components:
- Query reformulation: Context-aware query rewriting with JSON output
- Relevance grading: Binary assessment of document relevance
"""

# Query Reformulation Components
REFORMULATION_BASE_INSTRUCTIONS = """
You are a query preprocessor for semantic search. Your task is to make queries self-contained when needed.
"""

REFORMULATION_OUTPUT_FORMAT = """
OUTPUT FORMAT:
Return a JSON object with this exact structure:
{"reasoning": "brief explanation", "reformulated_query": "the query", "was_modified": true/false}
"""

REFORMULATION_CRITICAL_RULES = """
CRITICAL RULES:
1. If the query is ALREADY SELF-CONTAINED, return it UNCHANGED with was_modified=false
   - Pronouns in the SAME SENTENCE as their referent do NOT need reformulation
   - "What is Quickshell and how do I install it?" → "it" refers to "Quickshell" in the same sentence → UNCHANGED

2. ONLY replace pronouns if they refer to something from PREVIOUS CONVERSATION TURNS that cannot be understood from the current query alone

3. NEVER add information that is not explicitly stated in the conversation history
4. NEVER invent or assume context - only use what is clearly present
5. Keep reformulated queries concise (1-2 sentences max)
"""

REFORMULATION_EXAMPLES = """
<example>
History: (empty)
Query: "What is Quickshell and how do I install it on Ubuntu?"
Output: {"reasoning": "Query is self-contained. The pronoun 'it' refers to 'Quickshell' within the same sentence.", "reformulated_query": "What is Quickshell and how do I install it on Ubuntu?", "was_modified": false}
</example>

<example>
History: User asked about FastAPI
Query: "How do I install it?"
Output: {"reasoning": "The pronoun 'it' refers to FastAPI from conversation history.", "reformulated_query": "How do I install FastAPI?", "was_modified": true}
</example>

<example>
History: (empty)
Query: "Tell me about Python and its features"
Output: {"reasoning": "Query is self-contained. The pronoun 'its' refers to 'Python' within the same sentence.", "reformulated_query": "Tell me about Python and its features", "was_modified": false}
</example>

<example>
History: User discussed PostgreSQL setup
Query: "Is it faster than MySQL?"
Output: {"reasoning": "The pronoun 'it' refers to PostgreSQL from conversation history.", "reformulated_query": "Is PostgreSQL faster than MySQL?", "was_modified": true}
</example>

<example>
History: (empty)
Query: "Compare React and Vue"
Output: {"reasoning": "Query is self-contained with no ambiguous references.", "reformulated_query": "Compare React and Vue", "was_modified": false}
</example>
"""


def build_reformulation_prompt(
    user_query: str,
    formatted_history: str,
) -> str:
    """Build query reformulation prompt with modular components.

    Extracted from query_service.py for consistency with RAG prompts.
    Follows same modular pattern for maintainability and A/B testing.

    Args:
        user_query: Current user query to reformulate
        formatted_history: XML formatted chat history string

    Returns:
        Complete reformulation prompt string
    """
    return (
        REFORMULATION_BASE_INSTRUCTIONS
        + "\n"
        + REFORMULATION_OUTPUT_FORMAT
        + "\n"
        + REFORMULATION_CRITICAL_RULES
        + "\n"
        + REFORMULATION_EXAMPLES
        + f"\n\nConversation History:\n{formatted_history}"
        + f"\n\nCurrent Query: {user_query}\n\nOutput:"
    )


# Relevance Grading Components
RELEVANCE_GRADING_INSTRUCTIONS = """
You are a grader assessing relevance of retrieved documents to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant.
"""

RELEVANCE_OUTPUT_FORMAT = """
OUTPUT FORMAT:
Return a JSON object with this exact structure:
{"score": "yes"} or {"score": "no"}

- "yes" means the document is relevant to the question
- "no" means the document is NOT relevant to the question

CRITICAL: Return ONLY the JSON object. No preamble, no explanation, no markdown code blocks.
"""

RELEVANCE_EXAMPLES = """
<example>
Question: "What is FastAPI?"
Document: "FastAPI is a modern web framework for building APIs with Python 3.7+"
Output: {"score": "yes"}
</example>

<example>
Question: "How do I install Docker?"
Document: "Kubernetes is a container orchestration platform that manages Docker containers."
Output: {"score": "yes"}
</example>

<example>
Question: "What is FastAPI?"
Document: "Chocolate cake is a delicious dessert made with cocoa powder."
Output: {"score": "no"}
</example>
"""


def build_relevance_grading_prompt(query: str, document: str) -> str:
    """Build relevance grading prompt.

    Extracted from query_service.py for consistency.
    Simple binary relevance assessment for iterative fallback.

    Args:
        query: User question
        document: Retrieved document text

    Returns:
        Complete grading prompt string
    """
    return (
        RELEVANCE_GRADING_INSTRUCTIONS
        + "\n"
        + RELEVANCE_OUTPUT_FORMAT
        + "\n"
        + RELEVANCE_EXAMPLES
        + f"\n\nHere is the retrieved document:\n{document}"
        + f"\n\nHere is the user question: {query}"
        + "\n\nOutput:"
    )
