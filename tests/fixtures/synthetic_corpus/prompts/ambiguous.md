You are generating test data for a RAG query classification system.

Generate 40 AMBIGUOUS queries for a document management and code documentation RAG system.

**Definition of AMBIGUOUS:**

- Query contains pronouns (it, that, this, they, those, these, one) that reference previous context
- Query has context references ("the previous", "earlier", "above", "again", "mentioned")
- Query REQUIRES chat history to understand the user's intent
- Direct search would fail without resolving references

**Domain context:** RAGitect is a privacy-first document management app for developers. Users discuss technical docs, code, APIs.

**Output format:** JSON array with this schema:
{
"id": "ambiguous-NNN",
"query": "the ambiguous query",
"chat_history": [
{"role": "user", "content": "..."},
{"role": "assistant", "content": "..."}
],
"classification": "ambiguous",
"expected_reformulation": "the self-contained reformulated query",
"reformulation_trigger": "pronoun_it|pronoun_that|pronoun_this|pronoun_they|context_ref|ellipsis",
"notes": "brief explanation of what needs resolution"
}

**Distribution of triggers (aim for variety):**

- 12 queries with "it" pronoun
- 8 queries with "that/this/these/those"
- 5 queries with "they/them"
- 5 queries with "one" (e.g., "How do I configure one?")
- 5 queries with context references ("the previous", "earlier", "above")
- 5 queries with ellipsis (omitted subject, e.g., "And for Python?" after discussing JavaScript)

**Chat history guidelines:**

- Keep history realistic (2-4 turns max)
- Assistant responses should be brief summaries (1-2 sentences)
- Make the referent clearly identifiable in history

Generate 40 diverse ambiguous queries now:
