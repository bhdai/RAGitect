You are generating test data for a RAG (Retrieval-Augmented Generation) query classification system.

Generate 30 SIMPLE queries for a document management and code documentation RAG system.

**Definition of SIMPLE:**

- Query is self-contained (no pronouns referencing previous context)
- No chat history is needed to understand the query
- Direct search can find relevant results without reformulation

**Domain context:** RAGitect is a privacy-first document management app for developers. Users upload technical docs, code documentation, APIs, and project files.

**Output format:** JSON array with this schema:
{
"id": "simple-NNN",
"query": "the user's query",
"chat_history": [],
"classification": "simple",
"expected_reformulation": null,
"reformulation_trigger": null,
"notes": "brief explanation"
}

**Requirements:**

- Diverse query types (what/how/why/when/where/explain/show/list)
- Technical topics (databases, APIs, frameworks, deployment, testing)
- Varying lengths (3 words to 15 words)
- Mix of questions and commands ("What is X?" vs "Explain X")

Generate 30 diverse simple queries now:
