You are generating test data for a RAG query classification system.

Generate 20 COMPLEX queries for a document management and code documentation RAG system.

**Definition of COMPLEX:**

- Query requires comparison between multiple items
- Query spans multiple concepts or documents
- Query has multi-hop reasoning (needs info from multiple places)
- Query has both context references AND requires decomposition

**Domain context:** RAGitect is a privacy-first document management app for developers.

**Output format:** JSON array with this schema:
{
"id": "complex-NNN",
"query": "the complex query",
"chat_history": [
{"role": "user", "content": "..."},
{"role": "assistant", "content": "..."}
],
"classification": "complex",
"expected_reformulation": "the fully resolved, expanded query",
"reformulation_trigger": "comparison|multi_hop|decomposition|aggregation",
"notes": "brief explanation of complexity"
}

**Types to include:**

- 8 comparison queries ("Compare X vs Y", "What's the difference between...")
- 6 multi-hop queries ("Based on X, how do I achieve Y?")
- 4 aggregation queries ("List all the...", "Summarize the approaches...")
- 2 decomposition queries (complex intent that could split into sub-queries)

Generate 20 diverse complex queries now:
