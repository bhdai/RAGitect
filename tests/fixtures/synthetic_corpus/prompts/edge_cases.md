You are generating EDGE CASE test data for a RAG query classification system.

Generate 15 TRICKY edge case queries that might confuse a simple classifier.

**Types of edge cases:**

1. **False positives for ambiguous (5):** Queries that LOOK ambiguous but are actually simple
   - "What is THIS error message: ConnectionRefusedError?"
   - Has "this" but it's a demonstrative, not a pronoun
2. **False negatives for ambiguous (5):** Queries that look simple but need context
   - "Documentation?" (too short, relies heavily on context)
   - "More details?" (elliptical, needs context)

3. **Borderline simple/ambiguous (3):** Could go either way
   - "Tell me about the database" (which database?)
   - Might be simple (user has one DB) or ambiguous (multiple DBs discussed)

4. **Classification conflicts (2):** Queries that could be multiple categories
   - "Compare it with the other option" (ambiguous + complex)

**Output format:** Same JSON schema as before, but add:
{
"id": "edge-NNN",
"query": "...",
"chat_history": [...],
"classification": "the CORRECT classification",
"expected_reformulation": "...",
"reformulation_trigger": "...",
"edge_case_type": "false_positive_ambiguous|false_negative_ambiguous|borderline|conflict",
"notes": "why this is tricky"
}

Generate 15 diverse edge cases now:
