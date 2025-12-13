"""Real LLM Comparison Tests for Phase 1 Query Hotfix

This test runs actual queries through Ollama to measure:
- Real latency with the simplified prompt
- Token usage reduction
- Output quality (no answer generation)

Prerequisites:
- Ollama running on http://localhost:11434
- Model: llama3.2:3b (or similar)

Run with: uv run pytest tests/services/test_query_real_llm.py -v -s
"""

import time

import pytest
from langchain_community.chat_models import ChatOllama

from ragitect.services.query_service import (
    _build_reformulation_prompt,
    _validate_reformulated_query,
    format_chat_history,
    reformulate_query_with_chat_history,
)

# Apply asyncio and integration markers to all tests in this module
# Tests require Ollama to be running, so they are integration tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# Test queries for real LLM evaluation (subset from baseline)
REAL_LLM_TEST_QUERIES = [
    {
        "query": "How do I install it?",
        "history": [
            {"role": "user", "content": "What is FastAPI?"},
            {
                "role": "assistant",
                "content": "FastAPI is a modern Python web framework.",
            },
        ],
    },
    {
        "query": "Show me an example",
        "history": [
            {"role": "user", "content": "Explain Python decorators"},
            {"role": "assistant", "content": "Decorators modify function behavior."},
        ],
    },
    {
        "query": "What does that mean?",
        "history": [
            {"role": "user", "content": "Tell me about async/await"},
            {
                "role": "assistant",
                "content": "Async/await allows concurrent execution.",
            },
        ],
    },
    {
        "query": "Which one should I use?",
        "history": [
            {
                "role": "user",
                "content": "What's the difference between List and Tuple?",
            },
            {
                "role": "assistant",
                "content": "Lists are mutable, tuples are immutable.",
            },
        ],
    },
    {
        "query": "Is there a better way?",
        "history": [
            {"role": "user", "content": "How do I read a file in Python?"},
            {"role": "assistant", "content": "Use open() with a context manager."},
        ],
    },
    {
        "query": "Tell me more",
        "history": [
            {"role": "user", "content": "What is type hinting in Python?"},
            {"role": "assistant", "content": "Type hints specify variable types."},
        ],
    },
    {
        "query": "How does this work internally?",
        "history": [
            {"role": "user", "content": "What is a Python generator?"},
            {
                "role": "assistant",
                "content": "Generators yield values lazily using yield.",
            },
        ],
    },
    {
        "query": "Can you compare them?",
        "history": [
            {
                "role": "user",
                "content": "What's the difference between REST and GraphQL?",
            },
            {
                "role": "assistant",
                "content": "REST uses endpoints, GraphQL uses queries.",
            },
        ],
    },
]


@pytest.fixture
def ollama_llm():
    """Create Ollama LLM instance for testing"""
    return ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.1,  # Low temperature for consistency
    )


class TestRealLLMComparison:
    """Real LLM performance comparison for Phase 1 hotfix"""

    async def test_real_llm_reformulation_sample(self, ollama_llm):
        """Test reformulation with real LLM on sample queries

        This measures actual latency and validates output quality
        with the simplified prompt.
        """
        results = []
        total_latency = 0
        validation_failures = 0

        print("\n" + "=" * 80)
        print("REAL LLM COMPARISON TEST - Phase 1 Simplified Prompt")
        print("=" * 80)

        for idx, test_case in enumerate(REAL_LLM_TEST_QUERIES, 1):
            query = test_case["query"]
            history = test_case["history"]

            print(f"\n[{idx}/{len(REAL_LLM_TEST_QUERIES)}] Testing query: {query}")

            start_time = time.time()
            reformulated = await reformulate_query_with_chat_history(
                ollama_llm, query, history
            )
            latency = (time.time() - start_time) * 1000

            # Validate output
            is_valid = _validate_reformulated_query(reformulated)

            print(f"  Original:     {query}")
            print(f"  Reformulated: {reformulated}")
            print(f"  Latency:      {latency:.2f}ms")
            print(f"  Validation:   {'✓ PASS' if is_valid else '✗ FAIL'}")

            results.append(
                {
                    "query": query,
                    "reformulated": reformulated,
                    "latency_ms": latency,
                    "validation_passed": is_valid,
                }
            )

            total_latency += latency
            if not is_valid:
                validation_failures += 1

        # Summary
        avg_latency = total_latency / len(REAL_LLM_TEST_QUERIES)
        validation_pass_rate = (
            (len(REAL_LLM_TEST_QUERIES) - validation_failures)
            / len(REAL_LLM_TEST_QUERIES)
            * 100
        )

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total queries tested: {len(REAL_LLM_TEST_QUERIES)}")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Validation pass rate: {validation_pass_rate:.1f}%")
        print(f"Validation failures: {validation_failures}")
        print("=" * 80)

        # Calculate prompt metrics
        sample_history = format_chat_history(REAL_LLM_TEST_QUERIES[0]["history"])
        sample_prompt = _build_reformulation_prompt(
            REAL_LLM_TEST_QUERIES[0]["query"], sample_history
        )

        print("\nPrompt Metrics:")
        print(f"  New prompt length: {len(sample_prompt)} chars")
        print(f"  Estimated tokens: {len(sample_prompt) // 4}")
        print("  Baseline was: 2354 chars (588 tokens)")
        print(
            f"  Reduction: {((2354 - len(sample_prompt)) / 2354 * 100):.1f}% fewer chars"
        )
        print("=" * 80)

        # Phase 1 success criteria
        assert validation_pass_rate >= 90, (
            f"Validation pass rate too low: {validation_pass_rate}%"
        )
        print("\n✅ Phase 1 Hotfix: Output validation working correctly")

    async def test_prompt_reduction_measurement(self, ollama_llm):
        """Measure prompt token reduction achieved in Phase 1"""
        # Baseline prompt length (from test_query_baseline.py)
        BASELINE_AVG_LENGTH = 2354
        BASELINE_AVG_TOKENS = 588

        # Measure new prompt length
        sample_history = format_chat_history(
            [
                {"role": "user", "content": "What is FastAPI?"},
                {
                    "role": "assistant",
                    "content": "FastAPI is a modern Python web framework.",
                },
            ]
        )
        new_prompt = _build_reformulation_prompt("How do I install it?", sample_history)

        new_length = len(new_prompt)
        new_tokens = new_length // 4

        reduction_pct = ((BASELINE_AVG_LENGTH - new_length) / BASELINE_AVG_LENGTH) * 100

        print("\n" + "=" * 80)
        print("PROMPT REDUCTION ANALYSIS")
        print("=" * 80)
        print("Baseline (Pre-Phase 1):")
        print(f"  Average length: {BASELINE_AVG_LENGTH} chars")
        print(f"  Estimated tokens: {BASELINE_AVG_TOKENS}")
        print("\nNew (Phase 1 Simplified):")
        print(f"  Length: {new_length} chars")
        print(f"  Estimated tokens: {new_tokens}")
        print("\nReduction:")
        print(f"  Characters: {reduction_pct:.1f}%")
        print(
            f"  Tokens: {((BASELINE_AVG_TOKENS - new_tokens) / BASELINE_AVG_TOKENS) * 100:.1f}%"
        )
        print("=" * 80)

        # Phase 1 acceptance criteria: 60%+ token reduction
        assert reduction_pct >= 60, (
            f"Token reduction insufficient: {reduction_pct:.1f}%"
        )
        print(f"\n✅ Phase 1 Target Met: {reduction_pct:.1f}% reduction (target: 60%+)")


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v", "-s"])
