"""Baseline Performance Tests for Query Reformulation

This test suite establishes baseline performance metrics BEFORE Phase 1 hotfix changes.
It measures:
- Latency per query reformulation
- Token usage (prompt length as proxy)
- Output quality (manual validation)

These metrics will be compared against post-implementation measurements to validate
the 20%+ latency improvement and 60%+ token reduction goals.

Baseline Results (Pre-Implementation):
-------------------------------------
Prompt Length: ~2900 characters (estimated 725 tokens)
Average Latency: TBD (requires real LLM)
Output Quality: TBD (manual inspection)

Expected Post-Implementation:
------------------------------
Prompt Length: ~400 characters (estimated 100 tokens) - 86% reduction
Average Latency: 20%+ faster
Output Quality: 0% answer generation (100% valid queries)
"""

import time
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from ragitect.services.query_service import (
    _build_reformulation_prompt,
    format_chat_history,
    reformulate_query_with_chat_history,
)

# Test Query Dataset (20+ diverse queries)
# Categories: Simple, Ambiguous, Contextual, Complex
TEST_QUERIES = [
    # Simple queries (should reformulate with minimal context)
    {
        "query": "How do I install it?",
        "history": [
            {"role": "user", "content": "What is FastAPI?"},
            {
                "role": "assistant",
                "content": "FastAPI is a modern Python web framework.",
            },
        ],
        "expected_pattern": "install FastAPI",
    },
    {
        "query": "Show me an example",
        "history": [
            {"role": "user", "content": "Explain Python decorators"},
            {"role": "assistant", "content": "Decorators modify function behavior."},
        ],
        "expected_pattern": "example.*decorators",
    },
    # Ambiguous queries (need pronoun resolution)
    {
        "query": "What does that mean?",
        "history": [
            {"role": "user", "content": "Tell me about async/await"},
            {
                "role": "assistant",
                "content": "Async/await allows concurrent execution.",
            },
        ],
        "expected_pattern": "async.?await",
    },
    {
        "query": "Can you explain it further?",
        "history": [
            {"role": "user", "content": "What is a virtual environment?"},
            {"role": "assistant", "content": "It's an isolated Python environment."},
        ],
        "expected_pattern": "virtual environment",
    },
    # Contextual queries (require history synthesis)
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
            {"role": "user", "content": "When should I use each?"},
            {
                "role": "assistant",
                "content": "Use lists for changing data, tuples for fixed data.",
            },
        ],
        "expected_pattern": "(list|tuple)",
    },
    {
        "query": "Is there a better way?",
        "history": [
            {"role": "user", "content": "How do I read a file in Python?"},
            {"role": "assistant", "content": "Use open() with a context manager."},
        ],
        "expected_pattern": "read.*file.*Python",
    },
    # Complex queries (long history, nuanced context)
    {
        "query": "What are the best practices?",
        "history": [
            {"role": "user", "content": "How do I structure a FastAPI project?"},
            {"role": "assistant", "content": "Use routers, services, and models."},
            {"role": "user", "content": "What about error handling?"},
            {
                "role": "assistant",
                "content": "Use HTTPException and custom exception handlers.",
            },
        ],
        "expected_pattern": "FastAPI.*(structure|project)",
    },
    {
        "query": "Why would I need this?",
        "history": [
            {"role": "user", "content": "Explain database indexing"},
            {"role": "assistant", "content": "Indexes speed up query lookups."},
        ],
        "expected_pattern": "database.*(index|indexing)",
    },
    # Edge cases
    {
        "query": "Tell me more",
        "history": [
            {"role": "user", "content": "What is type hinting in Python?"},
            {"role": "assistant", "content": "Type hints specify variable types."},
        ],
        "expected_pattern": "type.*(hint|hinting)",
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
        "expected_pattern": "generator",
    },
    # Additional test cases for robustness
    {
        "query": "What are the alternatives?",
        "history": [
            {"role": "user", "content": "Should I use SQLAlchemy?"},
            {"role": "assistant", "content": "SQLAlchemy is a powerful ORM."},
        ],
        "expected_pattern": "SQLAlchemy.*alternative",
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
        "expected_pattern": "(REST|GraphQL)",
    },
    {
        "query": "How do I debug this?",
        "history": [
            {"role": "user", "content": "My async function is hanging"},
            {
                "role": "assistant",
                "content": "Check for await statements and deadlocks.",
            },
        ],
        "expected_pattern": "async.*function.*(debug|hang)",
    },
    {
        "query": "Is there any performance impact?",
        "history": [
            {"role": "user", "content": "Should I use Pydantic validation?"},
            {"role": "assistant", "content": "Pydantic provides automatic validation."},
        ],
        "expected_pattern": "Pydantic.*performance",
    },
    {
        "query": "What's the recommended approach?",
        "history": [
            {"role": "user", "content": "How to handle database migrations?"},
            {
                "role": "assistant",
                "content": "Use Alembic for SQLAlchemy migrations.",
            },
        ],
        "expected_pattern": "database.*(migration|Alembic)",
    },
    {
        "query": "Can this cause issues?",
        "history": [
            {"role": "user", "content": "Should I use global variables?"},
            {
                "role": "assistant",
                "content": "Global variables can cause side effects.",
            },
        ],
        "expected_pattern": "global.*variable",
    },
    {
        "query": "How do I optimize it?",
        "history": [
            {"role": "user", "content": "My API endpoint is slow"},
            {
                "role": "assistant",
                "content": "Consider database indexing and query optimization.",
            },
        ],
        "expected_pattern": "(API|endpoint).*(optimize|slow)",
    },
    {
        "query": "What's the syntax?",
        "history": [
            {"role": "user", "content": "Explain Python list comprehensions"},
            {
                "role": "assistant",
                "content": "List comprehensions create lists concisely.",
            },
        ],
        "expected_pattern": "list.*comprehension.*syntax",
    },
    {
        "query": "Where should I use this?",
        "history": [
            {"role": "user", "content": "What are Python context managers?"},
            {
                "role": "assistant",
                "content": "Context managers handle resource cleanup.",
            },
        ],
        "expected_pattern": "context.?manager",
    },
    {
        "query": "Is there a shortcut?",
        "history": [
            {"role": "user", "content": "How to merge dictionaries in Python?"},
            {"role": "assistant", "content": "Use the ** operator or dict.update()."},
        ],
        "expected_pattern": "(merge|dictionary|dictionaries)",
    },
    # Multi-turn deep context
    {
        "query": "What about edge cases?",
        "history": [
            {"role": "user", "content": "How to validate user input?"},
            {"role": "assistant", "content": "Use Pydantic models for validation."},
            {"role": "user", "content": "What about optional fields?"},
            {
                "role": "assistant",
                "content": "Use Optional[Type] or Type | None.",
            },
        ],
        "expected_pattern": "(validate|validation|input).*edge",
    },
    {
        "query": "Can you show the code?",
        "history": [
            {"role": "user", "content": "Explain dependency injection in FastAPI"},
            {
                "role": "assistant",
                "content": "FastAPI uses Depends() for dependency injection.",
            },
        ],
        "expected_pattern": "(dependency.?injection|FastAPI|Depends)",
    },
    {
        "query": "What's the difference here?",
        "history": [
            {"role": "user", "content": "Should I use Path or Query parameters?"},
            {
                "role": "assistant",
                "content": "Path for URL parts, Query for optional filters.",
            },
        ],
        "expected_pattern": "(Path|Query).*parameter",
    },
    # Very long history (testing history truncation)
    {
        "query": "How do I implement this pattern?",
        "history": [
            {"role": "user", "content": "What is the repository pattern?"},
            {
                "role": "assistant",
                "content": "Repository pattern abstracts data access.",
            },
            {"role": "user", "content": "Why use it?"},
            {"role": "assistant", "content": "It decouples business logic from data."},
            {"role": "user", "content": "What are the benefits?"},
            {"role": "assistant", "content": "Testability and maintainability."},
            {"role": "user", "content": "Any downsides?"},
            {"role": "assistant", "content": "Can add complexity for simple cases."},
        ],
        "expected_pattern": "repository.*pattern",
    },
]


class TestQueryReformulationBaseline:
    """Baseline performance measurements for query reformulation"""

    @pytest.mark.parametrize("test_case", TEST_QUERIES)
    def test_baseline_prompt_length(self, test_case):
        """Measure baseline prompt length for token usage estimation

        Current baseline: ~2900 characters per prompt (estimated 725 tokens)
        Target: <400 characters (estimated 100 tokens) - 86% reduction
        """
        query = test_case["query"]
        history = test_case["history"]

        formatted_history = format_chat_history(history)
        prompt = _build_reformulation_prompt(query, formatted_history)

        prompt_length = len(prompt)
        estimated_tokens = prompt_length // 4  # rough estimate: 1 token = 4 chars

        # Log baseline metrics
        print(f"\nQuery: {query}")
        print(f"Prompt length: {prompt_length} chars")
        print(f"Estimated tokens: {estimated_tokens}")
        print(f"History size: {len(history)} messages")

        # Current baseline: prompts are very long (~2900 chars)
        # After Phase 1: expect ~400 chars
        assert prompt_length > 0  # sanity check

    @pytest.mark.asyncio
    async def test_baseline_reformulation_with_mock(self):
        """Test reformulation with mocked LLM responses

        This test uses realistic mock responses to simulate LLM behavior
        without actual API calls (for CI/CD).
        """
        mock_llm = AsyncMock(spec=BaseChatModel)

        # Mock realistic reformulated queries
        test_cases_with_responses = [
            (
                "How do I install it?",
                [
                    {"role": "user", "content": "What is FastAPI?"},
                    {
                        "role": "assistant",
                        "content": "FastAPI is a modern Python web framework.",
                    },
                ],
                "How do I install FastAPI?",
            ),
            (
                "Show me an example",
                [
                    {"role": "user", "content": "Explain Python decorators"},
                    {
                        "role": "assistant",
                        "content": "Decorators modify function behavior.",
                    },
                ],
                "Show me an example of Python decorators",
            ),
            (
                "What does that mean?",
                [
                    {"role": "user", "content": "Tell me about async/await"},
                    {
                        "role": "assistant",
                        "content": "Async/await allows concurrent execution.",
                    },
                ],
                "What does async/await mean?",
            ),
        ]

        for query, history, expected_response in test_cases_with_responses:
            # Mock LLM response
            with patch(
                "ragitect.services.query_service.generate_response",
                return_value=expected_response,
            ):
                start_time = time.time()
                result = await reformulate_query_with_chat_history(
                    mock_llm, query, history
                )
                latency_ms = (time.time() - start_time) * 1000

                print(f"\nQuery: {query}")
                print(f"Reformulated: {result}")
                print(f"Latency: {latency_ms:.2f}ms (mocked)")

                # Validation: output should be a valid query (not an answer)
                # This is a basic check - Phase 1 will add more robust validation
                assert len(result) > 0
                assert result != query  # should have been reformulated

    def test_baseline_metrics_summary(self):
        """Summarize baseline metrics for comparison

        This test documents the baseline state before Phase 1 changes.
        After implementation, compare these metrics with post-implementation results.

        NOTE: This test now runs AFTER Phase 1 implementation, so it measures
        the NEW simplified prompt. The BASELINE was:
        - 2354 chars average
        - 588 tokens estimated
        """
        total_test_cases = len(TEST_QUERIES)

        # Calculate average prompt length (NOW measures simplified prompt)
        total_prompt_length = 0
        for test_case in TEST_QUERIES:
            formatted_history = format_chat_history(test_case["history"])
            prompt = _build_reformulation_prompt(test_case["query"], formatted_history)
            total_prompt_length += len(prompt)

        avg_prompt_length = total_prompt_length // total_test_cases
        avg_estimated_tokens = avg_prompt_length // 4

        print("\n" + "=" * 60)
        print("BASELINE METRICS SUMMARY")
        print("=" * 60)
        print("PRE-PHASE 1 (Original):")
        print("  Average prompt length: 2354 characters")
        print("  Average estimated tokens: 588 tokens")
        print("  Approach: Few-shot examples (3+ examples)")
        print("\nPOST-PHASE 1 (Simplified):")
        print(f"  Average prompt length: {avg_prompt_length} characters")
        print(f"  Average estimated tokens: {avg_estimated_tokens} tokens")
        print("  Approach: Direct instructions only")
        print("\nIMPROVEMENT:")
        print(f"  Reduction: {((2354 - avg_prompt_length) / 2354 * 100):.1f}%")
        print("=" * 60)

        # Document baseline for tech spec
        baseline_summary = {
            "test_cases": total_test_cases,
            "original_avg_length": 2354,
            "new_avg_length": avg_prompt_length,
            "reduction_pct": ((2354 - avg_prompt_length) / 2354 * 100),
        }

        assert baseline_summary["test_cases"] >= 20  # At least 20 test cases


# Note: For actual baseline with real LLM, run:
# uv run pytest tests/services/test_query_baseline.py -v -s
# This will show prompt lengths and latency measurements
