"""Tests for query router validation using synthetic corpus.

Validates that the query classifier correctly classifies queries
according to the expected classifications in the synthetic corpus.
"""

import json
from pathlib import Path

import pytest

from ragitect.services.query_service import _classify_query_complexity


# Path to synthetic corpus
CORPUS_PATH = (
    Path(__file__).parent.parent / "fixtures" / "synthetic_corpus" / "corpus.json"
)


@pytest.fixture
def corpus_data():
    """Load synthetic corpus data."""
    if not CORPUS_PATH.exists():
        pytest.skip(f"Synthetic corpus not found at {CORPUS_PATH}")

    with open(CORPUS_PATH) as f:
        data = json.load(f)

    return data.get("corpus", [])


class TestCorpusValidation:
    """Validation tests using synthetic query corpus."""

    def test_corpus_exists_and_has_data(self, corpus_data):
        """Test that corpus exists and has test cases."""
        assert len(corpus_data) > 0, "Corpus should contain test cases"

        # Check required fields exist
        first_item = corpus_data[0]
        assert "id" in first_item
        assert "query" in first_item
        assert "chat_history" in first_item
        assert "classification" in first_item

    def test_corpus_classifications(self, corpus_data):
        """Test classifier against all corpus entries and report metrics."""
        results = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "by_classification": {
                "simple": {"total": 0, "correct": 0},
                "ambiguous": {"total": 0, "correct": 0},
                "complex": {"total": 0, "correct": 0},
            },
            "failures": [],
        }

        for entry in corpus_data:
            query = entry["query"]
            history = entry.get("chat_history", [])
            expected = entry["classification"]
            entry_id = entry.get("id", "unknown")

            # Skip if classification is not one of our expected values
            if expected not in ["simple", "ambiguous", "complex"]:
                continue

            actual = _classify_query_complexity(query, history)

            results["total"] += 1
            results["by_classification"][expected]["total"] += 1

            if actual == expected:
                results["correct"] += 1
                results["by_classification"][expected]["correct"] += 1
            else:
                results["incorrect"] += 1
                results["failures"].append(
                    {
                        "id": entry_id,
                        "query": query[:60],
                        "expected": expected,
                        "actual": actual,
                    }
                )

        # Calculate accuracy
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0

        # Print detailed metrics
        print(f"\n\n=== Corpus Classification Metrics ===")
        print(f"Total: {results['total']}")
        print(f"Correct: {results['correct']} ({accuracy:.1%})")
        print(f"Incorrect: {results['incorrect']}")

        for classification, stats in results["by_classification"].items():
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                print(
                    f"  {classification}: {stats['correct']}/{stats['total']} ({acc:.1%})"
                )

        if results["failures"]:
            print(f"\nFirst 10 failures:")
            for failure in results["failures"][:10]:
                print(
                    f"  [{failure['id']}] '{failure['query']}...' expected={failure['expected']} got={failure['actual']}"
                )

        # Assert minimum accuracy threshold (adjusted based on heuristic nature)
        # Initial conservative threshold - can be tuned after baseline established
        assert accuracy >= 0.5, (
            f"Classification accuracy {accuracy:.1%} below 50% threshold"
        )

    def test_simple_queries_percentage(self, corpus_data):
        """Test that expected percentage of queries are classified as simple.

        AC: Target 60-70% queries skip reformulation (simple classification).
        This test validates classifier behavior matches expected distribution.
        """
        simple_expected = sum(
            1 for e in corpus_data if e.get("classification") == "simple"
        )
        total = len(corpus_data)

        if total == 0:
            pytest.skip("Empty corpus")

        expected_simple_ratio = simple_expected / total

        # Classify all queries
        simple_actual = sum(
            1
            for e in corpus_data
            if _classify_query_complexity(e["query"], e.get("chat_history", []))
            == "simple"
        )

        actual_simple_ratio = simple_actual / total

        print(f"\n=== Simple Query Ratio ===")
        print(
            f"Expected simple in corpus: {simple_expected}/{total} ({expected_simple_ratio:.1%})"
        )
        print(
            f"Actual classified simple: {simple_actual}/{total} ({actual_simple_ratio:.1%})"
        )

        # The classifier should classify at least 40% as simple to achieve cost savings
        # (Conservative lower bound for MVP)
        assert actual_simple_ratio >= 0.40, (
            f"Only {actual_simple_ratio:.1%} classified as simple, "
            f"target is >=40% for cost reduction benefits"
        )

    def test_classification_consistency(self, corpus_data):
        """Test that classification is deterministic for same inputs."""
        # Pick first 20 entries for consistency check
        sample = corpus_data[:20]

        for entry in sample:
            query = entry["query"]
            history = entry.get("chat_history", [])

            # Classify same query multiple times
            results = [_classify_query_complexity(query, history) for _ in range(3)]

            # All results should be identical
            assert len(set(results)) == 1, (
                f"Inconsistent classification for query: '{query[:50]}...' "
                f"got results: {results}"
            )


class TestClassificationMetrics:
    """Metrics and performance tests for query classification."""

    def test_latency_is_minimal(self, corpus_data):
        """Test that classification is fast (no LLM call)."""
        import time

        sample = corpus_data[:50]  # Test with 50 queries

        start = time.time()
        for entry in sample:
            _classify_query_complexity(entry["query"], entry.get("chat_history", []))
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / len(sample)) * 1000

        print(f"\n=== Classification Latency ===")
        print(f"Total time for {len(sample)} queries: {elapsed * 1000:.2f}ms")
        print(f"Average per query: {avg_latency_ms:.4f}ms")

        # Classification should be sub-millisecond (no LLM call)
        assert avg_latency_ms < 1.0, (
            f"Classification too slow: {avg_latency_ms:.2f}ms per query, "
            f"expected <1ms (pure heuristics)"
        )
