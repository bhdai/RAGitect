"""Tests for query router pattern.

Tests for _classify_query_complexity function that classifies
queries as 'simple', 'ambiguous', or 'complex'.
"""

import pytest

# Import will fail until implementation exists - this is expected (RED phase)
from ragitect.services.query_service import _classify_query_complexity

pytestmark = [pytest.mark.asyncio]


class TestQueryClassifier:
    """Tests for _classify_query_complexity function."""

    # ========== SIMPLE QUERY TESTS ==========

    def test_simple_query_no_history(self):
        """Test query with no history is classified as simple."""
        result = _classify_query_complexity("What is Python?", [])
        assert result == "simple"

    def test_simple_query_empty_history(self):
        """Test query with empty list history is classified as simple."""
        result = _classify_query_complexity("What is FastAPI?", [])
        assert result == "simple"

    def test_simple_query_direct_question(self):
        """Test direct question without pronouns is simple even with history."""
        history = [{"role": "user", "content": "Tell me about Python"}]
        result = _classify_query_complexity("What is FastAPI?", history)
        # No pronouns, no context refs - should be simple
        assert result == "simple"

    # ========== AMBIGUOUS QUERY TESTS (Pronouns) ==========

    def test_pronoun_it_triggers_ambiguous(self):
        """Test query with 'it' pronoun is classified as ambiguous."""
        history = [{"role": "user", "content": "Tell me about FastAPI"}]
        result = _classify_query_complexity("How do I install it?", history)
        assert result == "ambiguous"

    def test_pronoun_that_triggers_ambiguous(self):
        """Test query with 'that' pronoun is classified as ambiguous."""
        history = [{"role": "user", "content": "Tell me about decorators"}]
        result = _classify_query_complexity("Can you explain that more?", history)
        assert result == "ambiguous"

    def test_pronoun_this_triggers_ambiguous(self):
        """Test query with 'this' pronoun is classified as ambiguous."""
        history = [{"role": "user", "content": "Show me async code"}]
        result = _classify_query_complexity("What does this mean?", history)
        assert result == "ambiguous"

    def test_pronoun_they_triggers_ambiguous(self):
        """Test query with 'they' pronoun is classified as ambiguous."""
        history = [{"role": "user", "content": "Tell me about threads"}]
        result = _classify_query_complexity("How do they work?", history)
        assert result == "ambiguous"

    def test_pronoun_them_triggers_ambiguous(self):
        """Test query with 'them' pronoun is classified as ambiguous."""
        history = [{"role": "user", "content": "List Python frameworks"}]
        result = _classify_query_complexity("Tell me more about them", history)
        assert result == "ambiguous"

    def test_pronoun_those_triggers_ambiguous(self):
        """Test query with 'those' pronoun is classified as ambiguous."""
        history = [{"role": "user", "content": "Show me the errors"}]
        result = _classify_query_complexity("How do I fix those?", history)
        assert result == "ambiguous"

    def test_pronoun_these_triggers_ambiguous(self):
        """Test query with 'these' pronoun is classified as ambiguous."""
        history = [{"role": "user", "content": "Here are the packages"}]
        result = _classify_query_complexity("Install these", history)
        assert result == "ambiguous"

    def test_multiple_pronouns_triggers_ambiguous(self):
        """Test query with multiple pronouns is classified as ambiguous."""
        history = [
            {"role": "user", "content": "Tell me about decorators and generators"}
        ]
        result = _classify_query_complexity(
            "How do they work and why are they useful?", history
        )
        assert result == "ambiguous"

    # ========== AMBIGUOUS QUERY TESTS (Context References) ==========

    def test_context_ref_previous_triggers_ambiguous(self):
        """Test query with 'the previous' is classified as ambiguous."""
        history = [{"role": "user", "content": "Show me a code example"}]
        result = _classify_query_complexity(
            "Show me the previous example again", history
        )
        assert result == "ambiguous"

    def test_context_ref_earlier_triggers_ambiguous(self):
        """Test query with 'earlier' is classified as ambiguous."""
        history = [{"role": "user", "content": "Discuss API design"}]
        result = _classify_query_complexity(
            "What did you say earlier about endpoints?", history
        )
        assert result == "ambiguous"

    def test_context_ref_before_triggers_ambiguous(self):
        """Test query with 'before' is classified as ambiguous."""
        history = [{"role": "user", "content": "Tell me about caching"}]
        result = _classify_query_complexity("Like we discussed before", history)
        assert result == "ambiguous"

    def test_context_ref_above_triggers_ambiguous(self):
        """Test query with 'above' is classified as ambiguous."""
        history = [{"role": "user", "content": "Show some code"}]
        result = _classify_query_complexity("Fix the error above", history)
        assert result == "ambiguous"

    def test_context_ref_again_triggers_ambiguous(self):
        """Test query with 'again' is classified as ambiguous."""
        history = [{"role": "user", "content": "Explain async/await"}]
        result = _classify_query_complexity("Explain again please", history)
        assert result == "ambiguous"

    # ========== COMPLEX QUERY TESTS ==========

    def test_complex_compare_triggers_complex(self):
        """Test comparison query with 'compare' is classified as complex."""
        result = _classify_query_complexity("Compare FastAPI and Flask", [])
        assert result == "complex"

    def test_complex_vs_triggers_complex(self):
        """Test comparison query with 'vs' is classified as complex."""
        result = _classify_query_complexity("FastAPI vs Flask", [])
        assert result == "complex"

    def test_complex_versus_triggers_complex(self):
        """Test comparison query with 'versus' is classified as complex."""
        result = _classify_query_complexity("Python versus JavaScript for web", [])
        assert result == "complex"

    def test_complex_difference_between_triggers_complex(self):
        """Test query with 'difference between' is classified as complex."""
        result = _classify_query_complexity(
            "What is the difference between threads and processes?", []
        )
        assert result == "complex"

    def test_complex_both_triggers_complex(self):
        """Test query with 'both' is classified as complex."""
        result = _classify_query_complexity(
            "Explain both decorators and generators", []
        )
        assert result == "complex"

    def test_complex_all_of_triggers_complex(self):
        """Test query with 'all of' is classified as complex."""
        result = _classify_query_complexity("Show me all of the async patterns", [])
        assert result == "complex"

    def test_complex_each_of_triggers_complex(self):
        """Test query with 'each of' is classified as complex."""
        result = _classify_query_complexity("Describe each of the HTTP methods", [])
        assert result == "complex"

    # ========== EDGE CASES ==========

    def test_empty_history_no_pronouns_is_simple(self):
        """Test empty history with no pronouns returns simple."""
        result = _classify_query_complexity("What is Python?", [])
        assert result == "simple"

    def test_pronoun_without_history_is_simple(self):
        """Test pronoun without history returns simple (no context to resolve)."""
        result = _classify_query_complexity("How do I install it?", [])
        assert result == "simple"

    def test_complex_overrides_ambiguous(self):
        """Test complex pattern takes priority over ambiguous markers."""
        history = [{"role": "user", "content": "Tell me about Flask"}]
        # Contains 'them' (pronoun) AND 'compare' (complex)
        result = _classify_query_complexity("Compare them with Django", history)
        assert result == "complex"

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        result = _classify_query_complexity("COMPARE FastAPI VS Flask", [])
        assert result == "complex"

    def test_pronoun_in_word_not_detected(self):
        """Test that pronouns embedded in words are not detected."""
        # 'with' contains 'it' but should not trigger ambiguous
        history = [{"role": "user", "content": "Tell me about Python"}]
        result = _classify_query_complexity("What can I do with Python?", history)
        assert result == "simple"
