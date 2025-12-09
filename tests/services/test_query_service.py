"""Tests for query_service.py"""

import pytest

from ragitect.services.query_service import (
    _should_reformulate,
    _extract_reformulated_query,
    _validate_reformulated_query,
    format_chat_history,
    _build_reformulation_prompt,
)


class TestShouldReformulate:
    """Test query reformulation decision logic"""

    def test_returns_false_for_empty_history(self):
        assert _should_reformulate("What is FastAPI?", []) is False

    def test_returns_false_for_single_word_query(self):
        history = [{"role": "user", "content": "Tell me about Python"}]
        assert _should_reformulate("explain", history) is False

    def test_returns_false_for_very_long_query(self):
        history = [{"role": "user", "content": "Previous question"}]
        long_query = " ".join(["word"] * 51)
        assert _should_reformulate(long_query, history) is False

    def test_returns_true_for_normal_query_with_history(self):
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        assert _should_reformulate("How do I install it?", history) is True

    def test_returns_true_for_multi_word_short_query(self):
        history = [{"role": "user", "content": "Previous context"}]
        assert _should_reformulate("tell me more", history) is True


class TestExtractReformulatedQuery:
    """Test reformulated query extraction logic"""

    def test_removes_common_prefix_reformulated_query(self):
        response = "Reformulated Query: How do I install FastAPI?"
        result = _extract_reformulated_query(response)
        assert result == "How do I install FastAPI?"

    def test_removes_prefix_reformulated(self):
        response = "Reformulated: Show me examples"
        result = _extract_reformulated_query(response)
        assert result == "Show me examples"

    def test_removes_prefix_query(self):
        response = "Query: What is async?"
        result = _extract_reformulated_query(response)
        assert result == "What is async?"

    def test_removes_verbose_prefix(self):
        response = "Here's the reformulated query: Explain Python decorators"
        result = _extract_reformulated_query(response)
        assert result == "Explain Python decorators"

    def test_removes_surrounding_quotes(self):
        response = '"How do I use async functions?"'
        result = _extract_reformulated_query(response)
        assert result == "How do I use async functions?"

    def test_handles_single_quotes(self):
        response = "'What is FastAPI?'"
        result = _extract_reformulated_query(response)
        assert result == "What is FastAPI?"

    def test_handles_clean_response(self):
        response = "What is the difference between async and sync?"
        result = _extract_reformulated_query(response)
        assert result == "What is the difference between async and sync?"

    def test_strips_whitespace(self):
        response = "   Reformulated Query:   How to use decorators?   "
        result = _extract_reformulated_query(response)
        assert result == "How to use decorators?"


class TestFormatChatHistory:
    """Test chat history XML formatting"""

    def test_formats_empty_history(self):
        result = format_chat_history([])
        assert result == "<chat_history>\n</chat_history>"

    def test_formats_single_message(self):
        history = [{"role": "user", "content": "Hello"}]
        result = format_chat_history(history)

        assert "<chat_history>" in result
        assert '<message role="user">Hello</message>' in result
        assert "</chat_history>" in result

    def test_formats_multiple_messages(self):
        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
        result = format_chat_history(history)

        assert '<message role="user">Question 1</message>' in result
        assert '<message role="assistant">Answer 1</message>' in result
        assert '<message role="user">Question 2</message>' in result

    def test_raises_error_on_missing_role(self):
        history = [{"content": "Hello"}]
        with pytest.raises(ValueError, match="missing 'role' key"):
            format_chat_history(history)

    def test_raises_error_on_missing_content(self):
        history = [{"role": "user"}]
        with pytest.raises(ValueError, match="missing 'content' key"):
            format_chat_history(history)

    def test_preserves_message_order(self):
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        result = format_chat_history(history)

        lines = result.split("\n")
        assert "First" in lines[1]
        assert "Second" in lines[2]
        assert "Third" in lines[3]


class TestBuildReformulationPrompt:
    """Test prompt building for reformulation"""

    def test_includes_user_query(self):
        user_query = "How do I install it?"
        formatted_history = "<chat_history>\n</chat_history>"

        prompt = _build_reformulation_prompt(user_query, formatted_history)

        assert "How do I install it?" in prompt

    def test_includes_formatted_history(self):
        user_query = "What is that?"
        formatted_history = '<chat_history>\n<message role="user">Tell me about FastAPI</message>\n</chat_history>'

        prompt = _build_reformulation_prompt(user_query, formatted_history)

        assert formatted_history in prompt

    def test_includes_instructions(self):
        prompt = _build_reformulation_prompt(
            "test query", "<chat_history>\n</chat_history>"
        )

        # Check for key instruction elements
        assert "reformulate" in prompt.lower() or "reformulated" in prompt.lower()
        assert "Rules:" in prompt or "rules:" in prompt.lower()

    def test_simplified_prompt_is_shorter(self):
        """Phase 1: Verify prompt is significantly shorter than baseline"""
        user_query = "How do I use it?"
        formatted_history = """<chat_history>
<message role="user">What is FastAPI?</message>
<message role="assistant">FastAPI is a modern Python web framework.</message>
</chat_history>"""

        prompt = _build_reformulation_prompt(user_query, formatted_history)

        # New prompt should be ~400 chars (vs baseline ~2350 chars)
        # Allow some variance for history content
        assert len(prompt) < 600, f"Prompt too long: {len(prompt)} chars"
        assert "Example" not in prompt, "Prompt should not contain few-shot examples"


class TestValidateReformulatedQuery:
    """Test output validation for reformulated queries (Phase 1 Hotfix)"""

    # Valid queries (should return True)
    def test_accepts_valid_question_with_question_mark(self):
        assert _validate_reformulated_query("What is FastAPI?") is True

    def test_accepts_valid_how_to_question(self):
        assert _validate_reformulated_query("How do I install FastAPI?") is True

    def test_accepts_valid_explain_query(self):
        assert _validate_reformulated_query("Explain Python decorators") is True

    def test_accepts_valid_when_should_query(self):
        assert (
            _validate_reformulated_query("When should I use async functions in Python?")
            is True
        )

    def test_accepts_short_query_without_question_mark(self):
        # Short queries without periods are acceptable
        assert _validate_reformulated_query("Python async functions") is True

    # Invalid queries (should return False - answer patterns)
    def test_rejects_answer_with_because(self):
        assert (
            _validate_reformulated_query(
                "You should use async functions because they are faster for I/O operations."
            )
            is False
        )

    def test_rejects_answer_with_this_means(self):
        assert (
            _validate_reformulated_query(
                "This means that decorators modify function behavior using the @ syntax."
            )
            is False
        )

    def test_rejects_answer_with_it_is(self):
        assert (
            _validate_reformulated_query(
                "It is a modern Python web framework for building APIs."
            )
            is False
        )

    def test_rejects_definitional_answer(self):
        assert (
            _validate_reformulated_query(
                "FastAPI is a modern Python web framework for building APIs with automatic validation."
            )
            is False
        )

    def test_rejects_long_statement_without_question(self):
        # Long statement (>80 chars), no question words, ends with period
        assert (
            _validate_reformulated_query(
                "Decorators are special functions that wrap other functions to modify their behavior without changing the source code directly."
            )
            is False
        )

    def test_rejects_statement_form_answer(self):
        # Capital start, >5 words, ends with period, no question words
        assert (
            _validate_reformulated_query(
                "Python uses the async and await keywords for asynchronous programming."
            )
            is False
        )

    # Edge cases
    def test_rejects_empty_string(self):
        assert _validate_reformulated_query("") is False

    def test_rejects_whitespace_only(self):
        assert _validate_reformulated_query("   ") is False

    def test_accepts_query_with_is_a_but_has_question_mark(self):
        # "is a" is okay if it's part of a question
        assert _validate_reformulated_query("What is a Python decorator?") is True

    def test_accepts_long_query_with_question_words(self):
        # Long text is okay if it has question words
        assert (
            _validate_reformulated_query(
                "How do I configure FastAPI to use async database connections with SQLAlchemy and handle connection pooling?"
            )
            is True
        )

    def test_rejects_you_should_pattern(self):
        assert (
            _validate_reformulated_query(
                "You should use Pydantic models for validation."
            )
            is False
        )

    def test_rejects_you_can_pattern(self):
        assert (
            _validate_reformulated_query(
                "You can install it using pip install fastapi."
            )
            is False
        )

    def test_accepts_command_style_query(self):
        # Imperative queries are valid
        assert (
            _validate_reformulated_query("Show examples of Python decorators") is True
        )
