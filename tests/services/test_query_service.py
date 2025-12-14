"""Tests for query_service.py"""

from unittest.mock import AsyncMock, patch
import pytest

from ragitect.services.query_service import (
    format_chat_history,
    _build_reformulation_prompt,
    adaptive_query_processing,
    reformulate_query_with_chat_history,
)

pytestmark = [pytest.mark.asyncio]


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

    def test_prompt_instructs_no_prefixes(self):
        """Phase 2: Verify prompt tells LLM not to use prefixes"""
        prompt = _build_reformulation_prompt(
            "test query", "<chat_history>\n</chat_history>"
        )

        # Check for anti-prefix instructions
        assert "no label" in prompt.lower() or "no prefix" in prompt.lower()
        assert "directly" in prompt.lower() or "only" in prompt.lower()


class TestAdaptiveQueryProcessing:
    """Test adaptive query processing based on complexity classification"""

    async def test_simple_query_returns_original(self):
        """Simple query with no history returns original query unchanged"""
        mock_llm = AsyncMock()
        user_query = "What is Python?"
        chat_history = []

        result = await adaptive_query_processing(mock_llm, user_query, chat_history)

        assert result == user_query
        mock_llm.assert_not_called()  # No LLM call for simple queries

    async def test_ambiguous_query_triggers_reformulation(self):
        """Query with pronoun and history triggers reformulation"""
        mock_llm = AsyncMock()
        user_query = "How do I install it?"
        chat_history = [{"role": "user", "content": "Tell me about FastAPI"}]

        with patch(
            "ragitect.services.query_service.reformulate_query_with_chat_history"
        ) as mock_reformulate:
            mock_reformulate.return_value = "How do I install FastAPI?"

            result = await adaptive_query_processing(mock_llm, user_query, chat_history)

            assert result == "How do I install FastAPI?"
            mock_reformulate.assert_called_once_with(mock_llm, user_query, chat_history)

    async def test_complex_query_triggers_reformulation(self):
        """Comparison query triggers reformulation"""
        mock_llm = AsyncMock()
        user_query = "Compare FastAPI vs Flask"
        chat_history = []

        with patch(
            "ragitect.services.query_service.reformulate_query_with_chat_history"
        ) as mock_reformulate:
            mock_reformulate.return_value = "Compare FastAPI vs Flask for web APIs"

            result = await adaptive_query_processing(mock_llm, user_query, chat_history)

            assert result == "Compare FastAPI vs Flask for web APIs"
            mock_reformulate.assert_called_once()


class TestReformulateQueryWithChatHistory:
    """Test query reformulation with chat history"""

    async def test_reformulates_query_successfully(self):
        """Successfully reformulates query using LLM"""
        mock_llm = AsyncMock()
        user_query = "How do I install it?"
        chat_history = [{"role": "user", "content": "Tell me about FastAPI"}]

        with patch("ragitect.services.query_service.generate_response") as mock_gen:
            mock_gen.return_value = "How do I install FastAPI?"

            result = await reformulate_query_with_chat_history(
                mock_llm, user_query, chat_history
            )

            assert result == "How do I install FastAPI?"
            mock_gen.assert_called_once()

    async def test_returns_original_on_empty_response(self):
        """Returns original query when LLM returns empty response"""
        mock_llm = AsyncMock()
        user_query = "How do I install it?"
        chat_history = [{"role": "user", "content": "Tell me about FastAPI"}]

        with patch("ragitect.services.query_service.generate_response") as mock_gen:
            mock_gen.return_value = "   "  # Whitespace only

            result = await reformulate_query_with_chat_history(
                mock_llm, user_query, chat_history
            )

            assert result == user_query

    async def test_returns_original_on_exception(self):
        """Returns original query when LLM raises exception"""
        mock_llm = AsyncMock()
        user_query = "How do I install it?"
        chat_history = [{"role": "user", "content": "Tell me about FastAPI"}]

        with patch("ragitect.services.query_service.generate_response") as mock_gen:
            mock_gen.side_effect = Exception("LLM API error")

            result = await reformulate_query_with_chat_history(
                mock_llm, user_query, chat_history
            )

            assert result == user_query

    async def test_limits_history_to_10_messages(self):
        """Limits history to last 10 messages for token efficiency"""
        mock_llm = AsyncMock()
        user_query = "What is that?"
        # Create 15 messages
        chat_history = [{"role": "user", "content": f"Message {i}"} for i in range(15)]

        with patch("ragitect.services.query_service.generate_response") as mock_gen:
            mock_gen.return_value = "What is that thing you mentioned?"

            with patch(
                "ragitect.services.query_service.format_chat_history"
            ) as mock_format:
                mock_format.return_value = "<chat_history></chat_history>"

                await reformulate_query_with_chat_history(
                    mock_llm, user_query, chat_history
                )

                # Should only pass last 10 messages
                call_args = mock_format.call_args[0][0]
                assert len(call_args) == 10
                assert call_args[0]["content"] == "Message 5"  # First of last 10
