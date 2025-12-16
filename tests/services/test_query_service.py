"""Tests for query_service.py"""

from unittest.mock import AsyncMock, patch
import pytest

from ragitect.services.query_service import (
    format_chat_history,
    _build_reformulation_prompt,
    _parse_reformulation_response,
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

    def test_prompt_includes_guardrails(self):
        """Verify prompt includes research-backed guardrails against over-reformulation"""
        user_query = "How do I use it?"
        formatted_history = """<chat_history>
<message role="user">What is FastAPI?</message>
<message role="assistant">FastAPI is a modern Python web framework.</message>
</chat_history>"""

        prompt = _build_reformulation_prompt(user_query, formatted_history)

        # Check for critical guardrails (research-backed)
        assert "SELF-CONTAINED" in prompt or "self-contained" in prompt
        assert "UNCHANGED" in prompt or "unchanged" in prompt
        assert "NEVER" in prompt  # Never add info not in history
        # Examples help LLM understand when NOT to reformulate
        assert "Example" in prompt or "example" in prompt

    def test_prompt_instructs_no_prefixes(self):
        """Phase 2: Verify prompt uses JSON format to avoid prefix issues"""
        prompt = _build_reformulation_prompt(
            "test query", "<chat_history>\n</chat_history>"
        )

        # JSON format inherently avoids prefix issues - verify JSON structure is requested
        assert "json" in prompt.lower()
        assert "reformulated_query" in prompt


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


# =============================================================================
# 🔴 RED PHASE: Tests for Structured Output Parsing
# These tests define expected behavior - they should FAIL until implementation
# =============================================================================


class TestParseReformulationResponse:
    """Tests for _parse_reformulation_response function.

    🔴 RED PHASE: These tests define expected behavior.
    They should FAIL until implementation is complete.
    """

    def test_valid_json_response(self):
        """JSON response extracts reformulated_query correctly."""
        response = '{"reasoning": "test", "reformulated_query": "What is FastAPI?", "was_modified": true}'
        result = _parse_reformulation_response(response, "original")
        assert result == "What is FastAPI?"

    def test_json_with_code_blocks(self):
        """JSON wrapped in markdown code blocks is parsed."""
        response = '```json\n{"reasoning": "test", "reformulated_query": "Test query", "was_modified": false}\n```'
        result = _parse_reformulation_response(response, "original")
        assert result == "Test query"

    def test_plain_text_with_explanation(self):
        """Plain text with parenthetical explanation uses fallback."""
        response = "What is FastAPI?\n\n(I reformulated because...)"
        result = _parse_reformulation_response(response, "original")
        assert result == "What is FastAPI?"

    def test_plain_text_with_prefix(self):
        """Plain text with 'Output:' prefix is cleaned."""
        response = "Output: What is FastAPI?"
        result = _parse_reformulation_response(response, "original")
        assert result == "What is FastAPI?"

    def test_empty_response_returns_original(self):
        """Empty response returns original query."""
        result = _parse_reformulation_response("", "original query")
        assert result == "original query"

    def test_whitespace_only_returns_original(self):
        """Whitespace-only response returns original query."""
        result = _parse_reformulation_response("   \n\n  ", "original query")
        assert result == "original query"

    def test_was_modified_true_extracts_query(self):
        """JSON with was_modified=true extracts reformulated_query."""
        response = '{"reasoning": "replaced pronoun", "reformulated_query": "How do I install FastAPI?", "was_modified": true}'
        result = _parse_reformulation_response(response, "How do I install it?")
        assert result == "How do I install FastAPI?"

    def test_was_modified_false_extracts_query(self):
        """JSON with was_modified=false extracts unchanged query."""
        response = '{"reasoning": "already self-contained", "reformulated_query": "What is Python?", "was_modified": false}'
        result = _parse_reformulation_response(response, "What is Python?")
        assert result == "What is Python?"

    def test_json_embedded_in_verbose_response(self):
        """JSON embedded in verbose response with markdown is extracted."""
        response = """### User:
The conversation history is empty, so we need to check if the query is self-contained.

### Analysis
The query contains a pronoun "it" that refers to "Quickshell" within the same sentence.

### Output

{"reasoning": "Query is self-contained.", "reformulated_query": "What is Quickshell and how can I install it?", "was_modified": false}"""
        result = _parse_reformulation_response(response, "original")
        assert result == "What is Quickshell and how can I install it?"


class TestBuildReformulationPromptStructuredOutput:
    """Tests for updated _build_reformulation_prompt with JSON output.

    🔴 RED PHASE: Define expected new prompt structure.
    """

    def test_prompt_requests_json_output(self):
        """Prompt should request JSON output format."""
        prompt = _build_reformulation_prompt(
            "test query", "<chat_history></chat_history>"
        )
        assert "JSON" in prompt or "json" in prompt
        assert "reformulated_query" in prompt
        assert "was_modified" in prompt

    def test_prompt_has_few_shot_examples(self):
        """Prompt should contain 4+ few-shot examples."""
        prompt = _build_reformulation_prompt(
            "test query", "<chat_history></chat_history>"
        )
        example_count = prompt.count("<example>")
        assert example_count >= 4, f"Expected 4+ examples, found {example_count}"

    def test_prompt_includes_same_sentence_rule(self):
        """Prompt should explain same-sentence pronoun handling."""
        prompt = _build_reformulation_prompt(
            "test query", "<chat_history></chat_history>"
        )
        assert "same sentence" in prompt.lower() or "same-sentence" in prompt.lower()

    def test_prompt_includes_quickshell_example(self):
        """Prompt should include the Quickshell self-contained example."""
        prompt = _build_reformulation_prompt(
            "test query", "<chat_history></chat_history>"
        )
        assert "Quickshell" in prompt or "quickshell" in prompt


class TestReformulateQueryWithChatHistoryParsing:
    """Integration tests for reformulate_query_with_chat_history parsing.

    🔴 RED PHASE: Ensure end-to-end parsing works correctly.
    """

    async def test_parses_json_response_correctly(self):
        """When LLM returns JSON, extracts reformulated_query field."""
        mock_llm = AsyncMock()

        with patch("ragitect.services.query_service.generate_response") as mock_gen:
            mock_gen.return_value = '{"reasoning": "test", "reformulated_query": "What is FastAPI?", "was_modified": false}'

            result = await reformulate_query_with_chat_history(
                mock_llm, "What is FastAPI?", []
            )

            assert result == "What is FastAPI?"
            assert "reasoning" not in result

    async def test_cleans_plain_text_with_explanation(self):
        """When LLM returns text with explanation, removes explanation."""
        mock_llm = AsyncMock()

        with patch("ragitect.services.query_service.generate_response") as mock_gen:
            mock_gen.return_value = (
                "What is FastAPI?\n\n(I kept the query unchanged because...)"
            )

            result = await reformulate_query_with_chat_history(
                mock_llm, "What is FastAPI?", []
            )

            assert result == "What is FastAPI?"
            assert "because" not in result
