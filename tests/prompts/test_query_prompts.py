"""Tests for query prompt composition module.

Tests the modular query prompt system following TDD approach.
"""


class TestBuildReformulationPrompt:
    """Test suite for build_reformulation_prompt function."""

    def test_includes_all_components(self):
        """Test reformulation prompt includes all modules."""
        from ragitect.prompts.query_prompts import build_reformulation_prompt

        prompt = build_reformulation_prompt("test query", "<chat_history/>")

        assert "query preprocessor" in prompt.lower()
        assert "JSON" in prompt
        assert "CRITICAL RULES" in prompt
        assert "<example>" in prompt or "example" in prompt.lower()
        assert "test query" in prompt

    def test_includes_output_format_specification(self):
        """Test that output format specifies JSON structure."""
        from ragitect.prompts.query_prompts import build_reformulation_prompt

        prompt = build_reformulation_prompt("test query", "<chat_history/>")

        assert "reasoning" in prompt
        assert "reformulated_query" in prompt
        assert "was_modified" in prompt

    def test_includes_chat_history(self):
        """Test that chat history is included in prompt."""
        from ragitect.prompts.query_prompts import build_reformulation_prompt

        history = "<chat_history><message>Previous message</message></chat_history>"
        prompt = build_reformulation_prompt("test query", history)

        assert "Previous message" in prompt
        assert "Conversation History" in prompt

    def test_includes_critical_rules(self):
        """Test that critical rules for reformulation are included."""
        from ragitect.prompts.query_prompts import build_reformulation_prompt

        prompt = build_reformulation_prompt("test query", "<chat_history/>")

        # Check for key rules
        assert "UNCHANGED" in prompt or "unchanged" in prompt
        assert "was_modified=false" in prompt or "was_modified" in prompt
        assert "NEVER" in prompt

    def test_includes_examples(self):
        """Test that few-shot examples are included."""
        from ragitect.prompts.query_prompts import build_reformulation_prompt

        prompt = build_reformulation_prompt("test query", "<chat_history/>")

        # Check for examples
        assert "FastAPI" in prompt or "example" in prompt.lower()
        assert "History:" in prompt


class TestBuildRelevanceGradingPrompt:
    """Test suite for build_relevance_grading_prompt function."""

    def test_includes_grading_instructions(self):
        """Test relevance grading prompt includes grading instructions."""
        from ragitect.prompts.query_prompts import build_relevance_grading_prompt

        prompt = build_relevance_grading_prompt(
            "Python features", "Python is versatile"
        )

        assert "grader" in prompt.lower()
        assert "relevance" in prompt.lower()

    def test_includes_binary_output_format(self):
        """Test that output format specifies binary yes/no."""
        from ragitect.prompts.query_prompts import build_relevance_grading_prompt

        prompt = build_relevance_grading_prompt("test query", "test document")

        assert "yes" in prompt.lower()
        assert "no" in prompt.lower()
        assert "binary" in prompt.lower() or "score" in prompt.lower()

    def test_includes_query_and_document(self):
        """Test that query and document are included."""
        from ragitect.prompts.query_prompts import build_relevance_grading_prompt

        prompt = build_relevance_grading_prompt(
            "What is Python used for?",
            "Python is a programming language for web development.",
        )

        assert "What is Python used for?" in prompt
        assert "Python is a programming language for web development." in prompt


class TestPromptConstants:
    """Test suite for query prompt constant values."""

    def test_reformulation_base_instructions_not_empty(self):
        """Test that REFORMULATION_BASE_INSTRUCTIONS is defined and not empty."""
        from ragitect.prompts.query_prompts import REFORMULATION_BASE_INSTRUCTIONS

        assert REFORMULATION_BASE_INSTRUCTIONS
        assert len(REFORMULATION_BASE_INSTRUCTIONS) > 20

    def test_reformulation_output_format_not_empty(self):
        """Test that REFORMULATION_OUTPUT_FORMAT is defined and not empty."""
        from ragitect.prompts.query_prompts import REFORMULATION_OUTPUT_FORMAT

        assert REFORMULATION_OUTPUT_FORMAT
        assert len(REFORMULATION_OUTPUT_FORMAT) > 20

    def test_reformulation_critical_rules_not_empty(self):
        """Test that REFORMULATION_CRITICAL_RULES is defined and not empty."""
        from ragitect.prompts.query_prompts import REFORMULATION_CRITICAL_RULES

        assert REFORMULATION_CRITICAL_RULES
        assert len(REFORMULATION_CRITICAL_RULES) > 50

    def test_reformulation_examples_not_empty(self):
        """Test that REFORMULATION_EXAMPLES is defined and not empty."""
        from ragitect.prompts.query_prompts import REFORMULATION_EXAMPLES

        assert REFORMULATION_EXAMPLES
        assert len(REFORMULATION_EXAMPLES) > 100

    def test_relevance_grading_instructions_not_empty(self):
        """Test that RELEVANCE_GRADING_INSTRUCTIONS is defined and not empty."""
        from ragitect.prompts.query_prompts import RELEVANCE_GRADING_INSTRUCTIONS

        assert RELEVANCE_GRADING_INSTRUCTIONS
        assert len(RELEVANCE_GRADING_INSTRUCTIONS) > 30

    def test_relevance_output_format_not_empty(self):
        """Test that RELEVANCE_OUTPUT_FORMAT is defined and not empty."""
        from ragitect.prompts.query_prompts import RELEVANCE_OUTPUT_FORMAT

        assert RELEVANCE_OUTPUT_FORMAT
        assert len(RELEVANCE_OUTPUT_FORMAT) > 30
