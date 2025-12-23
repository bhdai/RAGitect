"""Tests for RAG prompt composition module.

Tests the modular prompt system following TDD approach.
"""


class TestBuildRagSystemPrompt:
    """Test suite for build_rag_system_prompt function."""

    def test_includes_all_default_components(self):
        """Test that modular prompt includes all components by default."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [
            {"document_name": "test.pdf", "similarity": 0.9, "content": "Test content"}
        ]

        prompt = build_rag_system_prompt(chunks)

        # Should include all modules
        assert "<system_instructions>" in prompt
        assert "<citation_rules>" in prompt
        assert "<citation_examples>" in prompt
        assert "<documents>" in prompt
        assert '<document index="1">' in prompt

    def test_includes_base_prompt_components(self):
        """Test that base prompt includes required identity and constraints."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [{"document_name": "test.pdf", "similarity": 0.9, "content": "Test"}]

        prompt = build_rag_system_prompt(chunks)

        # Check for base prompt elements
        assert "IDENTITY" in prompt
        assert "research librarian" in prompt.lower()
        assert "ABSOLUTE CONSTRAINTS" in prompt
        assert "CAPABILITIES" in prompt

    def test_without_examples(self):
        """Test prompt composition with examples disabled."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [{"document_name": "test.pdf", "similarity": 0.9, "content": "Test"}]

        prompt = build_rag_system_prompt(chunks, include_examples=False)

        assert "<citation_rules>" in prompt
        assert "<citation_examples>" not in prompt

    def test_without_citations(self):
        """Test prompt composition with citations disabled."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [{"document_name": "test.pdf", "similarity": 0.9, "content": "Test"}]

        prompt = build_rag_system_prompt(chunks, include_citations=False)

        assert "<citation_rules>" not in prompt
        assert "<citation_examples>" not in prompt
        assert "<documents>" in prompt  # Documents still included

    def test_enhanced_negative_constraints_present(self):
        """Test that SurfSense-inspired negative constraints are included."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [{"document_name": "test.pdf", "similarity": 0.9, "content": "Test"}]

        prompt = build_rag_system_prompt(chunks)

        # Check for enhanced constraints from SurfSense analysis
        assert "NEVER fabricate" in prompt or "DO NOT fabricate" in prompt
        assert "EXPLICITLY FORBIDDEN" in prompt or "FORBIDDEN" in prompt
        assert "Markdown links" in prompt

    def test_chunk_formatting_with_metadata(self):
        """Test that chunks are formatted as XML with correct structure."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [
            {"document_name": "doc1.pdf", "similarity": 0.95, "content": "Content A"},
            {"document_name": "doc2.md", "similarity": 0.85, "content": "Content B"},
        ]

        prompt = build_rag_system_prompt(chunks)

        # Verify new XML format
        assert '<document index="1">' in prompt
        assert '<document index="2">' in prompt
        assert "<source>doc1.pdf</source>" in prompt
        assert "<source>doc2.md</source>" in prompt
        assert "<content>" in prompt
        assert "</content>" in prompt
        assert "Content A" in prompt
        assert "Content B" in prompt
        # Verify old format removed
        assert "[Chunk 1]" not in prompt
        assert "(From:" not in prompt
        # Verify similarity score is hidden (AC2)
        assert "0.95" not in prompt
        assert "0.85" not in prompt
        assert "Similarity:" not in prompt

    def test_empty_chunks_handled(self):
        """Test that empty chunk list produces valid prompt."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        prompt = build_rag_system_prompt([])

        assert "<documents>" in prompt
        assert "No relevant context found" in prompt

    def test_custom_instructions_included(self):
        """Test that custom instructions are properly included."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [{"document_name": "test.pdf", "similarity": 0.9, "content": "Test"}]

        prompt = build_rag_system_prompt(
            chunks, custom_instructions="Focus on code examples only."
        )

        assert "<custom_instructions>" in prompt
        assert "Focus on code examples only." in prompt

    def test_xml_escaping_in_content(self):
        """Test that XML special characters in content are escaped."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [
            {
                "document_name": "test.pdf",
                "similarity": 0.9,
                "content": "Code: if (a < b && c > d) { return; }",
            }
        ]

        prompt = build_rag_system_prompt(chunks)

        # XML special chars should be escaped
        assert "&lt;" in prompt
        assert "&gt;" in prompt
        assert "&amp;" in prompt

    def test_output_format_instructions_present(self):
        """Test that output format instructions are included."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [{"document_name": "test.pdf", "similarity": 0.9, "content": "Test"}]

        prompt = build_rag_system_prompt(chunks)

        assert "RESPONSE FORMAT" in prompt
        assert "OUTPUT STYLE" in prompt
        assert "markdown" in prompt.lower()

    def test_build_rag_system_prompt_uses_xml_format(self):
        """Test that context chunks are formatted as structured XML."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [
            {"content": "Chunk 1", "document_name": "a.md", "similarity": 0.9},
            {"content": "Chunk 2", "document_name": "b.md", "similarity": 0.8},
        ]
        prompt = build_rag_system_prompt(chunks)

        # Verify new XML format
        assert '<document index="1">' in prompt
        assert '<document index="2">' in prompt
        assert "<source>" in prompt
        assert "<content>" in prompt

        # Verify old format removed
        assert "[Chunk 1]" not in prompt
        assert "(From:" not in prompt
        assert "Similarity:" not in prompt

    def test_build_rag_system_prompt_hides_similarity(self):
        """Test that similarity score is NOT included in LLM context."""
        from ragitect.prompts.rag_prompts import build_rag_system_prompt

        chunks = [{"content": "X", "document_name": "y.md", "similarity": 0.12345}]
        prompt = build_rag_system_prompt(chunks)

        assert "0.12" not in prompt
        assert "similarity" not in prompt.lower().replace("<citation", "").replace(
            "citation_", ""
        )  # avoid matching XML tags


class TestCitationFormat:
    """Test suite for citation format (AC5: [cite: N] format)."""

    def test_citation_instructions_use_cite_format(self):
        """Test that citation instructions use [cite: N] format (1-based)."""
        from ragitect.prompts.rag_prompts import RAG_CITATION_INSTRUCTIONS

        # Should use [cite: 1], [cite: 2] format
        assert "[cite: 1]" in RAG_CITATION_INSTRUCTIONS
        assert "[cite: 2]" in RAG_CITATION_INSTRUCTIONS

        # Should NOT use bare [1], [2] format (old format)
        assert "cite:" in RAG_CITATION_INSTRUCTIONS.lower()

    def test_citation_examples_use_cite_format(self):
        """Test that citation examples use [cite: N] format (1-based)."""
        from ragitect.prompts.rag_prompts import RAG_CITATION_EXAMPLES

        # Examples should demonstrate [cite: N] format
        assert "[cite: 1]" in RAG_CITATION_EXAMPLES
        assert "[cite: 2]" in RAG_CITATION_EXAMPLES

    def test_forbidden_patterns_include_old_format(self):
        """Test that old [N] format is listed as forbidden."""
        from ragitect.prompts.rag_prompts import RAG_CITATION_INSTRUCTIONS

        # The old bare [N] format should be forbidden
        # Check for mention of forbidden patterns including brackets
        assert "FORBIDDEN" in RAG_CITATION_INSTRUCTIONS


class TestFormatChunkAsXml:
    """Test suite for format_chunk_as_xml() helper function."""

    def test_format_chunk_as_xml_basic(self):
        """Test basic XML structure output."""
        from ragitect.prompts.rag_prompts import format_chunk_as_xml

        chunk = {"content": "Hello", "document_name": "test.md", "similarity": 0.85}
        result = format_chunk_as_xml(chunk, 1)

        assert '<document index="1">' in result
        assert "<source>test.md</source>" in result
        assert "<content>" in result
        assert "Hello" in result
        assert "</content>" in result
        assert "</document>" in result

    def test_format_chunk_as_xml_hides_similarity(self):
        """Test that similarity score is NOT included in output."""
        from ragitect.prompts.rag_prompts import format_chunk_as_xml

        chunk = {"content": "Hi", "document_name": "x.md", "similarity": 0.99}
        result = format_chunk_as_xml(chunk, 1)

        assert "0.99" not in result
        assert "similarity" not in result.lower()
        assert "relevance" not in result.lower()

    def test_format_chunk_as_xml_escapes_special_chars(self):
        """Test XML escaping for special characters."""
        from ragitect.prompts.rag_prompts import format_chunk_as_xml

        chunk = {
            "content": "if (a < b && c > d)",
            "document_name": "code<>.md",
            "similarity": 0.5,
        }
        result = format_chunk_as_xml(chunk, 1)

        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result

    def test_format_chunk_as_xml_handles_missing_fields(self):
        """Test fallback handling for missing fields."""
        from ragitect.prompts.rag_prompts import format_chunk_as_xml

        chunk = {"content": "Just content"}
        result = format_chunk_as_xml(chunk, 3)

        assert '<document index="3">' in result
        assert "<source>Unknown</source>" in result

    def test_format_chunk_as_xml_different_indices(self):
        """Test that various 1-based indices work correctly."""
        from ragitect.prompts.rag_prompts import format_chunk_as_xml

        chunk = {"content": "Test", "document_name": "doc.md", "similarity": 0.5}

        assert '<document index="1">' in format_chunk_as_xml(chunk, 1)
        assert '<document index="5">' in format_chunk_as_xml(chunk, 5)
        assert '<document index="10">' in format_chunk_as_xml(chunk, 10)


class TestPromptConstants:
    """Test suite for prompt constant values."""

    def test_rag_base_prompt_not_empty(self):
        """Test that RAG_BASE_PROMPT is defined and not empty."""
        from ragitect.prompts.rag_prompts import RAG_BASE_PROMPT

        assert RAG_BASE_PROMPT
        assert len(RAG_BASE_PROMPT) > 100

    def test_rag_citation_instructions_not_empty(self):
        """Test that RAG_CITATION_INSTRUCTIONS is defined and not empty."""
        from ragitect.prompts.rag_prompts import RAG_CITATION_INSTRUCTIONS

        assert RAG_CITATION_INSTRUCTIONS
        assert len(RAG_CITATION_INSTRUCTIONS) > 100

    def test_rag_citation_examples_not_empty(self):
        """Test that RAG_CITATION_EXAMPLES is defined and not empty."""
        from ragitect.prompts.rag_prompts import RAG_CITATION_EXAMPLES

        assert RAG_CITATION_EXAMPLES
        assert len(RAG_CITATION_EXAMPLES) > 100

    def test_rag_output_format_not_empty(self):
        """Test that RAG_OUTPUT_FORMAT is defined and not empty."""
        from ragitect.prompts.rag_prompts import RAG_OUTPUT_FORMAT

        assert RAG_OUTPUT_FORMAT
        assert len(RAG_OUTPUT_FORMAT) > 50
