"""Tests for engine.py"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ragitect.engine import ChatEngine


class TestChatEngineFormatChatHistory:
    """Test chat history formatting"""

    def test_formats_empty_history(self):
        result = ChatEngine._format_chat_history_for_prompt([])
        assert result == []

    def test_formats_user_message(self):
        history = [{"role": "user", "content": "Hello"}]
        result = ChatEngine._format_chat_history_for_prompt(history)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_formats_assistant_message(self):
        history = [{"role": "assistant", "content": "Hi there!"}]
        result = ChatEngine._format_chat_history_for_prompt(history)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hi there!"

    def test_formats_conversation(self):
        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
        result = ChatEngine._format_chat_history_for_prompt(history)

        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], HumanMessage)

    def test_preserves_content(self):
        history = [
            {"role": "user", "content": "Specific question here"},
            {"role": "assistant", "content": "Specific answer here"},
        ]
        result = ChatEngine._format_chat_history_for_prompt(history)

        assert result[0].content == "Specific question here"
        assert result[1].content == "Specific answer here"

    def test_handles_unknown_role_gracefully(self):
        history = [{"role": "unknown", "content": "test"}]

        # Should not raise, just logs warning
        result = ChatEngine._format_chat_history_for_prompt(history)

        # Unknown roles are skipped
        assert len(result) == 0


class TestChatEngineBuildPromptMessages:
    """Test prompt message building"""

    def test_includes_system_message(self):
        system = "You are a helpful assistant"
        context = "Context text"
        query = "User query"
        history = []

        messages = ChatEngine._build_prompt_messages(system, context, query, history)

        assert len(messages) >= 1
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == system

    def test_includes_context_and_query(self):
        system = "System"
        context = "Retrieved context"
        query = "What is this?"
        history = []

        messages = ChatEngine._build_prompt_messages(system, context, query, history)

        # Last message should contain context and query
        last_message = messages[-1]
        assert isinstance(last_message, HumanMessage)
        assert "Retrieved context" in last_message.content
        assert "What is this?" in last_message.content

    def test_includes_chat_history(self):
        system = "System"
        context = "Context"
        query = "Query"
        history = [
            HumanMessage(content="Previous question"),
            AIMessage(content="Previous answer"),
        ]

        messages = ChatEngine._build_prompt_messages(system, context, query, history)

        # Should have: system + history + context/query
        assert len(messages) >= 4
        assert messages[1].content == "Previous question"
        assert messages[2].content == "Previous answer"

    def test_correct_message_order(self):
        system = "System"
        context = "Context"
        query = "Query"
        history = [HumanMessage(content="History")]

        messages = ChatEngine._build_prompt_messages(system, context, query, history)

        # Order: system, history, then context+query
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "History"
        assert isinstance(messages[-1], HumanMessage)
        assert "Context" in messages[-1].content

    def test_handles_empty_history(self):
        system = "System"
        context = "Context"
        query = "Query"
        history = []

        messages = ChatEngine._build_prompt_messages(system, context, query, history)

        # Should have: system + context/query
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
