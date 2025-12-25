"""Tests for RAG agent Pydantic schemas.

Tests for SearchStrategy and Search models used in strategy generation.
"""

import pytest


class TestSearchModel:
    """Tests for Search Pydantic model."""

    def test_search_creation_valid(self):
        """Test Search model with valid data."""
        from ragitect.agents.rag.schemas import Search

        search = Search(term="FastAPI installation", reasoning="Main topic query")

        assert search.term == "FastAPI installation"
        assert search.reasoning == "Main topic query"

    def test_search_requires_term(self):
        """Test Search model requires term field."""
        from ragitect.agents.rag.schemas import Search
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            Search(reasoning="Only reasoning provided")

        assert "term" in str(exc_info.value)

    def test_search_requires_reasoning(self):
        """Test Search model requires reasoning field."""
        from ragitect.agents.rag.schemas import Search
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            Search(term="Some search term")

        assert "reasoning" in str(exc_info.value)

    def test_search_string_fields(self):
        """Test Search model fields are strings."""
        from ragitect.agents.rag.schemas import Search

        search = Search(term="query", reasoning="reason")

        assert isinstance(search.term, str)
        assert isinstance(search.reasoning, str)


class TestSearchStrategyModel:
    """Tests for SearchStrategy Pydantic model."""

    def test_strategy_creation_single_search(self):
        """Test SearchStrategy with a single search term."""
        from ragitect.agents.rag.schemas import Search, SearchStrategy

        strategy = SearchStrategy(
            reasoning="Simple direct question",
            searches=[Search(term="Python basics", reasoning="Direct query")],
        )

        assert strategy.reasoning == "Simple direct question"
        assert len(strategy.searches) == 1
        assert strategy.searches[0].term == "Python basics"

    def test_strategy_creation_multiple_searches(self):
        """Test SearchStrategy with multiple search terms (2-5)."""
        from ragitect.agents.rag.schemas import Search, SearchStrategy

        searches = [
            Search(term="FastAPI routing", reasoning="API routing patterns"),
            Search(term="FastAPI dependencies", reasoning="Dependency injection"),
            Search(term="FastAPI middleware", reasoning="Request processing"),
        ]
        strategy = SearchStrategy(
            reasoning="Complex query about FastAPI architecture",
            searches=searches,
        )

        assert len(strategy.searches) == 3
        assert all(isinstance(s, Search) for s in strategy.searches)

    def test_strategy_requires_reasoning(self):
        """Test SearchStrategy requires reasoning field."""
        from ragitect.agents.rag.schemas import Search, SearchStrategy
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            SearchStrategy(searches=[Search(term="query", reasoning="reason")])

        assert "reasoning" in str(exc_info.value)

    def test_strategy_requires_searches(self):
        """Test SearchStrategy requires searches field."""
        from ragitect.agents.rag.schemas import SearchStrategy
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            SearchStrategy(reasoning="Some reasoning without searches")

        assert "searches" in str(exc_info.value)

    def test_strategy_searches_is_list(self):
        """Test SearchStrategy searches field is a list."""
        from ragitect.agents.rag.schemas import Search, SearchStrategy

        strategy = SearchStrategy(
            reasoning="Test",
            searches=[Search(term="a", reasoning="b")],
        )

        assert isinstance(strategy.searches, list)

    def test_strategy_empty_searches_allowed(self):
        """Test SearchStrategy can have empty searches list (edge case)."""
        from ragitect.agents.rag.schemas import SearchStrategy

        # This should be valid at model level, but business logic may require >= 1
        strategy = SearchStrategy(
            reasoning="No searches needed",
            searches=[],
        )

        assert strategy.searches == []

    def test_strategy_max_searches_not_enforced_at_model_level(self):
        """Test SearchStrategy doesn't enforce max searches (logic enforces 5)."""
        from ragitect.agents.rag.schemas import Search, SearchStrategy

        # Create 7 searches - model accepts, logic limits to 5
        searches = [
            Search(term=f"search_{i}", reasoning=f"reason_{i}") for i in range(7)
        ]
        strategy = SearchStrategy(
            reasoning="Many searches",
            searches=searches,
        )

        assert len(strategy.searches) == 7  # Model doesn't limit

    def test_strategy_serialization(self):
        """Test SearchStrategy serializes to dict correctly."""
        from ragitect.agents.rag.schemas import Search, SearchStrategy

        strategy = SearchStrategy(
            reasoning="Test reasoning",
            searches=[
                Search(term="term1", reasoning="reason1"),
                Search(term="term2", reasoning="reason2"),
            ],
        )

        data = strategy.model_dump()

        assert data["reasoning"] == "Test reasoning"
        assert len(data["searches"]) == 2
        assert data["searches"][0]["term"] == "term1"
