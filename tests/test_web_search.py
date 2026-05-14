"""Tests for the web_search tool — env-key gating + result-shape contract.

The Tavily API is never called in tests; ``_do_search`` is monkey-patched.
"""

from __future__ import annotations

import sys

from taste_agent.tools.web_search import _ENV_KEY, web_search

# The package's ``__init__.py`` re-exports the StructuredTool as
# ``taste_agent.tools.web_search``, which shadows the module attribute. Grab
# the real module from sys.modules so monkeypatch can target it.
_ws_module = sys.modules["taste_agent.tools.web_search"]


def test_web_search_returns_sentinel_when_no_api_key(monkeypatch):
    monkeypatch.delenv(_ENV_KEY, raising=False)
    result = web_search.invoke({"query": "best cappuccino Belgrade"})
    assert len(result) == 1
    assert result[0]["status"] == "error"
    assert _ENV_KEY in result[0]["content"]


def test_web_search_returns_normalized_results(monkeypatch):
    monkeypatch.setenv(_ENV_KEY, "fake-key")

    def fake_search(query: str, max_results: int):
        assert query == "Iva restaurant Belgrade reviews"
        assert max_results == 5
        return [
            {
                "title": "Iva Review",
                "url": "https://example.com/iva",
                "content": "Loved the tasting menu.",
                "score": 0.92,
            }
        ]

    monkeypatch.setattr(_ws_module, "_do_search", fake_search)
    result = web_search.invoke(
        {"query": "Iva restaurant Belgrade reviews", "max_results": 5}
    )
    assert len(result) == 1
    assert result[0]["title"] == "Iva Review"
    assert result[0]["url"].startswith("https://")
    assert isinstance(result[0]["score"], float)


def test_web_search_max_results_defaults_to_five(monkeypatch):
    monkeypatch.setenv(_ENV_KEY, "fake-key")
    captured = {}

    def fake_search(query: str, max_results: int):
        captured["max"] = max_results
        return [
            {"title": "Web search unavailable", "url": "", "content": "fallback", "score": 0.0}
        ]

    monkeypatch.setattr(_ws_module, "_do_search", fake_search)
    web_search.invoke({"query": "x"})
    assert captured["max"] == 5


def test_web_search_handles_empty_tavily_results(monkeypatch):
    """If Tavily yields no hits, the tool returns a non-empty sentinel result."""
    monkeypatch.setenv(_ENV_KEY, "fake-key")

    def fake_search(query: str, max_results: int):
        return []

    monkeypatch.setattr(_ws_module, "_do_search", fake_search)
    result = web_search.invoke({"query": "anything"})
    assert len(result) == 1
    assert result[0]["status"] == "error"
    assert "No web search results" in result[0]["content"]
