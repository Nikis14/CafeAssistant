"""Tests for the LangGraph-backed place_web_fallback tool."""

from __future__ import annotations

import sys

from taste_agent.tools.place_web_fallback import place_web_fallback, reset_graph_cache

_mod = sys.modules["taste_agent.tools.place_web_fallback"]


def setup_function():
    reset_graph_cache()


def test_place_web_fallback_returns_normalized_candidates(monkeypatch):
    def fake_search(query: str, max_results: int):
        assert "nice restaurant with good coffee" in query
        assert max_results == 5
        return [
            {
                "title": "Cafe Moskva review",
                "url": "https://example.com/moskva",
                "content": "Cafe Moskva is a classic Belgrade spot with coffee and meals.",
                "score": 0.91,
            }
        ]

    def fake_extract(raw_results, *, query: str, location: str, max_results: int):
        assert raw_results[0]["url"] == "https://example.com/moskva"
        assert location == "Belgrade"
        return [
            {
                "name": "Cafe Moskva",
                "reason": "Classic Belgrade restaurant-cafe with strong coffee.",
                "review_snippet": "Classic Belgrade spot with coffee and meals.",
                "neighborhood": "Stari Grad",
                "website_url": "https://cafemoskva.rs",
                "maps_url": "https://maps.example/moskva",
                "evidence_url": "https://example.com/moskva",
            }
        ]

    monkeypatch.setattr(_mod, "_do_search", fake_search)
    monkeypatch.setattr(_mod, "_extract_candidates", fake_extract)

    result = place_web_fallback.invoke(
        {"query": "nice restaurant with good coffee", "location": "Belgrade"}
    )
    assert len(result) == 1
    assert result[0]["name"] == "Cafe Moskva"
    assert result[0]["source"] == "web_fallback"
    assert result[0]["status"] == "ok"
    assert "https://example.com/moskva" in result[0]["reason"]


def test_place_web_fallback_propagates_search_error(monkeypatch):
    def fake_search(query: str, max_results: int):
        return [
            {
                "title": "Web search unavailable",
                "url": "",
                "content": "TAVILY_API_KEY is not set. Query: x",
                "score": 0.0,
                "status": "error",
            }
        ]

    monkeypatch.setattr(_mod, "_do_search", fake_search)

    result = place_web_fallback.invoke({"query": "x", "location": "Belgrade"})
    assert len(result) == 1
    assert result[0]["source"] == "error"
    assert result[0]["status"] == "error"
    assert "TAVILY_API_KEY" in result[0]["reason"]
