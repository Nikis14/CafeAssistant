"""Tests for the LangGraph-backed place_web_enrichment tool."""

import sys
import types

from taste_agent.tools.place_web_fallback import (
    place_web_enrichment,
    reset_current_model_id,
    reset_graph_cache,
    set_current_model_id,
)

_mod = sys.modules["taste_agent.tools.place_web_fallback"]


def setup_function():
    reset_graph_cache()


def test_place_web_enrichment_returns_normalized_candidates(monkeypatch):
    seen_queries: list[str] = []

    def fake_search(query: str, max_results: int):
        seen_queries.append(query)
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

    result = place_web_enrichment.invoke(
        {"query": "nice restaurant with good coffee", "location": "Belgrade"}
    )
    assert len(seen_queries) == 3
    assert any(q == "best nice restaurant with good coffee in Belgrade" for q in seen_queries)
    assert len(result) == 1
    assert result[0]["name"] == "Cafe Moskva"
    assert result[0]["source"] == "web_enrichment"
    assert result[0]["status"] == "ok"
    assert "https://example.com/moskva" in result[0]["reason"]


def test_place_web_enrichment_accepts_null_optional_urls(monkeypatch):
    def fake_search(query: str, max_results: int):
        return [
            {
                "title": "Cafe Moskva review",
                "url": "https://example.com/moskva",
                "content": "Cafe Moskva is a classic Belgrade spot with coffee and meals.",
                "score": 0.91,
            }
        ]

    def fake_extract(raw_results, *, query: str, location: str, max_results: int):
        return [
            {
                "name": "Cafe Moskva",
                "reason": "Classic Belgrade restaurant-cafe with strong coffee.",
                "review_snippet": "Classic Belgrade spot with coffee and meals.",
                "neighborhood": "Stari Grad",
                "website_url": None,
                "maps_url": None,
                "evidence_url": "https://example.com/moskva",
            }
        ]

    monkeypatch.setattr(_mod, "_do_search", fake_search)
    monkeypatch.setattr(_mod, "_extract_candidates", fake_extract)

    result = place_web_enrichment.invoke({"query": "coffee", "location": "Belgrade"})
    assert len(result) == 1
    assert result[0]["source"] == "web_enrichment"
    assert result[0]["status"] == "ok"
    assert result[0]["website_url"] == ""
    assert result[0]["maps_url"] == ""


def test_place_web_enrichment_propagates_search_error(monkeypatch):
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

    result = place_web_enrichment.invoke({"query": "x", "location": "Belgrade"})
    assert len(result) == 1
    assert result[0]["source"] == "error"
    assert result[0]["status"] == "error"
    assert "TAVILY_API_KEY" in result[0]["reason"]


def test_place_web_enrichment_surfaces_empty_extractor_result(monkeypatch):
    def fake_search(query: str, max_results: int):
        return [
            {
                "title": "Cafe Moskva review",
                "url": "https://example.com/moskva",
                "content": "Cafe Moskva is a classic Belgrade spot with coffee and meals.",
                "score": 0.91,
            }
        ]

    def fake_extract(raw_results, *, query: str, location: str, max_results: int):
        return []

    monkeypatch.setattr(_mod, "_do_search", fake_search)
    monkeypatch.setattr(_mod, "_extract_candidates", fake_extract)

    result = place_web_enrichment.invoke({"query": "coffee", "location": "Belgrade"})
    assert len(result) == 1
    assert result[0]["source"] == "error"
    assert result[0]["status"] == "error"
    assert "extractor returned no candidates" in result[0]["reason"]


def test_place_web_enrichment_surfaces_extractor_parse_failure(monkeypatch):
    def fake_search(query: str, max_results: int):
        return [
            {
                "title": "Cafe Moskva review",
                "url": "https://example.com/moskva",
                "content": "Cafe Moskva is a classic Belgrade spot with coffee and meals.",
                "score": 0.91,
            }
        ]

    def fake_extract(raw_results, *, query: str, location: str, max_results: int):
        raise ValueError("bad json")

    monkeypatch.setattr(_mod, "_do_search", fake_search)
    monkeypatch.setattr(_mod, "_extract_candidates", fake_extract)

    result = place_web_enrichment.invoke({"query": "coffee", "location": "Belgrade"})
    assert len(result) == 1
    assert result[0]["source"] == "error"
    assert result[0]["status"] == "error"
    assert result[0]["reason"] == "Web enrichment extractor returned an invalid payload."


def test_place_web_enrichment_uses_current_selected_model(monkeypatch):
    def fake_search(query: str, max_results: int):
        return [
            {
                "title": "Cafe Moskva review",
                "url": "https://example.com/moskva",
                "content": "Cafe Moskva is a classic Belgrade spot with coffee and meals.",
                "score": 0.91,
            }
        ]

    captured: dict[str, str] = {}

    class _FakeChatLiteLLM:
        def __init__(self, model: str, **kwargs):
            captured["model"] = model

        def invoke(self, _messages):
            return types.SimpleNamespace(
                content='{"candidates":[{"name":"Cafe Moskva","reason":"Classic Belgrade cafe.","review_snippet":"Cafe Moskva is a classic Belgrade spot.","neighborhood":"Stari Grad","website_url":"","maps_url":"","evidence_url":"https://example.com/moskva"}]}'
            )

    monkeypatch.setattr(_mod, "_do_search", fake_search)
    monkeypatch.setitem(
        sys.modules, "langchain_litellm", types.SimpleNamespace(ChatLiteLLM=_FakeChatLiteLLM)
    )

    token = set_current_model_id("openai/gpt-5-mini")
    try:
        result = place_web_enrichment.invoke({"query": "coffee", "location": "Belgrade"})
    finally:
        reset_current_model_id(token)

    assert captured["model"] == "openai/gpt-5-mini"
    assert result[0]["name"] == "Cafe Moskva"


def test_place_web_enrichment_dedupes_raw_results_across_query_variants(monkeypatch):
    def fake_search(query: str, max_results: int):
        return [
            {
                "title": "Cafe Moskva review",
                "url": "https://example.com/moskva",
                "content": "Cafe Moskva is a classic Belgrade spot.",
                "score": 0.91,
            },
            {
                "title": "Cafe Moskva review",
                "url": "https://example.com/moskva",
                "content": "Cafe Moskva is a classic Belgrade spot.",
                "score": 0.91,
            },
        ]

    def fake_extract(raw_results, *, query: str, location: str, max_results: int):
        assert len(raw_results) == 1
        return [
            {
                "name": "Cafe Moskva",
                "reason": "Classic Belgrade restaurant-cafe.",
                "review_snippet": "Cafe Moskva is a classic Belgrade spot.",
                "neighborhood": "Stari Grad",
                "website_url": "",
                "maps_url": "",
                "evidence_url": "https://example.com/moskva",
            }
        ]

    monkeypatch.setattr(_mod, "_do_search", fake_search)
    monkeypatch.setattr(_mod, "_extract_candidates", fake_extract)

    result = place_web_enrichment.invoke({"query": "coffee", "location": "Belgrade"})
    assert len(result) == 1
    assert result[0]["name"] == "Cafe Moskva"
