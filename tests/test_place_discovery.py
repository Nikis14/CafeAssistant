"""Tests for the merged parallel place discovery tool."""

from __future__ import annotations

import sys

from taste_agent.tools.place_discovery import place_discovery, reset_graph_cache

_mod = sys.modules["taste_agent.tools.place_discovery"]


def setup_function():
    reset_graph_cache()


def test_place_discovery_merges_places_and_web(monkeypatch):
    def fake_places(query: str, location: str, max_results: int):
        return [
            {
                "name": "Cafe Moskva",
                "address": "Stari Grad",
                "reason": "Structured place match",
                "review_snippet": None,
                "website_url": "",
                "reservation_url": "",
                "phone": "",
                "maps_url": "",
                "source": "foursquare",
                "status": "ok",
            }
        ]

    class _FallbackTool:
        @staticmethod
        def invoke(payload):
            assert payload["location"] == "Belgrade"
            return [
                {
                    "name": "Cafe Moskva",
                    "address": "Stari Grad",
                    "reason": "Classic coffee-and-dessert stop. Source: https://example.com/moskva",
                    "review_snippet": "Classic coffee-and-dessert stop.",
                    "website_url": "https://cafemoskva.rs",
                    "reservation_url": "",
                    "phone": "",
                    "maps_url": "https://maps.example/moskva",
                    "source": "web_fallback",
                    "status": "ok",
                }
            ]

    monkeypatch.setattr(_mod, "places_search_run", fake_places)
    monkeypatch.setattr(_mod, "place_web_fallback", _FallbackTool())

    result = place_discovery.invoke(
        {"query": "nice restaurant with good coffee", "location": "Belgrade"}
    )

    assert len(result) == 1
    assert result[0]["name"] == "Cafe Moskva"
    assert result[0]["source"] == "places+web"
    assert "Structured place match" in result[0]["reason"]
    assert "Source: https://example.com/moskva" in result[0]["reason"]
    assert result[0]["website_url"] == "https://cafemoskva.rs"


def test_place_discovery_returns_web_when_places_fail(monkeypatch):
    def fake_places(query: str, location: str, max_results: int):
        return [
            {
                "name": "",
                "address": "Belgrade",
                "reason": "Places API unavailable.",
                "review_snippet": None,
                "website_url": "",
                "reservation_url": "",
                "phone": "",
                "maps_url": "",
                "source": "error",
                "status": "error",
            }
        ]

    class _FallbackTool:
        @staticmethod
        def invoke(payload):
            return [
                {
                    "name": "Sonder Roastery",
                    "address": "Vracar",
                    "reason": "Web-sourced place",
                    "review_snippet": "Good coffee.",
                    "website_url": "https://sonder.rs",
                    "reservation_url": "",
                    "phone": "",
                    "maps_url": "",
                    "source": "web_fallback",
                    "status": "ok",
                }
            ]

    monkeypatch.setattr(_mod, "places_search_run", fake_places)
    monkeypatch.setattr(_mod, "place_web_fallback", _FallbackTool())

    result = place_discovery.invoke({"query": "coffee", "location": "Belgrade"})

    assert len(result) == 1
    assert result[0]["name"] == "Sonder Roastery"
    assert result[0]["source"] == "web_fallback"
