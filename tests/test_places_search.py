"""Tests for the places_search skill — mock + Foursquare backend."""

from __future__ import annotations

import json
import sys
from io import BytesIO

from taste_agent.skills.places_search.places_search import (
    _FOURSQUARE_KEY_ENV,
    run as places_search_run,
)

# Same shadowing trick as test_web_search — the package __init__ doesn't
# re-export this one, but use sys.modules for symmetry.
_ps_module = sys.modules["taste_agent.skills.places_search.places_search"]


def test_specific_query_returns_matching_places():
    results = places_search_run("Best cappuccino in Belgrade", "Belgrade", 5)
    assert len(results) > 0
    # Koffein is tagged 'cappuccino' so it should rank
    names = [r["name"] for r in results]
    assert "Koffein" in names


def test_generic_query_returns_some_results():
    # "where" triggers the generic fallback even with zero tag overlap
    results = places_search_run("where can I eat", "Belgrade", 5)
    assert len(results) > 0


def test_max_results_respected():
    results = places_search_run("cafe", "Belgrade", 2)
    assert len(results) <= 2


def test_result_shape_matches_contract():
    results = places_search_run("quiet cafe with wifi", "Belgrade", 3)
    for r in results:
        assert {
            "name",
            "address",
            "reason",
            "review_snippet",
            "website_url",
            "reservation_url",
            "phone",
            "maps_url",
            "source",
            "status",
        } <= set(r.keys())
        assert isinstance(r["name"], str)
        assert isinstance(r["address"], str)
        assert isinstance(r["reason"], str)
        assert isinstance(r["website_url"], str)
        assert isinstance(r["reservation_url"], str)
        assert isinstance(r["phone"], str)
        assert isinstance(r["maps_url"], str)
        assert r["source"] in {"mock", "foursquare", "error"}
        assert r["status"] in {"ok", "error"}


def test_vegetarian_query_finds_vegetarian_tag():
    results = places_search_run("vegetarian restaurant in Belgrade", "Belgrade", 5)
    names = [r["name"] for r in results]
    assert "Iva New Balkan Cuisine" in names


def test_quiet_query_ranks_quiet_places_higher():
    results = places_search_run("quiet cafe", "Belgrade", 5)
    # Top result should have 'quiet' in its tags (Kafeterija or Iva)
    assert results[0]["name"] in {"Kafeterija", "Iva New Balkan Cuisine"}


# ── Foursquare backend ───────────────────────────────────────────────────────


class _FakeHTTPResponse:
    """Stand-in for urllib's response object — just enough for ``json.load``."""

    def __init__(self, payload: dict):
        self._body = BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args, **kwargs):
        return self._body.read(*args, **kwargs)


def test_foursquare_path_is_used_when_key_is_set(monkeypatch):
    monkeypatch.setenv(_FOURSQUARE_KEY_ENV, "fake-key")

    captured = {}

    def fake_urlopen(request, timeout=8):
        captured["url"] = request.full_url
        captured["auth"] = request.headers.get("Authorization")
        return _FakeHTTPResponse(
            {
                "results": [
                    {
                        "name": "Test Cafe",
                        "location": {
                            "address": "Knez Mihailova 1",
                            "locality": "Belgrade",
                            "country": "RS",
                        },
                        "website": "https://test.example",
                        "tel": "+38111111111",
                        "link": "https://maps.example/test-cafe",
                        "categories": [{"name": "Coffee Shop"}],
                    }
                ]
            }
        )

    monkeypatch.setattr(_ps_module.urllib.request, "urlopen", fake_urlopen)

    results = places_search_run("cappuccino", "Belgrade", 5)
    assert "api.foursquare.com" in captured["url"]
    assert captured["auth"] == "fake-key"
    assert len(results) == 1
    assert results[0]["name"] == "Test Cafe"
    assert "Knez Mihailova 1" in results[0]["address"]
    assert "Coffee Shop" in results[0]["reason"]
    assert results[0]["website_url"] == "https://test.example"
    assert results[0]["phone"] == "+38111111111"
    assert results[0]["maps_url"] == "https://maps.example/test-cafe"
    assert results[0]["source"] == "foursquare"
    assert results[0]["status"] == "ok"


def test_foursquare_url_error_returns_sentinel_not_mock(monkeypatch):
    """When the API key is set, a Foursquare failure must NOT silently fall
    back to Belgrade mock data — the mock is Belgrade-only and would surface
    fabricated results for non-Belgrade queries."""
    import urllib.error

    monkeypatch.setenv(_FOURSQUARE_KEY_ENV, "fake-key")

    def boom(request, timeout=8):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(_ps_module.urllib.request, "urlopen", boom)

    results = places_search_run("cappuccino", "Istanbul", 5)
    assert len(results) == 1
    assert results[0]["address"] == "Istanbul"
    assert "upstream Places API failed" in results[0]["reason"]
    assert results[0]["source"] == "error"
    assert results[0]["status"] == "error"


def test_foursquare_http_error_returns_sentinel_not_mock(monkeypatch):
    """HTTPError (e.g. 500 from Foursquare) must also fall through to a
    sentinel result, not to Belgrade mock data."""
    import urllib.error

    monkeypatch.setenv(_FOURSQUARE_KEY_ENV, "fake-key")

    def http_500(request, timeout=8):
        raise urllib.error.HTTPError(
            url=request.full_url, code=500, msg="internal", hdrs=None, fp=None
        )

    monkeypatch.setattr(_ps_module.urllib.request, "urlopen", http_500)

    results = places_search_run("ramen", "Tokyo", 5)
    assert len(results) == 1
    assert results[0]["address"] == "Tokyo"
    assert "upstream Places API failed" in results[0]["reason"]
    assert results[0]["source"] == "error"
    assert results[0]["status"] == "error"


def test_no_api_key_uses_mock(monkeypatch):
    monkeypatch.delenv(_FOURSQUARE_KEY_ENV, raising=False)
    results = places_search_run("cappuccino", "Belgrade", 5)
    names = {r["name"] for r in results}
    assert "Koffein" in names  # mock fixture
    assert all(r["source"] == "mock" for r in results)


def test_no_api_key_without_runtime_mocks_returns_sentinel(monkeypatch):
    monkeypatch.delenv(_FOURSQUARE_KEY_ENV, raising=False)
    monkeypatch.setenv("TASTE_AGENT_ALLOW_RUNTIME_MOCKS", "0")
    monkeypatch.setattr(_ps_module, "ALLOW_RUNTIME_MOCKS", False)
    results = places_search_run("cappuccino", "Belgrade", 5)
    assert len(results) == 1
    assert results[0]["source"] == "error"
    assert "no live Places API configured" in results[0]["reason"]
