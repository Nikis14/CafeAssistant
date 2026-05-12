"""Tests for the places_search skill."""

from __future__ import annotations

from taste_agent.skills.places_search.places_search import run as places_search_run


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
        assert {"name", "address", "reason", "review_snippet"} <= set(r.keys())
        assert isinstance(r["name"], str)
        assert isinstance(r["address"], str)
        assert isinstance(r["reason"], str)


def test_vegetarian_query_finds_vegetarian_tag():
    results = places_search_run("vegetarian restaurant in Belgrade", "Belgrade", 5)
    names = [r["name"] for r in results]
    assert "Iva New Balkan Cuisine" in names


def test_quiet_query_ranks_quiet_places_higher():
    results = places_search_run("quiet cafe", "Belgrade", 5)
    # Top result should have 'quiet' in its tags (Kafeterija or Iva)
    assert results[0]["name"] in {"Kafeterija", "Iva New Balkan Cuisine"}
