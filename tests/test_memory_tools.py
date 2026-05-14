"""Tests for the memory_read and memory_search tools."""

from __future__ import annotations

from taste_agent.memory import (
    EpisodicEvent,
    get_default_episodic,
    get_default_semantic,
)
from taste_agent.tools.memory_read import memory_read
from taste_agent.tools.memory_search import memory_search


def test_memory_read_returns_empty_dict_initially():
    assert memory_read.invoke({}) == {}


def test_memory_read_returns_current_facts():
    sem = get_default_semantic()
    sem.write("dietary", "vegetarian")
    sem.write("city", "Belgrade")
    result = memory_read.invoke({})
    assert result == {"dietary": "vegetarian", "city": "Belgrade"}


def test_memory_search_returns_sentinel_when_no_events():
    result = memory_search.invoke({"query": "anything"})
    assert len(result) == 1
    assert result[0]["status"] == "no_results"
    assert "No matching episodic memory" in result[0]["notes"]


def test_memory_search_returns_logged_event_dicts():
    epi = get_default_episodic()
    epi.log(EpisodicEvent(place_name="Iva", notes="loved the gnocchi", rating=5))
    epi.log(EpisodicEvent(place_name="Koffein", notes="great cappuccino"))

    result = memory_search.invoke({"query": "gnocchi", "k": 5})
    assert isinstance(result, list)
    place_names = {r["place_name"] for r in result}
    assert "Iva" in place_names


def test_memory_search_respects_k():
    epi = get_default_episodic()
    for i in range(5):
        epi.log(EpisodicEvent(place_name=f"Place {i}", notes=f"visit {i}"))
    result = memory_search.invoke({"query": "any", "k": 2})
    assert len(result) == 2


def test_memory_read_reflects_writes_via_writes_module():
    """Smoke test: writing via memory.writes is visible to memory_read."""
    from taste_agent.memory.writes import write_semantic

    write_semantic("favorite_cuisine", "balkan")
    facts = memory_read.invoke({})
    assert facts.get("favorite_cuisine") == "balkan"
