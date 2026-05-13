"""Tests for EpisodicMemory (Chroma vector store of dining experiences).

Uses the deterministic fake embedding so no model downloads happen. Vector
relevance can't be meaningfully asserted with fake embeddings — these tests
focus on the store's behavioral contract (write, count, search returns the
right schema).
"""

from __future__ import annotations

from taste_agent.memory.episodic import EpisodicMemory
from taste_agent.memory.schemas import EpisodicEvent


def _new_store() -> EpisodicMemory:
    # Unique collection name per call to keep state isolated.
    import uuid

    return EpisodicMemory(collection_name=f"test_{uuid.uuid4().hex[:8]}")


def test_count_starts_at_zero():
    store = _new_store()
    assert store.count() == 0


def test_log_returns_doc_id():
    store = _new_store()
    event = EpisodicEvent(place_name="Iva", notes="loved the gnocchi", rating=5)
    doc_id = store.log(event)
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0


def test_log_increments_count():
    store = _new_store()
    store.log(EpisodicEvent(place_name="Iva", notes="great"))
    store.log(EpisodicEvent(place_name="Koffein", notes="good cappuccino"))
    assert store.count() == 2


def test_log_fills_in_default_date_when_missing():
    store = _new_store()
    store.log(EpisodicEvent(place_name="X", notes="had lunch"))
    # Retrieve via search — date should now be set on the returned event
    results = store.search("lunch", k=1)
    assert len(results) == 1
    assert results[0].date is not None
    assert len(results[0].date) == 10  # YYYY-MM-DD


def test_search_returns_episodic_event_objects():
    store = _new_store()
    store.log(
        EpisodicEvent(
            place_name="Iva",
            notes="loved the gnocchi",
            rating=5,
            cuisine="Italian",
        )
    )
    results = store.search("gnocchi", k=5)
    assert len(results) == 1
    event = results[0]
    assert isinstance(event, EpisodicEvent)
    assert event.place_name == "Iva"
    assert event.notes == "loved the gnocchi"
    assert event.rating == 5
    assert event.cuisine == "Italian"


def test_search_respects_k():
    store = _new_store()
    for i in range(5):
        store.log(EpisodicEvent(place_name=f"Place {i}", notes=f"visit {i}"))
    results = store.search("any", k=2)
    assert len(results) == 2


def test_search_on_empty_store_returns_empty():
    store = _new_store()
    assert store.search("anything") == []


def test_clear_removes_all_events():
    store = _new_store()
    store.log(EpisodicEvent(place_name="X", notes="x"))
    store.log(EpisodicEvent(place_name="Y", notes="y"))
    store.clear()
    assert store.count() == 0


def test_default_singleton_is_stable():
    from taste_agent.memory.episodic import get_default, set_default

    set_default(None)
    a = get_default()
    b = get_default()
    assert a is b
    set_default(None)


# ── list_recent (date-ordered, not similarity-ordered) ───────────────────────


def test_list_recent_returns_events_in_date_descending_order():
    store = _new_store()
    store.log(EpisodicEvent(place_name="Old", notes="old visit", date="2026-01-01"))
    store.log(EpisodicEvent(place_name="New", notes="recent visit", date="2026-05-01"))
    store.log(EpisodicEvent(place_name="Mid", notes="mid visit", date="2026-03-01"))

    events = store.list_recent(k=5)
    assert [e.place_name for e in events] == ["New", "Mid", "Old"]


def test_list_recent_respects_k():
    store = _new_store()
    for i in range(5):
        store.log(
            EpisodicEvent(place_name=f"Place {i}", notes="x", date=f"2026-01-0{i + 1}")
        )
    assert len(store.list_recent(k=3)) == 3


def test_list_recent_empty_store_returns_empty_list():
    store = _new_store()
    assert store.list_recent() == []


def test_list_recent_handles_missing_dates_gracefully():
    """Events without an explicit date have today's date auto-filled by log(),
    so they should still order sanely. Hand-craft a corner case by relying on
    the auto-fill."""
    store = _new_store()
    store.log(EpisodicEvent(place_name="A", notes="a"))
    store.log(EpisodicEvent(place_name="B", notes="b"))
    events = store.list_recent()
    assert len(events) == 2
    assert {e.place_name for e in events} == {"A", "B"}
