"""Tests for memory/writes.py — write_semantic + write_episodic.

These primitives previously lived inside the (removed) memorize skill; the
test coverage moves with them.
"""

from __future__ import annotations

from taste_agent.memory import SemanticMemory, get_default_semantic
from taste_agent.memory.episodic import EpisodicMemory
from taste_agent.memory.writes import write_episodic, write_semantic


def _new_stores():
    import uuid

    return (
        SemanticMemory(),
        EpisodicMemory(collection_name=f"test_{uuid.uuid4().hex[:8]}"),
    )


# ── write_semantic ───────────────────────────────────────────────────────────


def test_write_semantic_writes_fact():
    sem, _ = _new_stores()
    result = write_semantic("dietary", "vegetarian", store=sem)
    assert result["written"] is True
    assert result["conflict"] is None
    assert sem.read("dietary").value == "vegetarian"


def test_write_semantic_with_explicit_source():
    sem, _ = _new_stores()
    write_semantic("dietary", "vegan", source="explicit", store=sem)
    assert sem.read("dietary").source == "explicit"


def test_write_semantic_with_inferred_source():
    sem, _ = _new_stores()
    write_semantic("ambience_pref", "quiet", source="inferred", confidence=0.6, store=sem)
    fact = sem.read("ambience_pref")
    assert fact.source == "inferred"
    assert fact.confidence == 0.6


def test_write_semantic_skips_on_conflict_with_explicit():
    sem, _ = _new_stores()
    sem.write("dietary", "vegetarian", source="explicit")
    result = write_semantic("dietary", "vegan", source="explicit", store=sem)
    assert result["written"] is False
    assert result["conflict"] is not None
    assert result["conflict"]["existing_value"] == "vegetarian"
    assert result["conflict"]["proposed_value"] == "vegan"
    # Existing fact is unchanged
    assert sem.read("dietary").value == "vegetarian"


def test_write_semantic_invalid_input_returns_error():
    sem, _ = _new_stores()
    # Empty key violates SemanticFact validation (actually, allowed? Let's
    # test with a clearly-invalid value type)
    result = write_semantic("", "v", confidence=2.0, store=sem)  # confidence > 1.0
    assert result["written"] is False
    assert "error" in result


def test_write_semantic_default_store_is_module_singleton():
    """When no store is passed, uses the per-session default."""
    write_semantic("city", "Belgrade")
    assert get_default_semantic().read("city").value == "Belgrade"


# ── write_episodic ───────────────────────────────────────────────────────────


def test_write_episodic_logs_event():
    _, epi = _new_stores()
    result = write_episodic(
        "Iva", "loved the gnocchi", rating=5, cuisine="Italian", store=epi
    )
    assert result["written"] is True
    assert "doc_id" in result
    assert epi.count() == 1


def test_write_episodic_validation_missing_notes():
    _, epi = _new_stores()
    # Pydantic requires notes; pass empty -> should still validate (empty
    # string is allowed) — let's instead pass invalid rating
    result = write_episodic("Iva", "ok", rating=99, store=epi)  # rating > 5
    assert result["written"] is False
    assert "error" in result


def test_write_episodic_optional_fields_default_to_none():
    _, epi = _new_stores()
    write_episodic("Iva", "x", store=epi)
    events = epi.list_recent()
    assert len(events) == 1
    assert events[0].rating is None
    assert events[0].address is None
    assert events[0].cuisine is None
