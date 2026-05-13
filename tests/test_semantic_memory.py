"""Tests for SemanticMemory (SQLite key-value store)."""

from __future__ import annotations

from taste_agent.memory.semantic import SemanticMemory


def test_read_unknown_key_returns_none():
    m = SemanticMemory()
    assert m.read("dietary") is None


def test_write_then_read_roundtrip():
    m = SemanticMemory()
    m.write("dietary", "vegetarian")
    fact = m.read("dietary")
    assert fact is not None
    assert fact.key == "dietary"
    assert fact.value == "vegetarian"
    assert fact.source == "explicit"
    assert fact.confidence == 1.0


def test_write_with_explicit_source_and_confidence():
    m = SemanticMemory()
    m.write("city", "Belgrade", source="inferred", confidence=0.6)
    fact = m.read("city")
    assert fact is not None
    assert fact.source == "inferred"
    assert fact.confidence == 0.6


def test_write_upserts_existing_key():
    m = SemanticMemory()
    m.write("dietary", "vegetarian")
    m.write("dietary", "vegan")
    fact = m.read("dietary")
    assert fact is not None
    assert fact.value == "vegan"


def test_write_preserves_created_at_on_update():
    m = SemanticMemory()
    m.write("city", "Belgrade")
    first = m.read("city")
    assert first is not None
    m.write("city", "Vienna")
    second = m.read("city")
    assert second is not None
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


def test_all_returns_every_fact():
    m = SemanticMemory()
    m.write("dietary", "vegetarian")
    m.write("city", "Belgrade")
    m.write("ambience_pref", "quiet")
    facts = m.all()
    keys = {f.key for f in facts}
    assert keys == {"dietary", "city", "ambience_pref"}


def test_as_dict_returns_flat_mapping():
    m = SemanticMemory()
    m.write("dietary", "vegetarian")
    m.write("city", "Belgrade")
    assert m.as_dict() == {"dietary": "vegetarian", "city": "Belgrade"}


def test_delete_removes_key():
    m = SemanticMemory()
    m.write("dietary", "vegetarian")
    assert m.delete("dietary") is True
    assert m.read("dietary") is None


def test_delete_unknown_key_returns_false():
    m = SemanticMemory()
    assert m.delete("nonexistent") is False


def test_clear_wipes_all():
    m = SemanticMemory()
    m.write("a", "1")
    m.write("b", "2")
    m.clear()
    assert m.all() == []


# ── Conflict detection ──────────────────────────────────────────────────────


def test_detect_conflict_when_explicit_value_differs():
    m = SemanticMemory()
    m.write("dietary", "vegetarian", source="explicit")
    conflict = m.detect_conflict("dietary", "vegan")
    assert conflict is not None
    assert conflict.value == "vegetarian"


def test_no_conflict_when_value_matches():
    m = SemanticMemory()
    m.write("dietary", "vegetarian", source="explicit")
    assert m.detect_conflict("dietary", "vegetarian") is None


def test_no_conflict_when_existing_is_inferred():
    """Inferred facts can be overwritten silently — only explicit user
    statements are protected by the conflict check."""
    m = SemanticMemory()
    m.write("dietary", "vegetarian", source="inferred", confidence=0.5)
    assert m.detect_conflict("dietary", "vegan") is None


def test_no_conflict_for_new_key():
    m = SemanticMemory()
    assert m.detect_conflict("unset_key", "anything") is None


def test_default_singleton_persists_within_process():
    """get_default returns the same instance across calls in a test (until
    set_default(None) resets it)."""
    from taste_agent.memory.semantic import get_default, set_default

    set_default(None)
    a = get_default()
    b = get_default()
    assert a is b
    set_default(None)
