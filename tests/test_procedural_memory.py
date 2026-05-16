"""Tests for ProceduralMemory (SQLite-backed inferred patterns)."""

from datetime import datetime, timezone

from taste_agent.memory import InferredPattern
from taste_agent.memory.procedural import ProceduralMemory


def test_count_starts_at_zero():
    p = ProceduralMemory()
    assert p.count() == 0


def test_last_derive_count_defaults_to_zero():
    p = ProceduralMemory()
    assert p.last_derive_episode_count() == 0


def test_replace_all_writes_patterns():
    p = ProceduralMemory()
    patterns = [
        InferredPattern(text="Prefers small intimate places", confidence=0.8, evidence_count=4),
        InferredPattern(text="Books Italian on weekdays", confidence=0.7, evidence_count=3),
    ]
    p.replace_all(patterns)
    assert p.count() == 2


def test_replace_all_is_wholesale():
    p = ProceduralMemory()
    p.replace_all(
        [InferredPattern(text="old pattern", confidence=0.5, evidence_count=2)]
    )
    p.replace_all(
        [InferredPattern(text="new pattern", confidence=0.9, evidence_count=5)]
    )
    all_patterns = p.all()
    assert len(all_patterns) == 1
    assert all_patterns[0].text == "new pattern"


def test_all_returns_ordered_by_confidence_then_evidence():
    p = ProceduralMemory()
    p.replace_all(
        [
            InferredPattern(text="low", confidence=0.5, evidence_count=10),
            InferredPattern(text="high", confidence=0.9, evidence_count=2),
            InferredPattern(text="mid-evidence-tie", confidence=0.5, evidence_count=20),
        ]
    )
    patterns = p.all()
    assert patterns[0].text == "high"  # confidence wins
    # The two confidence=0.5 entries are ordered by evidence count desc
    assert patterns[1].text == "mid-evidence-tie"
    assert patterns[2].text == "low"


def test_as_text_renders_for_prompt_injection():
    p = ProceduralMemory()
    p.replace_all(
        [InferredPattern(text="Prefers Italian", confidence=0.85, evidence_count=4)]
    )
    text = p.as_text()
    assert "Prefers Italian" in text
    assert "confidence 0.85" in text
    assert "evidence 4" in text


def test_as_text_empty_when_no_patterns():
    assert ProceduralMemory().as_text() == ""


def test_set_and_get_last_derive_episode_count():
    p = ProceduralMemory()
    p.set_last_derive_episode_count(7)
    assert p.last_derive_episode_count() == 7
    # Idempotent upsert
    p.set_last_derive_episode_count(15)
    assert p.last_derive_episode_count() == 15


def test_clear_wipes_patterns_and_meta():
    p = ProceduralMemory()
    p.replace_all([InferredPattern(text="x", confidence=0.5, evidence_count=2)])
    p.set_last_derive_episode_count(10)
    p.clear()
    assert p.count() == 0
    assert p.last_derive_episode_count() == 0


def test_derived_at_is_persisted():
    p = ProceduralMemory()
    when = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    p.replace_all(
        [InferredPattern(text="x", confidence=0.5, evidence_count=2, derived_at=when)]
    )
    assert p.all()[0].derived_at == when


def test_default_singleton_per_session_is_stable():
    from taste_agent.memory.procedural import get_default, set_default

    set_default(None)
    a = get_default()
    b = get_default()
    assert a is b
    set_default(None)
