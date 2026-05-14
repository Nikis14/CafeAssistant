"""Tests for memory/derive.py — pattern derivation + threshold trigger."""

from __future__ import annotations

import uuid

from taste_agent.memory import (
    EpisodicEvent,
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
)
from taste_agent.memory.derive import (
    _parse_payload,
    derive_patterns,
    maybe_derive_procedural,
)
from tests.fakes import FakeAgentModel


def _factory(json_response: str):
    def make(_id: str) -> FakeAgentModel:
        return FakeAgentModel(response=json_response)

    return make


def _new_stores():
    return (
        SemanticMemory(),
        EpisodicMemory(collection_name=f"test_{uuid.uuid4().hex[:8]}"),
        ProceduralMemory(),
    )


# ── _parse_payload ───────────────────────────────────────────────────────────


def test_parse_payload_basic():
    raw = '{"patterns": [{"text": "Prefers Italian", "confidence": 0.8, "evidence_count": 4}]}'
    payload = _parse_payload(raw)
    assert len(payload.patterns) == 1
    assert payload.patterns[0].text == "Prefers Italian"


def test_parse_payload_strips_markdown_fence():
    raw = '```json\n{"patterns": []}\n```'
    payload = _parse_payload(raw)
    assert payload.patterns == []


def test_parse_payload_empty_patterns():
    raw = '{"patterns": []}'
    payload = _parse_payload(raw)
    assert payload.patterns == []


def test_parse_payload_invalid_no_json_raises():
    import pytest

    with pytest.raises(ValueError, match="no JSON"):
        _parse_payload("just prose, no json")


# ── derive_patterns (single LLM call) ────────────────────────────────────────


def test_derive_patterns_returns_patterns_from_llm_json():
    sem, epi, _ = _new_stores()
    epi.log(EpisodicEvent(place_name="A", notes="loved it", rating=5))
    epi.log(EpisodicEvent(place_name="B", notes="great", rating=4))

    factory = _factory(
        '{"patterns": [{"text": "Highly rates dinners", "confidence": 0.85, "evidence_count": 2}]}'
    )
    patterns = derive_patterns(semantic=sem, episodic=epi, model_factory=factory)
    assert len(patterns) == 1
    assert patterns[0].text == "Highly rates dinners"
    assert patterns[0].confidence == 0.85
    assert patterns[0].evidence_count == 2
    assert patterns[0].derived_at is not None


def test_derive_patterns_empty_history_returns_empty():
    sem, epi, _ = _new_stores()
    factory = _factory('{"patterns": []}')
    patterns = derive_patterns(semantic=sem, episodic=epi, model_factory=factory)
    assert patterns == []


def test_derive_patterns_parse_failure_returns_none():
    """Codex P1: distinguish failure (None) from success-with-no-patterns
    ([]). Failure must NOT wipe accumulated patterns."""
    sem, epi, _ = _new_stores()
    factory = _factory("not valid json")
    patterns = derive_patterns(semantic=sem, episodic=epi, model_factory=factory)
    assert patterns is None


def test_derive_patterns_schema_drift_returns_none():
    """Coerced types (e.g. confidence as string) -> validation failure -> None."""
    sem, epi, _ = _new_stores()
    factory = _factory(
        '{"patterns": [{"text": "x", "confidence": "high", "evidence_count": 3}]}'
    )
    patterns = derive_patterns(semantic=sem, episodic=epi, model_factory=factory)
    assert patterns is None


def test_derive_patterns_success_with_no_patterns_returns_empty_list():
    """An empty patterns list is a VALID derivation result and should be
    distinct from None (the failure signal)."""
    sem, epi, _ = _new_stores()
    factory = _factory('{"patterns": []}')
    patterns = derive_patterns(semantic=sem, episodic=epi, model_factory=factory)
    assert patterns == []
    assert patterns is not None


# ── maybe_derive_procedural (threshold trigger) ──────────────────────────────


def test_maybe_derive_skips_when_below_threshold():
    sem, epi, proc = _new_stores()
    epi.log(EpisodicEvent(place_name="A", notes="x"))
    epi.log(EpisodicEvent(place_name="B", notes="y"))
    # Only 2 episodes, threshold is 5 → no derivation
    factory = _factory('{"patterns": []}')
    ran = maybe_derive_procedural(
        model_factory=factory,
        episode_threshold=5,
        semantic=sem,
        episodic=epi,
        procedural=proc,
    )
    assert ran is False
    assert proc.count() == 0


def test_maybe_derive_runs_when_threshold_crossed():
    sem, epi, proc = _new_stores()
    for i in range(5):
        epi.log(EpisodicEvent(place_name=f"P{i}", notes=f"visit {i}"))
    factory = _factory(
        '{"patterns": [{"text": "Visits frequently", "confidence": 0.7, "evidence_count": 5}]}'
    )
    ran = maybe_derive_procedural(
        model_factory=factory,
        episode_threshold=5,
        semantic=sem,
        episodic=epi,
        procedural=proc,
    )
    assert ran is True
    assert proc.count() == 1
    assert proc.last_derive_episode_count() == 5


def test_maybe_derive_idempotent_after_recent_run():
    """A second call right after a successful derivation should short-circuit."""
    sem, epi, proc = _new_stores()
    for i in range(5):
        epi.log(EpisodicEvent(place_name=f"P{i}", notes=f"visit {i}"))
    factory = _factory(
        '{"patterns": [{"text": "x", "confidence": 0.5, "evidence_count": 2}]}'
    )
    ran1 = maybe_derive_procedural(
        model_factory=factory,
        episode_threshold=5,
        semantic=sem,
        episodic=epi,
        procedural=proc,
    )
    ran2 = maybe_derive_procedural(
        model_factory=factory,
        episode_threshold=5,
        semantic=sem,
        episodic=epi,
        procedural=proc,
    )
    assert ran1 is True
    assert ran2 is False  # no new episodes since the first derive


def test_maybe_derive_failure_preserves_existing_patterns():
    """Codex P1 regression: when derive_patterns fails (returns None), the
    existing procedural patterns must NOT be wiped and the threshold counter
    must NOT advance. Otherwise a single transient LLM failure permanently
    loses accumulated patterns and blocks retry."""
    from taste_agent.memory.schemas import InferredPattern

    sem, epi, proc = _new_stores()
    # Seed existing patterns from a prior derivation
    proc.replace_all(
        [InferredPattern(text="prior pattern", confidence=0.9, evidence_count=5)]
    )
    proc.set_last_derive_episode_count(3)
    # Add enough new episodes to cross the threshold
    for i in range(5):
        epi.log(EpisodicEvent(place_name=f"P{i}", notes=f"visit {i}"))

    # LLM returns garbage → derive_patterns returns None
    bad_factory = _factory("not valid json at all")
    ran = maybe_derive_procedural(
        model_factory=bad_factory,
        episode_threshold=5,
        semantic=sem,
        episodic=epi,
        procedural=proc,
    )
    assert ran is False
    # Existing pattern survives
    assert proc.count() == 1
    assert proc.all()[0].text == "prior pattern"
    # Threshold counter was NOT advanced → next call will retry
    assert proc.last_derive_episode_count() == 3


def test_maybe_derive_success_with_empty_patterns_advances_and_clears():
    """When derive_patterns succeeds with an empty list (no patterns
    well-supported), that's a VALID result — clear the store and advance
    the threshold."""
    from taste_agent.memory.schemas import InferredPattern

    sem, epi, proc = _new_stores()
    proc.replace_all(
        [InferredPattern(text="stale", confidence=0.7, evidence_count=2)]
    )
    # Last derive at 0; add 5 episodes → delta crosses threshold
    proc.set_last_derive_episode_count(0)
    for i in range(5):
        epi.log(EpisodicEvent(place_name=f"P{i}", notes=f"visit {i}"))

    factory = _factory('{"patterns": []}')
    ran = maybe_derive_procedural(
        model_factory=factory,
        episode_threshold=5,
        semantic=sem,
        episodic=epi,
        procedural=proc,
    )
    assert ran is True
    # Empty list IS a valid result — old patterns cleared
    assert proc.count() == 0
    # Threshold advanced
    assert proc.last_derive_episode_count() == 5


def test_maybe_derive_passes_requested_model_id():
    sem, epi, proc = _new_stores()
    for i in range(5):
        epi.log(EpisodicEvent(place_name=f"P{i}", notes=f"visit {i}"))

    seen: list[str] = []

    def factory(model_id: str):
        seen.append(model_id)
        return FakeAgentModel(response='{"patterns": []}')

    maybe_derive_procedural(
        model_factory=factory,
        episode_threshold=5,
        semantic=sem,
        episodic=epi,
        procedural=proc,
        model_id="mistral/mistral-small-latest",
    )
    assert seen == ["mistral/mistral-small-latest"]
