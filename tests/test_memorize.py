"""Tests for the memorize skill — conflict detection + write to both stores."""

from __future__ import annotations

from taste_agent.memory.episodic import EpisodicMemory
from taste_agent.memory.semantic import SemanticMemory
from taste_agent.skills.memorize.memorize import _run_impl, run


def _new_stores():
    import uuid

    return (
        SemanticMemory(),
        EpisodicMemory(collection_name=f"test_{uuid.uuid4().hex[:8]}"),
    )


# ── Semantic-only paths ──────────────────────────────────────────────────────


def test_writes_single_semantic_fact():
    sem, epi = _new_stores()
    result = _run_impl(
        semantic_facts=[{"key": "dietary", "value": "vegetarian"}],
        episodic_events=None,
        semantic_store=sem,
        episodic_store=epi,
    )
    assert result["total_written"] == 1
    assert result["semantic_written"] == [{"key": "dietary", "value": "vegetarian"}]
    assert sem.read("dietary").value == "vegetarian"


def test_writes_multiple_semantic_facts():
    sem, epi = _new_stores()
    facts = [
        {"key": "dietary", "value": "vegetarian"},
        {"key": "city", "value": "Belgrade"},
        {"key": "ambience_pref", "value": "quiet"},
    ]
    result = _run_impl(
        semantic_facts=facts, episodic_events=None, semantic_store=sem, episodic_store=epi
    )
    assert result["total_written"] == 3
    assert sem.as_dict() == {
        "dietary": "vegetarian",
        "city": "Belgrade",
        "ambience_pref": "quiet",
    }


def test_semantic_conflict_skips_write_and_reports():
    sem, epi = _new_stores()
    sem.write("dietary", "vegetarian", source="explicit")
    result = _run_impl(
        semantic_facts=[{"key": "dietary", "value": "vegan"}],
        episodic_events=None,
        semantic_store=sem,
        episodic_store=epi,
    )
    assert result["semantic_written"] == []
    assert len(result["semantic_conflicts"]) == 1
    conflict = result["semantic_conflicts"][0]
    assert conflict["key"] == "dietary"
    assert conflict["existing_value"] == "vegetarian"
    assert conflict["proposed_value"] == "vegan"
    # Existing fact is unchanged
    assert sem.read("dietary").value == "vegetarian"


def test_invalid_semantic_fact_reported_in_conflicts():
    sem, epi = _new_stores()
    result = _run_impl(
        semantic_facts=[{"value": "missing-key"}],  # no 'key'
        episodic_events=None,
        semantic_store=sem,
        episodic_store=epi,
    )
    assert result["semantic_written"] == []
    assert len(result["semantic_conflicts"]) == 1
    assert "error" in result["semantic_conflicts"][0]


# ── Episodic-only paths ──────────────────────────────────────────────────────


def test_writes_single_episodic_event():
    sem, epi = _new_stores()
    result = _run_impl(
        semantic_facts=None,
        episodic_events=[
            {
                "place_name": "Iva",
                "notes": "loved the gnocchi",
                "rating": 5,
                "cuisine": "Italian",
            }
        ],
        semantic_store=sem,
        episodic_store=epi,
    )
    assert result["total_written"] == 1
    assert len(result["episodic_written"]) == 1
    assert result["episodic_written"][0]["place_name"] == "Iva"
    assert epi.count() == 1


def test_invalid_episodic_event_skipped_and_reported():
    sem, epi = _new_stores()
    result = _run_impl(
        semantic_facts=None,
        episodic_events=[{"place_name": "X"}],  # missing required 'notes'
        semantic_store=sem,
        episodic_store=epi,
    )
    assert result["episodic_written"] == []
    assert len(result["episodic_skipped"]) == 1
    assert result["episodic_skipped"][0]["place_name"] == "X"


# ── Mixed + edge cases ───────────────────────────────────────────────────────


def test_both_kinds_written_in_one_call():
    sem, epi = _new_stores()
    result = _run_impl(
        semantic_facts=[{"key": "dietary", "value": "vegetarian"}],
        episodic_events=[{"place_name": "Iva", "notes": "great"}],
        semantic_store=sem,
        episodic_store=epi,
    )
    assert result["total_written"] == 2
    assert sem.read("dietary") is not None
    assert epi.count() == 1


def test_empty_input_returns_zero_writes_with_warning():
    sem, epi = _new_stores()
    result = _run_impl(
        semantic_facts=None,
        episodic_events=None,
        semantic_store=sem,
        episodic_store=epi,
    )
    assert result["total_written"] == 0
    assert result["semantic_written"] == []
    assert result["episodic_written"] == []
    # Phase-3 fix: surface no-op as an explicit warning rather than silent success
    assert "warning" in result
    assert "no input" in result["warning"]


def test_both_empty_lists_treated_as_no_input():
    sem, epi = _new_stores()
    result = _run_impl(
        semantic_facts=[],
        episodic_events=[],
        semantic_store=sem,
        episodic_store=epi,
    )
    assert result["total_written"] == 0
    assert "warning" in result


def test_public_run_uses_module_defaults():
    """run() without injected stores should write to the process-default stores."""
    from taste_agent.memory.semantic import get_default

    result = run(semantic_facts=[{"key": "dietary", "value": "vegetarian"}])
    assert result["total_written"] == 1
    assert get_default().read("dietary").value == "vegetarian"
