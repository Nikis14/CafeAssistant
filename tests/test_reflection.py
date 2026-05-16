"""Tests for memory/reflection.py — tools + ContextVar collector + skip path.

The full ReAct loop (LLM emits tool calls, agent executes them, loops) needs
a tool-calling fake LLM. For Phase-4-equivalent coverage we test the tools
directly via .invoke() with a hand-set collector, plus the run_reflection
skip / model-factory-None paths.
"""

import pytest

from taste_agent.memory import get_default_semantic
from taste_agent.memory.reflection import (
    ReflectionResult,
    _collector,
    memorize_episodic,
    memorize_semantic,
    request_clarification,
    run_reflection,
)

# ── ContextVar collector setup ───────────────────────────────────────────────


@pytest.fixture
def collector():
    """Set up a fresh ReflectionResult in the ContextVar for the test."""
    result = ReflectionResult()
    token = _collector.set(result)
    try:
        yield result
    finally:
        _collector.reset(token)


# ── memorize_semantic tool ───────────────────────────────────────────────────


def test_memorize_semantic_writes_and_collects(collector):
    out = memorize_semantic.invoke(
        {"key": "dietary", "value": "vegetarian", "source": "explicit"}
    )
    assert "written" in out
    assert len(collector.semantic_writes) == 1
    assert collector.semantic_writes[0]["key"] == "dietary"
    assert collector.semantic_writes[0]["source"] == "explicit"
    assert collector.tool_calls == 1
    assert get_default_semantic().read("dietary").value == "vegetarian"


def test_memorize_semantic_conflict_collected_not_written(collector):
    get_default_semantic().write("dietary", "vegetarian", source="explicit")
    out = memorize_semantic.invoke(
        {"key": "dietary", "value": "vegan", "source": "explicit"}
    )
    assert "conflict" in out
    assert len(collector.semantic_conflicts) == 1
    assert collector.semantic_conflicts[0]["existing_value"] == "vegetarian"
    assert collector.semantic_conflicts[0]["proposed_value"] == "vegan"
    assert len(collector.semantic_writes) == 0
    # Existing unchanged
    assert get_default_semantic().read("dietary").value == "vegetarian"


def test_memorize_semantic_inferred_source(collector):
    memorize_semantic.invoke(
        {"key": "ambience", "value": "quiet", "confidence": 0.6, "source": "inferred"}
    )
    assert collector.semantic_writes[0]["source"] == "inferred"
    assert collector.semantic_writes[0]["confidence"] == 0.6


# ── memorize_episodic tool ───────────────────────────────────────────────────


def test_memorize_episodic_writes_and_collects(collector):
    out = memorize_episodic.invoke(
        {
            "place_name": "Iva",
            "notes": "loved the gnocchi",
            "rating": 5,
            "cuisine": "Italian",
        }
    )
    assert "logged episodic" in out
    assert len(collector.episodic_writes) == 1
    assert collector.episodic_writes[0]["place_name"] == "Iva"


def test_memorize_episodic_invalid_rating_reports_error(collector):
    out = memorize_episodic.invoke(
        {"place_name": "X", "notes": "n", "rating": 99}  # rating > 5
    )
    assert "error" in out
    assert len(collector.episodic_writes) == 0


# ── request_clarification tool ───────────────────────────────────────────────


def test_request_clarification_queues_question(collector):
    memorize_semantic.invoke({"key": "k", "value": "v"})  # bump tool_calls
    out = request_clarification.invoke(
        {"question": "Should I remember you prefer X generally?"}
    )
    assert "queued" in out
    assert len(collector.clarifications) == 1
    assert "prefer X" in collector.clarifications[0]
    assert collector.tool_calls == 2  # two tool calls so far


def test_tools_raise_without_collector():
    """Calling a reflection tool outside a run_reflection context must error
    loudly — silent no-ops would be a footgun."""
    # Ensure collector is None (autouse fixture clears it; assert defensively)
    assert _collector.get() is None
    with pytest.raises(RuntimeError, match="not initialized"):
        memorize_semantic.invoke({"key": "x", "value": "y"})


# ── run_reflection (skip + no-factory paths) ─────────────────────────────────


def test_run_reflection_skipped_when_skip_true():
    result = run_reflection("user said x", "agent said y", skip=True)
    assert result.skipped is True
    assert result.semantic_writes == []
    assert result.tool_calls == 0


def test_run_reflection_skipped_when_no_model_factory():
    result = run_reflection("hi", "hello", model_factory=None, skip=False)
    assert result.skipped is True


def test_run_reflection_skipped_via_env_var(monkeypatch):
    monkeypatch.setenv("TASTE_AGENT_SKIP_REFLECTION", "1")

    def factory(_id: str):
        # Should never be called when skip resolves to True
        raise AssertionError("model_factory should not be invoked when skipped")

    result = run_reflection("hi", "hello", model_factory=factory, skip=None)
    assert result.skipped is True


def test_run_reflection_clears_context_var_after_run():
    """The collector ContextVar must be reset after run_reflection so a leaked
    collector doesn't leak into the next reflection."""
    run_reflection("a", "b", skip=True)
    assert _collector.get() is None


# ── End-to-end ReAct loop (scripted tool-calling LLM) ───────────────────────


def test_run_reflection_drives_memorize_semantic_then_finishes(monkeypatch):
    """Scripted LLM emits a memorize_semantic tool call, then a terminal
    text reply. Verify the full ReAct loop executes the tool, the collector
    records the write, and the agent terminates."""
    from tests.fakes import FakeToolCallingChatModel

    monkeypatch.setenv("TASTE_AGENT_SKIP_REFLECTION", "0")

    def factory(_id: str):
        return FakeToolCallingChatModel(
            responses=[
                # Turn 1: emit a memorize_semantic tool call
                [
                    {
                        "name": "memorize_semantic",
                        "args": {
                            "key": "dietary",
                            "value": "vegetarian",
                            "source": "explicit",
                            "confidence": 1.0,
                        },
                        "id": "call_1",
                    }
                ],
                # Turn 2: terminal text after the tool result is returned
                "done — wrote vegetarian to memory",
            ]
        )

    result = run_reflection(
        user_message="I'm vegetarian, by the way.",
        agent_response="Got it.",
        model_factory=factory,
    )
    assert result.skipped is False
    assert result.error is None
    assert len(result.semantic_writes) == 1
    assert result.semantic_writes[0]["key"] == "dietary"
    assert result.tool_calls == 1


def test_run_reflection_request_clarification_collected(monkeypatch):
    """E2E: scripted LLM calls request_clarification with a question; the
    collector picks it up and the orchestrator (via format_agent_response)
    will weave it into the next reply."""
    from tests.fakes import FakeToolCallingChatModel

    monkeypatch.setenv("TASTE_AGENT_SKIP_REFLECTION", "0")

    def factory(_id: str):
        return FakeToolCallingChatModel(
            responses=[
                [
                    {
                        "name": "request_clarification",
                        "args": {
                            "question": "Should I update your dietary preference?"
                        },
                        "id": "call_1",
                    }
                ],
                "queued a question",
            ]
        )

    result = run_reflection(
        user_message="actually I prefer vegan now",
        agent_response="Noted.",
        model_factory=factory,
    )
    assert len(result.clarifications) == 1
    assert "dietary" in result.clarifications[0]
    assert result.tool_calls == 1


def test_run_reflection_passes_requested_model_id(monkeypatch):
    from tests.fakes import FakeToolCallingChatModel

    monkeypatch.setenv("TASTE_AGENT_SKIP_REFLECTION", "0")
    seen: list[str] = []

    def factory(model_id: str):
        seen.append(model_id)
        return FakeToolCallingChatModel(responses=["done"])

    run_reflection(
        user_message="hi",
        agent_response="hello",
        model_factory=factory,
        model_id="mistral/mistral-small-latest",
    )
    assert seen == ["mistral/mistral-small-latest"]
