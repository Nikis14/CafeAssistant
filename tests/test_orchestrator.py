"""Tests for orchestrator internals: text extraction, tool-call counting, cache."""

from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from taste_agent.orchestrator import (
    _build_output_context,
    _chat_model_kwargs,
    _count_tool_calls,
    _extract_known_booking_values,
    _extract_text,
    _format_output_note,
    _get_agent_parts,
    build_agent,
    reset_agent_cache,
)
from tests.fakes import FakeAgentModel

# ── _extract_text ────────────────────────────────────────────────────────────


def test_extract_text_plain_string():
    assert _extract_text(AIMessage(content="hello")) == "hello"


def test_extract_text_keeps_only_text_blocks():
    # Anthropic-style interleaved content: text + tool_use + thinking. Only
    # the text block should leak through to the user-visible response.
    content = [
        {"type": "text", "text": "Here's a recommendation."},
        {"type": "tool_use", "id": "call_1", "name": "geocode", "input": {"location": "BG"}},
        {"type": "thinking", "thinking": "the user wants a quiet cafe"},
    ]
    msg = AIMessage(content=content)
    assert _extract_text(msg) == "Here's a recommendation."


def test_extract_text_multiple_text_blocks_concatenated():
    content = [
        {"type": "text", "text": "Part A. "},
        {"type": "tool_use", "id": "x", "name": "y", "input": {}},
        {"type": "text", "text": "Part B."},
    ]
    assert _extract_text(AIMessage(content=content)) == "Part A. Part B."


def test_extract_text_empty_list_returns_empty():
    assert _extract_text(AIMessage(content=[])) == ""


def test_extract_text_no_text_blocks_returns_empty():
    content = [{"type": "tool_use", "id": "x", "name": "y", "input": {}}]
    assert _extract_text(AIMessage(content=content)) == ""


# ── _count_tool_calls ────────────────────────────────────────────────────────


def test_count_tool_calls_no_tools():
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert _count_tool_calls(msgs) == 0


def test_count_tool_calls_counts_per_message():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "geocode", "args": {"location": "Belgrade"}, "id": "c1"},
            {"name": "places_search", "args": {"query": "cafe"}, "id": "c2"},
        ],
    )
    assert _count_tool_calls([ai]) == 2


def test_count_tool_calls_sums_across_messages():
    a = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
    b = AIMessage(content="", tool_calls=[{"name": "y", "args": {}, "id": "2"}])
    assert _count_tool_calls([a, b]) == 2


# ── build_agent caching ──────────────────────────────────────────────────────


def test_get_agent_parts_caches_same_factory():
    """LLM + tools are cached per (model_id, factory). Graph itself is
    rebuilt per build_agent call so the system_prompt stays fresh."""
    reset_agent_cache()
    factory = lambda _id: FakeAgentModel(response="x")  # noqa: E731
    llm1, tools1 = _get_agent_parts("fake/1", factory)
    llm2, tools2 = _get_agent_parts("fake/1", factory)
    assert llm1 is llm2
    assert tools1 is tools2


def test_get_agent_parts_rebuilds_for_different_factory():
    reset_agent_cache()
    f1 = lambda _id: FakeAgentModel(response="x")  # noqa: E731
    f2 = lambda _id: FakeAgentModel(response="y")  # noqa: E731
    llm1, _ = _get_agent_parts("fake/1", f1)
    llm2, _ = _get_agent_parts("fake/1", f2)
    assert llm1 is not llm2


def test_reset_agent_cache_clears_entries():
    reset_agent_cache()
    factory = lambda _id: FakeAgentModel(response="x")  # noqa: E731
    llm1, _ = _get_agent_parts("fake/1", factory)
    reset_agent_cache()
    llm2, _ = _get_agent_parts("fake/1", factory)
    assert llm1 is not llm2


def test_build_agent_returns_fresh_graph_each_call():
    """The graph is rebuilt per call so a changed system_prompt takes effect
    immediately — even though the LLM + tools are reused."""
    reset_agent_cache()
    factory = lambda _id: FakeAgentModel(response="x")  # noqa: E731
    a1 = build_agent("fake/1", model_factory=factory, system_prompt_text="A")
    a2 = build_agent("fake/1", model_factory=factory, system_prompt_text="B")
    assert a1 is not a2


# ── _build_output_context (Phase 4 — judge grounding) ───────────────────────


def test_build_output_context_includes_facts():
    """Facts must appear so the judge can ground memory-derived claims."""
    msgs = [HumanMessage(content="best dinner spot?")]
    facts = {"dietary": "vegetarian", "city": "Belgrade"}
    summary = _build_output_context(msgs, facts=facts)
    assert "dietary=vegetarian" in summary
    assert "city=Belgrade" in summary
    assert "best dinner spot" in summary


def test_build_output_context_handles_no_facts():
    msgs = [HumanMessage(content="hi")]
    summary = _build_output_context(msgs, facts=None)
    assert "known user facts" not in summary
    assert "user: hi" in summary


def test_build_output_context_includes_tool_messages():
    msgs = [
        HumanMessage(content="cafe in Belgrade"),
        ToolMessage(content="Iva, Koffein, Kafeterija", tool_call_id="t1", name="places_search"),
    ]
    summary = _build_output_context(msgs)
    assert "tool[places_search]" in summary
    assert "Iva" in summary


def test_build_output_context_respects_per_tool_cap():
    big_content = "X" * 3000
    msgs = [ToolMessage(content=big_content, tool_call_id="t1", name="web_search")]
    summary = _build_output_context(msgs, per_tool_chars=2500, max_chars=10000)
    # Should include first 2500 of the tool content but not the full 3000
    assert "X" * 2500 in summary
    assert "X" * 2501 not in summary


def test_build_output_context_respects_overall_cap():
    msgs = [HumanMessage(content="A" * 10000)]
    summary = _build_output_context(msgs, max_chars=200)
    assert len(summary) == 200
    assert summary.endswith("...")


# ── _format_output_note (Phase 4 — surface-prefixed notes) ──────────────────


def test_format_note_prefixes_pii_concerns():
    from taste_agent.guardrails.output import OutputGuardrailResult

    result = OutputGuardrailResult(
        response_text="x", pii_leaked=1, pii_concerns=["email addresses (1)"]
    )
    note = _format_output_note(result)
    assert "[pii]" in note
    assert "email" in note


def test_format_note_prefixes_factuality_concerns():
    from taste_agent.guardrails.output import OutputGuardrailResult

    result = OutputGuardrailResult(
        response_text="x",
        factuality_ok=False,
        factuality_concerns=["fabricated place"],
    )
    note = _format_output_note(result)
    assert "[judge:factuality]" in note


def test_format_note_prefixes_citation_concerns():
    from taste_agent.guardrails.output import OutputGuardrailResult

    result = OutputGuardrailResult(
        response_text="x",
        citation_ok=False,
        citation_concerns=["unsupported hours"],
    )
    note = _format_output_note(result)
    assert "[judge:citation]" in note


def test_format_note_empty_when_no_concerns():
    from taste_agent.guardrails.output import OutputGuardrailResult

    assert _format_output_note(OutputGuardrailResult(response_text="ok")) == ""


def test_chat_model_kwargs_omit_temperature_for_gpt5():
    assert _chat_model_kwargs("openai/gpt-5") == {}
    assert _chat_model_kwargs("openai/gpt-5-mini") == {}
    assert _chat_model_kwargs("mistral/mistral-small-latest") == {"temperature": 0.2}


def test_extract_known_booking_values_accumulates_prior_user_details():
    history = [
        HumanMessage(content="Please book it for tomorrow at 12 o'clock."),
        HumanMessage(content="There will be two people."),
        HumanMessage(content="The name is Nick."),
    ]
    values = _extract_known_booking_values(
        history,
        now=datetime(2026, 5, 14, 14, 0),
    )
    assert values["date"] == "2026-05-15"
    assert values["time"] == "12:00"
    assert values["party_size"] == "2"
    assert values["contact_name"] == "Nick"


def test_extract_known_booking_values_keeps_latest_override():
    history = [
        HumanMessage(content="Book for two people tomorrow."),
        HumanMessage(content="Actually make it three people."),
    ]
    values = _extract_known_booking_values(
        history,
        now=datetime(2026, 5, 14, 14, 0),
    )
    assert values["party_size"] == "3"


def test_format_note_suppresses_factuality_when_ok_flag_true():
    """Advisory concerns without a fail flag should not be loudly surfaced."""
    from taste_agent.guardrails.output import OutputGuardrailResult

    result = OutputGuardrailResult(
        response_text="x",
        factuality_ok=True,
        factuality_concerns=["minor advisory"],
    )
    assert "judge:factuality" not in _format_output_note(result)


# ── format_agent_response_node — PII in clarifications (Codex P2) ───────────


def _make_format_state(clarifications=None):
    """Helper: build a minimal but real OrchestratorState for testing
    format_agent_response_node. Uses real GuardrailResult / OutputGuardrailResult
    so the schema stays in sync with the production types."""
    from langchain_core.messages import HumanMessage

    from taste_agent.guardrails.input import GuardrailResult
    from taste_agent.guardrails.output import OutputGuardrailResult
    from taste_agent.memory.reflection import ReflectionResult

    return {
        "guard_result": GuardrailResult(
            cleaned_text="hi", pii_redactions=0, injection_flagged=False, out_of_scope=False
        ),
        "out_guard": OutputGuardrailResult(response_text="ok"),
        "agent_messages": [HumanMessage(content="hi")],
        "facts": {},
        "patterns_text": "",
        "response_text": "ok",
        "pending_before_id": None,
        "reflection_result": ReflectionResult(clarifications=clarifications or []),
    }


def test_clarifications_are_pii_redacted_before_appending():
    """Reflection sub-agent's LLM output bypasses output_guardrail_node;
    format_agent_response_node must run a deterministic PII pass on the
    appended clarification block."""
    from taste_agent.orchestrator import format_agent_response_node

    state = _make_format_state(
        clarifications=["Should I email you at chef@iva.rs about this?"]
    )
    out = format_agent_response_node(state)
    assert "chef@iva.rs" not in out["response_text"]
    assert "[REDACTED-EMAIL]" in out["response_text"]
    assert out["debug"]["clarification_pii_redactions"] == 1


def test_clarifications_capped_at_two_per_turn():
    """If reflection queues many questions, only the first 2 are appended;
    the rest get dropped and counted in debug."""
    from taste_agent.orchestrator import format_agent_response_node

    state = _make_format_state(
        clarifications=[
            "Q1: should I remember A?",
            "Q2: should I remember B?",
            "Q3: should I remember C?",
            "Q4: should I remember D?",
        ]
    )
    out = format_agent_response_node(state)
    assert "Q1" in out["response_text"]
    assert "Q2" in out["response_text"]
    assert "Q3" not in out["response_text"]
    assert "Q4" not in out["response_text"]
    assert out["debug"]["clarifications_dropped"] == 2


def test_output_guardrail_node_does_not_append_user_visible_note():
    from taste_agent.guardrails.output import OutputGuardrailResult
    from taste_agent.orchestrator import output_guardrail_node

    state = {
        "response_text": "Try Iva. Call +381 64 123 4567.",
        "agent_messages": [],
        "facts": {},
        "skip_output_judge": True,
        "model_factory": None,
    }
    out = output_guardrail_node(state)
    assert isinstance(out["out_guard"], OutputGuardrailResult)
    assert "_Output guardrail:_" not in out["response_text"]


def test_memory_gate_skips_reflection_for_transactional_reply(monkeypatch):
    from taste_agent.memory.gating import MemoryGateDecision
    from taste_agent.orchestrator import reflection_node

    called = False

    def fail_run_reflection(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("run_reflection should not be called")

    monkeypatch.setattr("taste_agent.orchestrator.run_reflection", fail_run_reflection)

    state = {
        "memory_gate": MemoryGateDecision(
            should_reflect=False,
            allow_clarification=False,
            semantic_candidate=False,
            episodic_candidate=False,
            transactional_only=True,
            task_clarification_only=True,
            reason="transactional_only_turn",
            window_messages=[],
        ),
        "out_guard": type("Out", (), {"response_text": "ok"})(),
        "cleaned_text": "book it for tomorrow",
        "model_id": "mistral/mistral-small-latest",
        "skip_reflection": False,
    }

    out = reflection_node(state)

    assert called is False
    assert out["reflection_result"].skipped is True


def test_reflection_node_passes_window_and_gate_controls(monkeypatch):
    from taste_agent.memory.gating import MemoryGateDecision
    from taste_agent.memory.reflection import ReflectionResult
    from taste_agent.orchestrator import reflection_node

    seen: dict[str, object] = {}

    def fake_run_reflection(**kwargs):
        seen.update(kwargs)
        return ReflectionResult()

    monkeypatch.setattr("taste_agent.orchestrator.run_reflection", fake_run_reflection)

    decision = MemoryGateDecision(
        should_reflect=True,
        allow_clarification=False,
        semantic_candidate=True,
        episodic_candidate=False,
        transactional_only=False,
        task_clarification_only=False,
        reason="reply_to_memory_clarification",
        window_messages=[],
    )
    state = {
        "memory_gate": decision,
        "memory_window_text": "User: I usually prefer quiet places.\nAssistant: Noted.",
        "out_guard": type("Out", (), {"response_text": "Noted."})(),
        "cleaned_text": "yes please",
        "model_id": "mistral/mistral-small-latest",
        "skip_reflection": False,
        "model_factory": lambda _id: FakeAgentModel(response="unused"),
    }

    reflection_node(state)

    assert seen["conversation_window"] == state["memory_window_text"]
    assert seen["gate_reason"] == "reply_to_memory_clarification"
    assert seen["allow_clarification"] is False
