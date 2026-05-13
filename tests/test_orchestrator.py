"""Tests for orchestrator internals: text extraction, tool-call counting, cache."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from taste_agent.orchestrator import (
    _build_output_context,
    _count_tool_calls,
    _extract_text,
    _format_output_note,
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


def test_build_agent_caches_same_factory():
    reset_agent_cache()
    factory = lambda _id: FakeAgentModel(response="x")  # noqa: E731
    a1 = build_agent("fake/1", model_factory=factory)
    a2 = build_agent("fake/1", model_factory=factory)
    assert a1 is a2


def test_build_agent_rebuilds_for_different_factory():
    # Two distinct factory objects should yield distinct agents even for the
    # same model id — the cache key includes factory identity.
    reset_agent_cache()
    f1 = lambda _id: FakeAgentModel(response="x")  # noqa: E731
    f2 = lambda _id: FakeAgentModel(response="y")  # noqa: E731
    a1 = build_agent("fake/1", model_factory=f1)
    a2 = build_agent("fake/1", model_factory=f2)
    assert a1 is not a2


def test_reset_agent_cache_clears_entries():
    reset_agent_cache()
    factory = lambda _id: FakeAgentModel(response="x")  # noqa: E731
    a1 = build_agent("fake/1", model_factory=factory)
    reset_agent_cache()
    a2 = build_agent("fake/1", model_factory=factory)
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


def test_format_note_suppresses_factuality_when_ok_flag_true():
    """Advisory concerns without a fail flag should not be loudly surfaced."""
    from taste_agent.guardrails.output import OutputGuardrailResult

    result = OutputGuardrailResult(
        response_text="x",
        factuality_ok=True,
        factuality_concerns=["minor advisory"],
    )
    assert "judge:factuality" not in _format_output_note(result)
