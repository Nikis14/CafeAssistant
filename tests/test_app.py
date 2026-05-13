"""Tests for the Gradio adapter functions in app.py.

The app module imports gradio + langchain at module load. We import lazily so
that environments without gradio installed can still run the rest of the suite.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gradio")

from langchain_core.messages import AIMessage, HumanMessage

from app import _gradio_history_to_messages


def test_history_messages_format_dicts():
    history = [
        {"role": "user", "content": "Where is the best cappuccino?"},
        {"role": "assistant", "content": "Try Koffein."},
    ]
    msgs = _gradio_history_to_messages(history)
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "Where is the best cappuccino?"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "Try Koffein."


def test_history_legacy_tuple_format():
    history = [("Hi there", "Hello!")]
    msgs = _gradio_history_to_messages(history)
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "Hi there"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "Hello!"


def test_history_skips_empty_content():
    history = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "I'm listening"},
    ]
    msgs = _gradio_history_to_messages(history)
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)


def test_history_empty_list_returns_empty():
    assert _gradio_history_to_messages([]) == []


def test_history_ignores_unknown_role():
    history = [{"role": "system", "content": "ignored"}]
    assert _gradio_history_to_messages(history) == []


# ── Phase 3 fix: per-session memory scoping in chat_fn / _refresh_panels ────


class _FakeRequest:
    """Minimal stand-in for gr.Request that exposes session_hash."""

    def __init__(self, session_hash: str) -> None:
        self.session_hash = session_hash


def test_chat_fn_sets_session_id_from_request_session_hash(monkeypatch):
    """The session id seen by run_turn must match the Gradio request."""
    from taste_agent.memory import current_session_id

    captured: list[str] = []

    def fake_run_turn(message, history, model_id, **kwargs):
        captured.append(current_session_id())
        return "ok", {"n_facts_in_prompt": 0}

    monkeypatch.setattr("app.run_turn", fake_run_turn)

    from app import chat_fn

    chat_fn("hello", [], "Claude Sonnet 4.6", _FakeRequest("browser-abc"))
    assert captured == ["browser-abc"]


def test_chat_fn_restores_session_id_after_call(monkeypatch):
    """After the handler returns, the ContextVar must be back to default."""
    from taste_agent.memory import DEFAULT_SESSION_ID, current_session_id

    monkeypatch.setattr(
        "app.run_turn",
        lambda *_args, **_kwargs: ("ok", {"n_facts_in_prompt": 0}),
    )

    from app import chat_fn

    chat_fn("hi", [], "Claude Sonnet 4.6", _FakeRequest("xyz"))
    assert current_session_id() == DEFAULT_SESSION_ID


def test_refresh_panels_reads_from_session_memory(monkeypatch):
    """Panels surfaced to a session must show that session's facts."""
    from taste_agent.memory import (
        get_default_semantic,
        reset_session_id,
        set_session_id,
    )

    # Seed two sessions with different facts
    token_a = set_session_id("user-a")
    try:
        get_default_semantic().write("dietary", "vegetarian")
    finally:
        reset_session_id(token_a)

    token_b = set_session_id("user-b")
    try:
        get_default_semantic().write("dietary", "carnivore")
    finally:
        reset_session_id(token_b)

    from app import _refresh_panels

    semantic_a, _, _ = _refresh_panels(_FakeRequest("user-a"))
    semantic_b, _, _ = _refresh_panels(_FakeRequest("user-b"))
    assert semantic_a == {"dietary": "vegetarian"}
    assert semantic_b == {"dietary": "carnivore"}


def test_chat_fn_isolates_memory_writes_between_sessions(monkeypatch):
    """A write in one session must NOT be visible from another session."""
    from taste_agent.memory import get_default_semantic

    monkeypatch.setattr(
        "app.run_turn",
        lambda *_args, **_kwargs: ("ok", {"n_facts_in_prompt": 0}),
    )

    from app import chat_fn

    chat_fn("hi", [], "Claude Sonnet 4.6", _FakeRequest("alice"))
    # Simulate Alice memorizing something (skip the LLM path)
    from taste_agent.memory import reset_session_id, set_session_id

    token = set_session_id("alice")
    try:
        get_default_semantic().write("dietary", "vegan")
    finally:
        reset_session_id(token)

    # Bob's session — should NOT see Alice's dietary fact
    token = set_session_id("bob")
    try:
        bob_facts = get_default_semantic().as_dict()
    finally:
        reset_session_id(token)
    assert "dietary" not in bob_facts


def test_refresh_panels_falls_back_to_default_when_no_request():
    """Calling _refresh_panels with no request (e.g., initial component load)
    must not crash; it should read the default session."""
    from app import _refresh_panels

    semantic, episodic, facts_md = _refresh_panels(None)
    assert isinstance(semantic, dict)
    assert isinstance(episodic, list)
    assert "Facts in prompt" in facts_md
