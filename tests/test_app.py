"""Tests for the Gradio adapter functions in app.py.

The app module imports gradio + langchain at module load. We import lazily so
that environments without gradio installed can still run the rest of the suite.
"""

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


def test_history_extracts_text_from_list_content_blocks():
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please book the first one."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I need a date and time."},
            ],
        },
    ]
    msgs = _gradio_history_to_messages(history)
    assert len(msgs) == 2
    assert msgs[0].content == "Please book the first one."
    assert msgs[1].content == "I need a date and time."


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

    semantic_a, _, _, _ = _refresh_panels(_FakeRequest("user-a"))
    semantic_b, _, _, _ = _refresh_panels(_FakeRequest("user-b"))
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

    semantic, episodic, procedural, facts_md = _refresh_panels(None)
    assert isinstance(semantic, dict)
    assert isinstance(episodic, list)
    assert isinstance(procedural, list)
    assert "Facts:" in facts_md
    assert "Patterns:" in facts_md


# ── Multi-conversation per tab ──────────────────────────────────────────────


def test_title_for_empty_history_returns_new_chat():
    from app import _title_for

    assert _title_for([]) == "New chat"


def test_title_for_uses_first_user_message():
    from app import _title_for

    history = [
        {"role": "user", "content": "Best cappuccino in Belgrade?"},
        {"role": "assistant", "content": "Try Koffein"},
    ]
    assert _title_for(history) == "Best cappuccino in Belgrade?"


def test_title_for_truncates_long_messages():
    from app import _TITLE_MAXLEN, _title_for

    long_msg = "a" * (_TITLE_MAXLEN + 50)
    title = _title_for([{"role": "user", "content": long_msg}])
    assert title.endswith("...")
    assert len(title) == _TITLE_MAXLEN + 3


def test_title_for_skips_non_user_messages():
    from app import _title_for

    history = [
        {"role": "assistant", "content": "Hi there"},
    ]
    assert _title_for(history) == "New chat"


def test_title_for_handles_list_content_blocks():
    from app import _title_for

    history = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Please book the first one."}],
        },
        {"role": "assistant", "content": "OK"},
    ]
    assert _title_for(history) == "Please book the first one."


def test_conv_choices_returns_reverse_insertion_order():
    """Most recent (last-inserted) conversation should appear first."""
    from app import _conv_choices

    convs = {
        "aaaa1111": [{"role": "user", "content": "first chat"}],
        "bbbb2222": [{"role": "user", "content": "second chat"}],
    }
    choices = _conv_choices(convs)
    # Newest first → bbbb2222 before aaaa1111
    assert choices[0][1] == "bbbb2222"
    assert choices[1][1] == "aaaa1111"


def test_create_new_conversation_returns_fresh_id_and_empty_chatbot():
    from app import create_new_conversation

    convs, active, chatbot, _radio = create_new_conversation({})
    assert active in convs
    assert convs[active] == []
    assert chatbot == []


def test_create_new_conversation_preserves_existing():
    from app import create_new_conversation

    existing = {"old1": [{"role": "user", "content": "hi"}]}
    convs, active, _chatbot, _radio = create_new_conversation(existing)
    assert "old1" in convs
    assert convs["old1"] == [{"role": "user", "content": "hi"}]
    assert active != "old1"


def test_load_conversation_returns_history():
    from app import load_conversation

    convs = {"abc": [{"role": "user", "content": "hi"}]}
    history, active = load_conversation("abc", convs)
    assert history == [{"role": "user", "content": "hi"}]
    assert active == "abc"


def test_load_conversation_unknown_id_returns_empty():
    from app import load_conversation

    history, active = load_conversation("ghost", {"abc": []})
    assert history == []
    assert active is None


def test_send_message_auto_creates_conversation_on_first_send(monkeypatch):
    """When the user sends a message with no active conversation, a new one
    is auto-created (so a fresh tab works without clicking 'New chat')."""
    monkeypatch.setattr("app.chat_fn", lambda *_a, **_kw: "the reply")

    from app import send_message

    chatbot, convs, active, _radio, cleared_input = send_message(
        "hello",
        chat_history=[],
        conversations={},
        active_id=None,
        model_label="Claude Sonnet 4.6",
        request=_FakeRequest("user-x"),
    )
    assert active is not None
    assert active in convs
    assert chatbot[0] == {"role": "user", "content": "hello"}
    assert chatbot[1] == {"role": "assistant", "content": "the reply"}
    assert cleared_input == ""


def test_send_message_appends_to_active_conversation(monkeypatch):
    monkeypatch.setattr("app.chat_fn", lambda *_a, **_kw: "second reply")

    from app import send_message

    existing_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
    ]
    chatbot, convs, active, _radio, _input = send_message(
        "second",
        chat_history=existing_history,
        conversations={"conv1": existing_history},
        active_id="conv1",
        model_label="Claude Sonnet 4.6",
        request=_FakeRequest("user-x"),
    )
    assert active == "conv1"
    assert len(chatbot) == 4
    assert chatbot[-1]["content"] == "second reply"
    assert convs["conv1"] == chatbot


def test_send_message_empty_input_is_noop(monkeypatch):
    """Whitespace-only / empty input should not invoke the orchestrator."""
    called = []
    monkeypatch.setattr(
        "app.chat_fn",
        lambda *_a, **_kw: (called.append(True), "should not be called")[1],
    )

    from app import send_message

    chatbot, _convs, active, _radio, _input = send_message(
        "   ",
        chat_history=[],
        conversations={},
        active_id=None,
        model_label="Claude Sonnet 4.6",
        request=_FakeRequest("user-x"),
    )
    assert called == []
    assert chatbot == []
    assert active is None


def test_send_message_isolates_conversations_in_state(monkeypatch):
    """Adding to one conversation must not mutate another."""
    monkeypatch.setattr("app.chat_fn", lambda *_a, **_kw: "reply-A")

    from app import send_message

    convs = {
        "convA": [{"role": "user", "content": "A1"}],
        "convB": [{"role": "user", "content": "B1"}],
    }
    _chatbot, updated, active, _radio, _input = send_message(
        "A2",
        chat_history=convs["convA"],
        conversations=convs,
        active_id="convA",
        model_label="Claude Sonnet 4.6",
        request=_FakeRequest("user-x"),
    )
    assert active == "convA"
    # convB untouched
    assert updated["convB"] == [{"role": "user", "content": "B1"}]
    # convA grew
    assert len(updated["convA"]) == 3


def test_stage_user_message_appends_immediately():
    from app import stage_user_message

    chatbot, convs, active, _radio, cleared = stage_user_message(
        "hello",
        chat_history=[],
        conversations={},
        active_id=None,
    )
    assert active is not None
    assert chatbot == [{"role": "user", "content": "hello"}]
    assert convs[active] == chatbot
    assert cleared == ""


def test_complete_assistant_message_uses_last_user_turn(monkeypatch):
    monkeypatch.setattr("app.chat_fn", lambda message, history, *_a, **_kw: f"reply:{message}:{len(history)}")

    from app import complete_assistant_message

    staged = [{"role": "user", "content": "hello"}]
    chatbot, convs, active, _radio = complete_assistant_message(
        staged,
        {"conv1": staged},
        "conv1",
        "Claude Sonnet 4.6",
        _FakeRequest("user-x"),
    )
    assert active == "conv1"
    assert chatbot[-1] == {"role": "assistant", "content": "reply:hello:0"}
    assert convs["conv1"] == chatbot
