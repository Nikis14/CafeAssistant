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
