"""Tests for the session-id ContextVar seam in memory/_session.py.

Phase 3 behavior: one default session, everything still works. The seam
exists so Phase 4 can switch session ids per Gradio user / per LangGraph
thread without touching every call site.
"""

from taste_agent.memory import (
    DEFAULT_SESSION_ID,
    current_session_id,
    get_default_episodic,
    get_default_semantic,
    reset_session_id,
    set_session_id,
)


def test_default_session_id_is_the_constant():
    assert current_session_id() == DEFAULT_SESSION_ID


def test_default_singleton_unchanged_per_session():
    a = get_default_semantic()
    b = get_default_semantic()
    assert a is b


def test_different_sessions_get_distinct_semantic_stores():
    token_a = set_session_id("session-a")
    try:
        store_a = get_default_semantic()
        store_a.write("dietary", "vegetarian")
    finally:
        reset_session_id(token_a)

    token_b = set_session_id("session-b")
    try:
        store_b = get_default_semantic()
        # Different session → different store, no leakage
        assert store_b is not store_a
        assert store_b.read("dietary") is None
    finally:
        reset_session_id(token_b)


def test_different_sessions_get_distinct_episodic_stores():
    token_a = set_session_id("epi-a")
    try:
        store_a = get_default_episodic()
    finally:
        reset_session_id(token_a)

    token_b = set_session_id("epi-b")
    try:
        store_b = get_default_episodic()
        assert store_b is not store_a
    finally:
        reset_session_id(token_b)


def test_returning_to_a_session_returns_the_same_store():
    """A session's default singleton persists across enter/exit/re-enter."""
    token1 = set_session_id("persistent")
    try:
        first = get_default_semantic()
        first.write("city", "Belgrade")
    finally:
        reset_session_id(token1)

    # Bounce somewhere else
    token2 = set_session_id("elsewhere")
    try:
        get_default_semantic()
    finally:
        reset_session_id(token2)

    # Return to the original
    token3 = set_session_id("persistent")
    try:
        again = get_default_semantic()
        assert again is first
        assert again.read("city").value == "Belgrade"
    finally:
        reset_session_id(token3)


def test_set_default_none_only_clears_current_session():
    from taste_agent.memory import set_default_semantic

    token_a = set_session_id("clear-test-a")
    try:
        store_a = get_default_semantic()
        store_a.write("k", "v")
    finally:
        reset_session_id(token_a)

    token_b = set_session_id("clear-test-b")
    try:
        # Clear in session b only
        set_default_semantic(None)
    finally:
        reset_session_id(token_b)

    # Session a survives
    token_a2 = set_session_id("clear-test-a")
    try:
        store_a_again = get_default_semantic()
        assert store_a_again is store_a
        assert store_a_again.read("k").value == "v"
    finally:
        reset_session_id(token_a2)
