"""Session-id context variable for per-session memory scoping.

Phase 3 ships with a single global session (``"_default"``) — behavior is
unchanged from the pre-ContextVar version. The seam exists so Phase 4 can
introduce real per-session memory (per Gradio user, per LangGraph thread,
etc.) without rewriting every call site of ``get_default_semantic()`` /
``get_default_episodic()``.

Usage in Phase 4 will be:

    token = set_session_id("user-123")
    try:
        # everything inside sees the user-123 stores
        ...
    finally:
        reset_session_id(token)
"""

from __future__ import annotations

from contextvars import ContextVar, Token

DEFAULT_SESSION_ID = "_default"

_session_id: ContextVar[str] = ContextVar("taste_agent_session_id", default=DEFAULT_SESSION_ID)


def current_session_id() -> str:
    return _session_id.get()


def set_session_id(sid: str) -> Token[str]:
    return _session_id.set(sid)


def reset_session_id(token: Token[str]) -> None:
    _session_id.reset(token)
