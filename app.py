"""Gradio chat UI for Taste Agent.

Loads .env before importing the orchestrator so LangSmith and provider keys
are visible to LangChain's auto-tracing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# IMPORTANT: load env before importing taste_agent / langchain — env vars like
# LANGSMITH_TRACING are checked at import time by LangChain integrations.
load_dotenv(Path(__file__).parent / ".env")

import gradio as gr  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402

from taste_agent.config import DEFAULT_MODEL_ID, MODEL_REGISTRY  # noqa: E402
from taste_agent.logging_ import configure_logging, get_logger  # noqa: E402
from taste_agent.memory import (  # noqa: E402
    get_default_episodic,
    get_default_semantic,
    reset_session_id,
    set_session_id,
)
from taste_agent.orchestrator import run_turn  # noqa: E402

configure_logging()
logger = get_logger(__name__)

_LABEL_TO_ID: dict[str, str] = {m.label: m.litellm_id for m in MODEL_REGISTRY}
_MODEL_LABELS: list[str] = [m.label for m in MODEL_REGISTRY]
_DEFAULT_LABEL: str = next(
    (m.label for m in MODEL_REGISTRY if m.litellm_id == DEFAULT_MODEL_ID),
    _MODEL_LABELS[0],
)

# Per-Gradio-session debug snapshots, keyed by ``gr.Request.session_hash``.
# The post-turn refresh hook reads from here to render turn-level signals
# (``n_facts_in_prompt``) without threading them through ChatInterface's
# return shape. Per-session so two browser tabs don't see each other's stats.
_LAST_DEBUG_BY_SESSION: dict[str, dict[str, Any]] = {}


def _session_id_of(request: gr.Request | None) -> str:
    """Resolve the session id for a Gradio request, or fall back to the
    process default for tests / non-Gradio callers."""
    if request is None:
        from taste_agent.memory import DEFAULT_SESSION_ID

        return DEFAULT_SESSION_ID
    return request.session_hash or "_default"


def _gradio_history_to_messages(
    history: list[dict[str, str]] | list[tuple[str, str]],
) -> list[BaseMessage]:
    """Convert Gradio ChatInterface history to LangChain messages.

    Accepts both the modern messages format (list of ``{"role", "content"}``
    dicts, default since Gradio 4.40) and the legacy tuples format.
    """
    msgs: list[BaseMessage] = []
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content", "")
            if not content:
                continue
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
        elif isinstance(entry, tuple) and len(entry) == 2:
            user, ai = entry
            if user:
                msgs.append(HumanMessage(content=user))
            if ai:
                msgs.append(AIMessage(content=ai))
    return msgs


def chat_fn(
    message: str,
    history: list[dict[str, str]] | list[tuple[str, str]],
    model_label: str,
    request: gr.Request | None = None,
) -> str:
    """Handler wired to gr.ChatInterface.

    Scopes memory to the Gradio session: every browser tab gets its own
    semantic + episodic stores, so two users can't see each other's facts.
    """
    sid = _session_id_of(request)
    token = set_session_id(sid)
    try:
        model_id = _LABEL_TO_ID.get(model_label, DEFAULT_MODEL_ID)
        lc_history = _gradio_history_to_messages(history)
        response, debug = run_turn(message, lc_history, model_id)
        _LAST_DEBUG_BY_SESSION[sid] = debug
        logger.info("turn complete session=%s debug=%s", sid, debug)
        return response
    finally:
        reset_session_id(token)


def _snapshot_semantic() -> dict[str, str]:
    return get_default_semantic().as_dict()


def _snapshot_episodic(k: int = 5) -> list[dict[str, object]]:
    """Return the k most recently logged episodic events, ordered by date desc."""
    try:
        events = get_default_episodic().list_recent(k=k)
    except Exception as e:
        logger.warning("episodic snapshot failed: %s", e)
        return []
    return [e.model_dump(exclude_none=True) for e in events]


def _refresh_panels(
    request: gr.Request | None = None,
) -> tuple[dict[str, str], list[dict[str, object]], str]:
    """Re-compute the side panels and the per-turn signal line.

    Reads from the same session as ``chat_fn`` so the panels show *this
    user's* memory, not a process-global view.
    """
    sid = _session_id_of(request)
    token = set_session_id(sid)
    try:
        debug = _LAST_DEBUG_BY_SESSION.get(sid, {})
        facts_n = debug.get("n_facts_in_prompt", 0)
        return (
            _snapshot_semantic(),
            _snapshot_episodic(),
            f"**Facts in prompt this turn:** {facts_n}",
        )
    finally:
        reset_session_id(token)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Taste Agent", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Taste Agent\n"
            "Personalized restaurant & café recommender for Belgrade and beyond. "
            "Pick a model and ask anything food-related."
        )
        model_dropdown = gr.Dropdown(
            choices=_MODEL_LABELS,
            value=_DEFAULT_LABEL,
            label="Model",
            interactive=True,
        )
        with gr.Row():
            with gr.Column(scale=3):
                chat_iface = gr.ChatInterface(
                    fn=chat_fn,
                    additional_inputs=[model_dropdown],
                    type="messages",
                    chatbot=gr.Chatbot(type="messages", height=500),
                    examples=[
                        ["Where can I find the best cappuccino in Belgrade?"],
                        ["Quiet café for working from with good wifi"],
                        ["Remember that I'm vegetarian and prefer quiet places"],
                        ["Recommend a vegetarian-friendly restaurant near Stari Grad"],
                    ],
                )
            with gr.Column(scale=1):
                gr.Markdown("### What I remember about you")
                semantic_view = gr.JSON(
                    value=_snapshot_semantic,
                    label="Semantic facts (durable)",
                )
                gr.Markdown("### Recent dining experiences")
                episodic_view = gr.JSON(
                    value=_snapshot_episodic,
                    label="Episodic memory (date-ordered)",
                )
                facts_counter = gr.Markdown("**Facts in prompt this turn:** 0")
                refresh_btn = gr.Button("Refresh memory", variant="secondary")
                refresh_btn.click(
                    fn=_refresh_panels,
                    outputs=[semantic_view, episodic_view, facts_counter],
                )

        # Auto-refresh: whenever the chatbot value changes (i.e., a turn just
        # completed) re-render the side panels and the facts-counter line.
        # This is the live "see memory grow as you talk" pedagogical signal.
        chat_iface.chatbot.change(
            fn=_refresh_panels,
            outputs=[semantic_view, episodic_view, facts_counter],
        )
    return app


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
