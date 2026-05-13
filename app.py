"""Gradio chat UI for Taste Agent.

Loads .env before importing the orchestrator so LangSmith and provider keys
are visible to LangChain's auto-tracing.
"""

from __future__ import annotations

import uuid
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
    get_default_procedural,
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
# (``n_facts_in_prompt``) without threading them through Gradio's return shape.
# Per-session so two browser tabs don't see each other's stats.
_LAST_DEBUG_BY_SESSION: dict[str, dict[str, Any]] = {}


def _session_id_of(request: gr.Request | None) -> str:
    """Resolve the session id for a Gradio request, or fall back to the
    process default for tests / non-Gradio callers."""
    if request is None:
        from taste_agent.memory import DEFAULT_SESSION_ID

        return DEFAULT_SESSION_ID
    return request.session_hash or "_default"


# ── History adapters ────────────────────────────────────────────────────────


def _gradio_history_to_messages(
    history: list[dict[str, str]] | list[tuple[str, str]],
) -> list[BaseMessage]:
    """Convert Gradio chatbot history to LangChain messages.

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
    """Run one turn through the orchestrator. Returns the assistant's reply.

    Scopes memory to the Gradio session: every browser tab gets its own
    semantic + episodic stores. Multiple conversations *within* the same
    tab share one memory store (same session_hash) — like ChatGPT's
    "memory persists across chats for the same user".
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


# ── Memory snapshots (side panels) ──────────────────────────────────────────


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


def _snapshot_procedural() -> list[dict[str, object]]:
    """Return current inferred behavioral patterns for the side panel."""
    try:
        patterns = get_default_procedural().all()
    except Exception as e:
        logger.warning("procedural snapshot failed: %s", e)
        return []
    return [p.model_dump(exclude_none=True) for p in patterns]


def _refresh_panels(
    request: gr.Request | None = None,
) -> tuple[dict[str, str], list[dict[str, object]], list[dict[str, object]], str]:
    """Re-compute the side panels and the per-turn signal line.

    Reads from the same session as ``chat_fn`` so the panels show *this
    user's* memory, not a process-global view.
    """
    sid = _session_id_of(request)
    token = set_session_id(sid)
    try:
        debug = _LAST_DEBUG_BY_SESSION.get(sid, {})
        facts_n = debug.get("n_facts_in_prompt", 0)
        patterns_count = len(_snapshot_procedural())
        reflection_info = debug.get("reflection", {})
        reflection_summary = ""
        if reflection_info and not reflection_info.get("skipped"):
            sw = reflection_info.get("semantic_writes", 0)
            ew = reflection_info.get("episodic_writes", 0)
            cw = reflection_info.get("clarifications", 0)
            if sw or ew or cw:
                reflection_summary = (
                    f" · reflection: +{sw} fact(s), +{ew} event(s), {cw} clarification(s)"
                )
        return (
            _snapshot_semantic(),
            _snapshot_episodic(),
            _snapshot_procedural(),
            f"**Facts:** {facts_n} | **Patterns:** {patterns_count}{reflection_summary}",
        )
    finally:
        reset_session_id(token)


# ── Conversation-list state (within one tab) ────────────────────────────────
#
# Gradio doesn't ship a multi-conversation UI; we build one with a
# ``gr.State`` holding ``{conversation_id: list[message_dict]}`` plus an
# active-id pointer. Switching conversations swaps the chatbot's value;
# memory (semantic + episodic) is per-tab and shared across all
# conversations in that tab — the same model as ChatGPT.

_TITLE_MAXLEN = 40


def _new_conversation_id() -> str:
    return uuid.uuid4().hex[:8]


def _title_for(history: list[dict[str, str]]) -> str:
    """Use the first user message (truncated) as the conversation title."""
    for msg in history:
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = (msg.get("content") or "").strip()
            if text:
                if len(text) > _TITLE_MAXLEN:
                    return text[:_TITLE_MAXLEN] + "..."
                return text
    return "New chat"


def _conv_choices(
    conversations: dict[str, list[dict[str, str]]],
) -> list[tuple[str, str]]:
    """Build ``gr.Radio`` choices ``[(label, value), ...]`` from the store.

    Most-recently-created appears first so a fresh "New chat" lands on top.
    Conversations dict preserves insertion order; we reverse it.
    """
    items = list(conversations.items())
    items.reverse()
    return [
        (f"{cid[:4]} · {_title_for(history)}", cid)
        for cid, history in items
    ]


def create_new_conversation(
    conversations: dict[str, list[dict[str, str]]],
) -> tuple[Any, ...]:
    """Wire to the 'New chat' button."""
    new_id = _new_conversation_id()
    updated = {**conversations, new_id: []}
    return (
        updated,                                         # conversations_state
        new_id,                                          # active_conv_id
        [],                                              # chatbot value (empty)
        gr.update(choices=_conv_choices(updated), value=new_id),  # conv_list radio
    )


def load_conversation(
    selected_id: str | None,
    conversations: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, str]], str | None]:
    """Wire to the conversation-list radio's change event."""
    if selected_id is None or selected_id not in conversations:
        return [], None
    return conversations[selected_id], selected_id


def send_message(
    user_msg: str,
    chat_history: list[dict[str, str]],
    conversations: dict[str, list[dict[str, str]]],
    active_id: str | None,
    model_label: str,
    request: gr.Request | None = None,
) -> tuple[Any, ...]:
    """Wire to the textbox submit / Send button.

    Auto-creates a conversation on the first send if none is active.
    """
    if not user_msg or not user_msg.strip():
        # No-op: return unchanged state + cleared input
        return (
            chat_history,
            conversations,
            active_id,
            gr.update(choices=_conv_choices(conversations), value=active_id),
            "",
        )

    # Ensure an active conversation exists
    if active_id is None or active_id not in conversations:
        active_id = _new_conversation_id()
        conversations = {**conversations, active_id: []}

    # Call the orchestrator with the PRIOR conversation history. The current
    # user message lives in the ``user_msg`` parameter; ``chat_fn`` / agent_node
    # adds it once to the message list. Appending it here AND passing it
    # separately would send the same turn to the model twice.
    response = chat_fn(user_msg, chat_history, model_label, request)
    history_with_reply = [
        *chat_history,
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response},
    ]

    # Persist
    updated_conversations = {**conversations, active_id: history_with_reply}

    return (
        history_with_reply,
        updated_conversations,
        active_id,
        gr.update(choices=_conv_choices(updated_conversations), value=active_id),
        "",
    )


# ── UI assembly ─────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Taste Agent", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Taste Agent\n"
            "Personalized restaurant & café recommender for Belgrade and beyond. "
            "Pick a model, start a new chat, or pick a past one from the sidebar."
        )
        model_dropdown = gr.Dropdown(
            choices=_MODEL_LABELS,
            value=_DEFAULT_LABEL,
            label="Model",
            interactive=True,
        )

        # Per-session state (lives for the lifetime of this browser session)
        conversations_state = gr.State({})  # dict[conv_id, list[msg]]
        active_conv_state = gr.State(None)  # str | None

        with gr.Row():
            # ── Left sidebar: conversations ─────────────────────────────
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### Conversations")
                new_chat_btn = gr.Button("+ New chat", variant="primary")
                conv_list = gr.Radio(
                    choices=[],
                    value=None,
                    label="Past chats",
                    interactive=True,
                )

            # ── Middle: chat ────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=500, label="Chat")
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about restaurants, cafés, reservations...",
                        show_label=False,
                        scale=4,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            # ── Right sidebar: memory ───────────────────────────────────
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### What I remember about you")
                semantic_view = gr.JSON(
                    value=_snapshot_semantic,
                    label="Semantic facts (what you told me)",
                )
                gr.Markdown("### Recent dining experiences")
                episodic_view = gr.JSON(
                    value=_snapshot_episodic,
                    label="Episodic memory (date-ordered)",
                )
                gr.Markdown("### Patterns I've noticed")
                procedural_view = gr.JSON(
                    value=_snapshot_procedural,
                    label="Inferred patterns (derived every ~5 episodes)",
                )
                facts_counter = gr.Markdown("**Facts:** 0 | **Patterns:** 0")
                refresh_btn = gr.Button("Refresh memory", variant="secondary")
                refresh_btn.click(
                    fn=_refresh_panels,
                    outputs=[
                        semantic_view,
                        episodic_view,
                        procedural_view,
                        facts_counter,
                    ],
                )

        # ── Event wiring ────────────────────────────────────────────────

        # New chat: create a fresh conversation id, clear the chatbot.
        new_chat_btn.click(
            fn=create_new_conversation,
            inputs=[conversations_state],
            outputs=[conversations_state, active_conv_state, chatbot, conv_list],
        )

        # Picking a conversation from the radio: load its messages.
        conv_list.change(
            fn=load_conversation,
            inputs=[conv_list, conversations_state],
            outputs=[chatbot, active_conv_state],
        )

        # Submit: text-input enter OR send button.
        send_event_inputs = [
            msg_input,
            chatbot,
            conversations_state,
            active_conv_state,
            model_dropdown,
        ]
        send_event_outputs = [
            chatbot,
            conversations_state,
            active_conv_state,
            conv_list,
            msg_input,  # cleared after send
        ]
        msg_input.submit(
            fn=send_message,
            inputs=send_event_inputs,
            outputs=send_event_outputs,
        )
        send_btn.click(
            fn=send_message,
            inputs=send_event_inputs,
            outputs=send_event_outputs,
        )

        # Auto-refresh memory panels on every chatbot change (i.e., after each
        # turn or conversation switch). Live "memory grows as you talk" beat.
        chatbot.change(
            fn=_refresh_panels,
            outputs=[semantic_view, episodic_view, procedural_view, facts_counter],
        )

    return app


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
