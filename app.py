"""Gradio chat UI for Taste Agent.

Loads .env before importing the orchestrator so LangSmith and provider keys
are visible to LangChain's auto-tracing.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# IMPORTANT: load env before importing taste_agent / langchain — env vars like
# LANGSMITH_TRACING are checked at import time by LangChain integrations.
load_dotenv(Path(__file__).parent / ".env")

import gradio as gr  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402

from taste_agent.config import DEFAULT_MODEL_ID, MODEL_REGISTRY  # noqa: E402
from taste_agent.logging_ import configure_logging, get_logger  # noqa: E402
from taste_agent.orchestrator import run_turn  # noqa: E402

configure_logging()
logger = get_logger(__name__)

_LABEL_TO_ID: dict[str, str] = {m.label: m.litellm_id for m in MODEL_REGISTRY}
_MODEL_LABELS: list[str] = [m.label for m in MODEL_REGISTRY]
_DEFAULT_LABEL: str = next(
    (m.label for m in MODEL_REGISTRY if m.litellm_id == DEFAULT_MODEL_ID),
    _MODEL_LABELS[0],
)


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
) -> str:
    """Handler wired to gr.ChatInterface. Pure: history is provided each turn."""
    model_id = _LABEL_TO_ID.get(model_label, DEFAULT_MODEL_ID)
    lc_history = _gradio_history_to_messages(history)
    response, debug = run_turn(message, lc_history, model_id)
    logger.info("turn complete debug=%s", debug)
    return response


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
        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[model_dropdown],
            type="messages",
            chatbot=gr.Chatbot(type="messages", height=500),
            examples=[
                ["Where can I find the best cappuccino in Belgrade?"],
                ["Quiet café for working from with good wifi"],
                ["Recommend a vegetarian-friendly restaurant near Stari Grad"],
            ],
        )
    return app


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
