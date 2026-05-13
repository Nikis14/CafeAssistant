"""Reflection sub-agent: runs after every turn, updates memory automatically.

Why an agent (not a single LLM call): reflection needs to *read* existing
memory before deciding to *write*. That's a ReAct loop — read with
memory_read / memory_search, then write with memorize_* tools, then maybe
read again to verify, then queue a clarification if uncertain.

Architecture:

  - The reflection sub-agent is constructed per call in ``run_reflection``.
  - Its tools are LangChain ``@tool`` functions that share a ``ReflectionResult``
    via a ``ContextVar`` (no global state, no race between concurrent
    reflections — each ``run_reflection`` call sets up its own collector).
  - The orchestrator calls ``run_reflection(user_msg, agent_response, ...)``
    after the output guardrail and weaves any returned clarifications into
    the next reply.

Cost: one extra LLM call per turn (plus its tool-call iterations). For
production, run it sampled (every N turns), or batch a window. The seminar
runs it always-on so the magic is visible.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from taste_agent.logging_ import get_logger, trace
from taste_agent.memory.writes import write_episodic, write_semantic
from taste_agent.prompts import reflect_prompt
from taste_agent.tools.memory_read import memory_read
from taste_agent.tools.memory_search import memory_search

logger = get_logger(__name__)

ModelFactory = Callable[[str], BaseChatModel]

_REFLECTION_SKIP_ENV = "TASTE_AGENT_SKIP_REFLECTION"


@dataclass
class ReflectionResult:
    """Everything the reflection sub-agent collected during one run."""

    semantic_writes: list[dict[str, Any]] = field(default_factory=list)
    episodic_writes: list[dict[str, Any]] = field(default_factory=list)
    semantic_conflicts: list[dict[str, Any]] = field(default_factory=list)
    clarifications: list[str] = field(default_factory=list)
    tool_calls: int = 0
    skipped: bool = False
    error: str | None = None


# ContextVar so each run_reflection invocation gets its own isolated collector.
# Tools resolve this at call time. No globals, no cross-talk between concurrent
# reflections (which can happen if two Gradio sessions overlap).
_collector: ContextVar[ReflectionResult | None] = ContextVar(
    "reflection_collector", default=None
)


def _get_collector() -> ReflectionResult:
    coll = _collector.get()
    if coll is None:
        raise RuntimeError(
            "Reflection tool called outside a run_reflection() context — "
            "ContextVar is not initialized."
        )
    return coll


# ── Reflection-only tools ────────────────────────────────────────────────────


@tool
def memorize_semantic(
    key: str,
    value: str,
    confidence: float = 1.0,
    source: str = "explicit",
) -> str:
    """Write one durable user fact (semantic memory).

    Use ``source="explicit"`` when the user clearly stated this themselves.
    Use ``source="inferred"`` when you deduced it from context.

    Returns a short status string. If an existing explicit fact disagrees,
    the write is SKIPPED and the conflict is reported — call
    ``request_clarification`` to ask the user.
    """
    result = write_semantic(key, value, confidence=confidence, source=source)
    coll = _get_collector()
    coll.tool_calls += 1
    if result.get("written"):
        coll.semantic_writes.append(
            {"key": key, "value": value, "source": source, "confidence": confidence}
        )
        return f"written: {key}={value!r} (source={source})"
    if result.get("conflict"):
        coll.semantic_conflicts.append(result["conflict"])
        c = result["conflict"]
        return (
            f"conflict on {c['key']}: existing={c['existing_value']!r} "
            f"proposed={c['proposed_value']!r} — ask the user via request_clarification"
        )
    return f"error: {result.get('error', 'unknown')}"


@tool
def memorize_episodic(
    place_name: str,
    notes: str,
    rating: int | None = None,
    date: str | None = None,
    address: str | None = None,
    cuisine: str | None = None,
) -> str:
    """Log a dining experience the user described having.

    Do NOT log places the agent merely recommended — only experiences the
    user actually had.
    """
    result = write_episodic(
        place_name=place_name,
        notes=notes,
        rating=rating,
        date=date,
        address=address,
        cuisine=cuisine,
    )
    coll = _get_collector()
    coll.tool_calls += 1
    if result.get("written"):
        coll.episodic_writes.append(
            {"place_name": place_name, "doc_id": result["doc_id"]}
        )
        return f"logged episodic: {place_name}"
    return f"error: {result.get('error', 'unknown')}"


@tool
def request_clarification(question: str) -> str:
    """Queue a short question for the user, to be woven into the next reply.

    Use when something is potentially memory-worthy but you can't confidently
    write without confirmation.
    """
    coll = _get_collector()
    coll.tool_calls += 1
    q = question.strip()
    coll.clarifications.append(q)
    return f"clarification queued ({len(q)} chars)"


# ── Public entry point ──────────────────────────────────────────────────────


def run_reflection(
    user_message: str,
    agent_response: str,
    *,
    model_factory: ModelFactory | None = None,
    model_id: str = "anthropic/claude-haiku-4-5",
    skip: bool | None = None,
) -> ReflectionResult:
    """Spawn the reflection sub-agent. Returns the collected writes + clarifications.

    Args:
        user_message: what the user just said this turn.
        agent_response: what the agent replied this turn.
        model_factory: ``(model_id) -> BaseChatModel`` for the reflection LLM.
            If ``None``, reflection is skipped (graceful fallback).
        model_id: LiteLLM model id; default is Haiku for cost.
        skip: explicit override to disable reflection.
    """
    collector = ReflectionResult()

    if skip is None:
        skip = os.environ.get(_REFLECTION_SKIP_ENV) == "1"
    if skip or model_factory is None:
        collector.skipped = True
        return collector

    token = _collector.set(collector)
    try:
        # Lazy import — keeps module load light and tests fast.
        from langchain.agents import create_agent

        tools = [
            memory_read,
            memory_search,
            memorize_semantic,
            memorize_episodic,
            request_clarification,
        ]

        llm = model_factory(model_id)
        agent = create_agent(llm, tools)

        turn_payload = (
            f"User said: {user_message!r}\n\n"
            f"Agent replied: {agent_response!r}\n\n"
            "Reflect on this turn and update memory accordingly. Use your tools."
        )

        with trace("sub_agent:reflection", model=model_id):
            try:
                # Cap the sub-agent's iteration depth. Reflection should:
                # read once or twice → write a few times → maybe queue a
                # clarification → done. A misbehaving model that loops
                # read→read→read shouldn't burn through unlimited turns.
                agent.invoke(
                    {
                        "messages": [
                            SystemMessage(content=reflect_prompt()),
                            HumanMessage(content=turn_payload),
                        ]
                    },
                    config={"recursion_limit": 12},
                )
            except Exception as e:  # pragma: no cover - LLM API errors
                logger.warning("reflection sub-agent failed: %s", e)
                collector.error = str(e)
    finally:
        _collector.reset(token)

    return collector


def reset_collector_for_tests() -> None:
    """Test helper: clear the ContextVar so a leftover collector from a
    failed test doesn't bleed into the next."""
    import contextlib

    with contextlib.suppress(LookupError):
        _collector.set(None)
