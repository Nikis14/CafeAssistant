"""Orchestrator: input guardrails → ReAct agent → response.

Phase 1 keeps this deliberately linear: guardrail node, agent invocation,
return. Phase 2 adds the action-confirm gate after the agent. Phase 4 adds
the output guardrail.

Build-and-cache pattern: each model id maps to a constructed agent. The
function `build_agent` is the only place that knows about LiteLLM and the
skill registry — so tests can swap it via dependency injection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from taste_agent.config import DEFAULT_MODEL_ID, SKILLS_DIR
from taste_agent.guardrails import (
    approve,
    get_pending,
    run_input_guardrails,
)
from taste_agent.logging_ import get_logger, trace
from taste_agent.prompts import system_prompt
from taste_agent.skill_loader import load_all_skills
from taste_agent.skills.reserve_table.reserve_table import (
    cancel_reservation,
    finalize_reservation,
)
from taste_agent.tools import geocode

logger = get_logger(__name__)


# ── Model construction ───────────────────────────────────────────────────────
# Indirection so tests can inject a fake chat model without importing LiteLLM.
ModelFactory = Callable[[str], BaseChatModel]


def _default_model_factory(model_id: str) -> BaseChatModel:
    """Build a ChatLiteLLM for the given model id. Imports are lazy so tests
    that inject a fake factory don't need LiteLLM installed.
    """
    from langchain_litellm import ChatLiteLLM

    return ChatLiteLLM(model=model_id, temperature=0.2)


# ── Agent construction (cached per (model_id, factory)) ──────────────────────
# Cache key includes the factory identity so injecting a different factory
# (e.g., in tests or Phase 3's memory-aware factories) produces a fresh agent
# rather than returning a stale one built with the previous factory.
_AGENT_CACHE: dict[tuple[str, int], Any] = {}


def _build_agent_uncached(model_id: str, factory: ModelFactory) -> Any:
    """Construct a fresh agent. No caching. Imports langchain lazily."""
    # create_agent is the LangChain 1.0 replacement for
    # langgraph.prebuilt.create_react_agent.
    from langchain.agents import create_agent

    skills = load_all_skills(SKILLS_DIR)
    tools = [geocode, *skills]
    llm = factory(model_id)
    return create_agent(llm, tools)


def build_agent(model_id: str, model_factory: ModelFactory | None = None) -> Any:
    """Return a ReAct agent for the given model id, building once per (id, factory)."""
    factory = model_factory or _default_model_factory
    cache_key = (model_id, id(factory))
    if cache_key not in _AGENT_CACHE:
        logger.info("building agent for model=%s", model_id)
        _AGENT_CACHE[cache_key] = _build_agent_uncached(model_id, factory)
    return _AGENT_CACHE[cache_key]


def reset_agent_cache() -> None:
    """Clear the build cache. Useful in tests."""
    _AGENT_CACHE.clear()


# ── Approval-intent detection (deterministic, pre-agent) ─────────────────────
# We parse the user's intent with a keyword heuristic when there's a pending
# irreversible action. Deterministic on purpose: a model that misreads "no
# wait, yes" must not trigger an irreversible click. Phase 4 can upgrade this
# to an LLM judge — but the *gate* (taste_agent.guardrails.action.gate_action)
# stays deterministic regardless.

_APPROVE_WORDS = frozenset(
    {"yes", "y", "confirm", "ok", "okay", "sure", "proceed", "approve", "approved"}
)
_CANCEL_WORDS = frozenset(
    {"no", "n", "cancel", "stop", "abort", "nope", "nevermind"}
)


_MAX_INTENT_TOKENS = 3
_PUNCT_TO_STRIP = ".,!?;:'\""


def _detect_approval_intent(text: str) -> str | None:
    """Return 'approve' / 'cancel' / None based on a strict keyword scan.

    The detector is deliberately conservative:

    - Only short messages (≤3 tokens) count as intent. Longer replies are
      treated as conversation and fall through to the agent. This stops false
      positives like "What time does Café Yes open?" from finalizing an
      irreversible action.
    - If both approve and cancel words appear in the same short reply (e.g.,
      "no actually yes"), the result is None — the orchestrator will re-prompt
      rather than guess. Wrong-direction errors here finalize a real reservation.
    """
    cleaned = text.translate(str.maketrans("", "", _PUNCT_TO_STRIP)).strip().lower()
    tokens = cleaned.split()

    if not tokens or len(tokens) > _MAX_INTENT_TOKENS:
        return None

    token_set = set(tokens)
    has_approve = bool(token_set & _APPROVE_WORDS)
    has_cancel = bool(token_set & _CANCEL_WORDS)

    if has_approve and has_cancel:
        return None
    if has_approve:
        return "approve"
    if has_cancel:
        return "cancel"
    return None


# ── Turn execution ───────────────────────────────────────────────────────────


def _count_tool_calls(messages: list[BaseMessage]) -> int:
    count = 0
    for m in messages:
        calls = getattr(m, "tool_calls", None)
        if calls:
            count += len(calls)
    return count


def _extract_text(message: BaseMessage) -> str:
    """Pull the user-visible text out of a message content payload.

    Anthropic-style replies interleave content blocks: ``{"type": "text", ...}``,
    ``{"type": "tool_use", ...}``, ``{"type": "thinking", ...}``. We keep only
    ``text`` blocks — silently joining everything would leak tool-use IDs and
    thinking content into the chat UI.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def run_turn(
    user_text: str,
    history: list[BaseMessage] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    *,
    model_factory: ModelFactory | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run one conversational turn through the full pipeline.

    Args:
        user_text: raw input from the user.
        history: prior LangChain messages (empty for first turn).
        model_id: LiteLLM model identifier.
        model_factory: optional injection point for tests.

    Returns:
        (response_text, debug_info). ``debug_info`` includes pii_redactions
        and tool_calls counts.
    """
    history = history or []

    with trace("turn", model=model_id):
        # 0. Approval-flow handling. If there's a pending irreversible action
        # AND the user's text is a clear approve / cancel, handle deterministically
        # without invoking the agent. The action guardrail (gate_action) is the
        # one source of truth for "may this run" — never the LLM.
        #
        # Order note: this branch runs BEFORE the input guardrail. A "yes"
        # reply must not be rejected as out-of-scope, and a short reply has
        # near-zero attack surface for prompt injection (no room for a payload
        # alongside an approve/cancel keyword given the ≤3-token cap).
        pending_before = get_pending()
        if pending_before is not None:
            intent = _detect_approval_intent(user_text)
            if intent == "approve":
                if not approve(pending_before.action_id):
                    logger.warning(
                        "approve() failed for action_id=%s — pending was cleared "
                        "between detect and approve",
                        pending_before.action_id,
                    )
                    return (
                        "Sorry, that reservation is no longer pending. Please start over.",
                        {
                            "refused": False,
                            "approval_action": "stale",
                            "action_id": pending_before.action_id,
                        },
                    )
                with trace("finalize_pending", action_id=pending_before.action_id):
                    outcome = finalize_reservation(pending_before.action_id)
                msg = f"Done. {outcome['summary']}"
                return msg, {
                    "refused": False,
                    "approval_action": "confirmed",
                    "action_id": pending_before.action_id,
                }
            if intent == "cancel":
                with trace("cancel_pending", action_id=pending_before.action_id):
                    cancel_reservation(pending_before.action_id)
                return "Reservation cancelled. Let me know if you'd like to try again.", {
                    "refused": False,
                    "approval_action": "cancelled",
                    "action_id": pending_before.action_id,
                }
            # Unclear intent — fall through. Agent will see history and re-prompt.

        # 1. Input guardrails (deterministic, pre-LLM)
        guard = run_input_guardrails(user_text)
        if guard.refusal_message is not None:
            logger.warning("input refused: %s", guard.refusal_message)
            return guard.refusal_message, {
                "refused": True,
                "pii_redactions": guard.pii_redactions,
            }

        # 2. Build/get agent
        agent = build_agent(model_id, model_factory=model_factory)

        # 3. Invoke. We prepend SystemMessage per turn rather than passing
        # `prompt=...` to `create_agent` so the time-stamp in `system_prompt()`
        # stays fresh — the agent is cached and a static prompt would freeze
        # the time at build time. Trade-off: LangSmith shows the prompt as an
        # inline message rather than the agent's system slot. A callable
        # prompt would also work and is worth revisiting in Phase 4.
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt()),
            *history,
            HumanMessage(content=guard.cleaned_text),
        ]
        with trace("agent:invoke", n_messages=len(messages)):
            result = agent.invoke({"messages": messages})

        # 4. Extract response (Phase 4 wires the output guardrail here)
        all_msgs: list[BaseMessage] = result["messages"]
        final = all_msgs[-1]
        response_text = _extract_text(final) if isinstance(final, AIMessage) else str(final.content)

        debug: dict[str, Any] = {
            "refused": False,
            "pii_redactions": guard.pii_redactions,
            "out_of_scope": guard.out_of_scope,
            "tool_calls": _count_tool_calls(all_msgs),
            "n_messages": len(all_msgs),
        }

        # 5. If a *new* pending action was created during this turn, ensure the
        # user sees a clear confirmation prompt. The agent's response may
        # already mention it, but we make the deterministic CTA explicit so the
        # next-turn intent detector reliably catches "yes" / "no".
        pending_after = get_pending()
        is_new_pending = pending_after is not None and (
            pending_before is None or pending_after.action_id != pending_before.action_id
        )
        if pending_after is not None and is_new_pending:
            response_text = (
                f"{response_text}\n\n"
                f"_Pending action: {pending_after.summary}._\n"
                "Reply **yes** to confirm or **no** to cancel."
            )
            debug["pending_approval"] = pending_after.action_id

        return response_text, debug
