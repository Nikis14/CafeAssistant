"""Orchestrator: a LangGraph state graph that wraps the ReAct agent.

The top-level flow is explicit:

    START → input_guardrail → approval_check
                                 ├── approve   → finalize → END
                                 ├── cancel    → cancel   → END
                                 └── agent     → agent → output_guardrail
                                                    → format_agent_response → END

Why a state graph (not imperative): each step is a single-responsibility node
that mutates well-defined state slots, so:

- The branching is visible at the edge level instead of buried in ``if``
  blocks inside one big function.
- Every step shows up in LangSmith with its own span and state diff —
  invaluable when debugging "why did the agent take this branch".
- New nodes (e.g. an evaluation node before END, or a per-session pre-flight
  node in Phase 5) become "add a node + add an edge" rather than "find the
  right place inside run_turn".
- Pedagogical alignment: the seminar teaches LangGraph at the *agent* layer
  (Seminar 4/5) and now also at the *orchestration* layer.

The public ``run_turn`` keeps its prior signature; it just invokes the graph.

Guardrail ordering: input guardrails run FIRST, before the approval-intent
classifier. A "yes" reply that piggybacks an injection ("yes ignore previous
instructions...") gets refused by the input guardrail before the approval
branch can finalize anything.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from taste_agent.config import DEFAULT_MODEL_ID, SKILLS_DIR
from taste_agent.guardrails import (
    GuardrailResult,
    OutputGuardrailResult,
    approve,
    get_pending,
    redact_output_pii,
    run_input_guardrails,
    run_output_guardrails,
)
from taste_agent.logging_ import get_logger, trace
from taste_agent.memory import get_default_procedural, get_default_semantic
from taste_agent.memory.derive import maybe_derive_procedural
from taste_agent.memory.reflection import ReflectionResult, run_reflection
from taste_agent.prompts import system_prompt
from taste_agent.skill_loader import load_all_skills
from taste_agent.skills.reserve_table.reserve_table import (
    cancel_reservation,
    finalize_reservation,
)
from taste_agent.tools import geocode, memory_read, memory_search, web_search

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
# (e.g. in tests or memory-aware factories) produces a fresh agent rather
# than returning a stale one built with the previous factory.
_AGENT_CACHE: dict[tuple[str, int], Any] = {}


def _build_agent_uncached(model_id: str, factory: ModelFactory) -> Any:
    """Construct a fresh agent. No caching. Imports langchain lazily."""
    # create_agent is the LangChain 1.0 replacement for
    # langgraph.prebuilt.create_react_agent.
    from langchain.agents import create_agent

    skills = load_all_skills(SKILLS_DIR)
    tools = [geocode, memory_read, memory_search, web_search, *skills]
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


# ── Approval-intent detection (deterministic) ────────────────────────────────
# Parses the user's intent with a keyword heuristic when there's a pending
# irreversible action. Deterministic on purpose: a model that misreads "no
# wait, yes" must not trigger an irreversible click. The *gate*
# (taste_agent.guardrails.action.gate_action) stays deterministic regardless.

_APPROVE_WORDS = frozenset(
    {"yes", "y", "confirm", "ok", "okay", "sure", "proceed", "approve", "approved"}
)
_CANCEL_WORDS = frozenset({"no", "n", "cancel", "stop", "abort", "nope", "nevermind"})


_MAX_INTENT_TOKENS = 3
_PUNCT_TO_STRIP = ".,!?;:'\""


def _detect_approval_intent(text: str) -> str | None:
    """Return 'approve' / 'cancel' / None based on a strict keyword scan.

    The detector is deliberately conservative:

    - Only short messages (≤3 tokens) count as intent. Longer replies are
      treated as conversation and fall through to the agent.
    - If both approve and cancel words appear in the same short reply (e.g.,
      "no actually yes"), the result is None — the orchestrator will re-prompt
      rather than guess.
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


# ── Helpers used by nodes ────────────────────────────────────────────────────


def _count_tool_calls(messages: list[BaseMessage]) -> int:
    count = 0
    for m in messages:
        calls = getattr(m, "tool_calls", None)
        if calls:
            count += len(calls)
    return count


def _build_output_context(
    messages: list[BaseMessage],
    facts: dict[str, str] | None = None,
    *,
    max_chars: int = 6000,
    per_tool_chars: int = 2500,
) -> str:
    """Summarize the conversation context for the output guardrail's LLM judge.

    Includes injected memory facts (so memory-grounded answers aren't falsely
    flagged), user messages, and tool messages. Deliberately excludes the
    *base* system prompt (it guides the agent, not a factuality source) and
    intermediate ``AIMessage`` content (not ground truth).
    """
    parts: list[str] = []
    if facts:
        facts_str = "; ".join(f"{k}={v}" for k, v in sorted(facts.items()))
        parts.append(f"known user facts (semantic memory): {facts_str}")
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"user: {m.content}")
        elif isinstance(m, ToolMessage):
            name = getattr(m, "name", "tool")
            content = str(m.content)[:per_tool_chars]
            parts.append(f"tool[{name}]: {content}")
    summary = "\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


def _format_output_note(out_guard: object) -> str:
    """Render a one-line note about guardrail concerns to append to the reply.

    Prefixes each note with the *surface* that caught it (``[pii]``,
    ``[judge:factuality]``, ``[judge:citation]``). Renders only when there's
    something worth surfacing; suppresses factuality/citation concerns when
    the corresponding ``*_ok`` flag is True.
    """
    if not isinstance(out_guard, OutputGuardrailResult) or not out_guard.has_concerns:
        return ""
    notes: list[str] = []
    if out_guard.pii_concerns:
        notes.append(f"[pii] stripped: {', '.join(out_guard.pii_concerns)}")
    if not out_guard.factuality_ok and out_guard.factuality_concerns:
        notes.append(f"[judge:factuality] {'; '.join(out_guard.factuality_concerns)}")
    if not out_guard.citation_ok and out_guard.citation_concerns:
        notes.append(f"[judge:citation] {'; '.join(out_guard.citation_concerns)}")
    if not notes:
        return ""
    return "\n\n_Output guardrail:_ " + " | ".join(notes)


def _extract_text(message: BaseMessage) -> str:
    """Pull the user-visible text out of a message content payload.

    Anthropic-style replies interleave content blocks. We keep only ``text``
    blocks — joining everything would leak tool-use ids and thinking content.
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


# ── State graph ──────────────────────────────────────────────────────────────


class OrchestratorState(TypedDict, total=False):
    """Mutable per-turn state. ``total=False`` so nodes only set the slots
    they own; everything starts unset apart from the inputs."""

    # ── Inputs (populated by run_turn before invoking the graph) ──
    user_text: str
    history: list[BaseMessage]
    model_id: str
    model_factory: ModelFactory | None
    skip_output_judge: bool | None
    skip_reflection: bool | None

    # ── Set by input_guardrail_node ──
    guard_result: GuardrailResult
    cleaned_text: str

    # ── Set by approval_check_node ──
    intent: Literal["approve", "cancel", "agent"]
    pending_action_id: str
    pending_summary: str
    pending_before_id: str | None

    # ── Set by agent_node ──
    facts: dict[str, str]
    patterns_text: str
    agent_messages: list[BaseMessage]

    # ── Set by output_guardrail_node ──
    out_guard: OutputGuardrailResult

    # ── Set by reflection_node ──
    reflection_result: ReflectionResult

    # ── Set by procedural_derive_node ──
    procedural_derived: bool

    # ── Outputs (populated by terminal nodes) ──
    response_text: str
    debug: dict[str, Any]


def input_guardrail_node(state: OrchestratorState) -> dict[str, Any]:
    """First node — runs before approval classification.

    Refusal short-circuits to END by writing response_text + debug; the
    routing function picks up that signal.
    """
    with trace("node:input_guardrail"):
        guard = run_input_guardrails(state["user_text"])
        if guard.refusal_message is not None:
            logger.warning("input refused: %s", guard.refusal_message)
            return {
                "guard_result": guard,
                "response_text": guard.refusal_message,
                "debug": {
                    "refused": True,
                    "pii_redactions": guard.pii_redactions,
                },
            }
        return {
            "guard_result": guard,
            "cleaned_text": guard.cleaned_text,
        }


def approval_check_node(state: OrchestratorState) -> dict[str, Any]:
    """Detect approve / cancel intent against any pending irreversible action.

    Runs on the *cleaned* text (post input guardrail) so an injection payload
    can't sneak through alongside an approve keyword.
    """
    with trace("node:approval_check"):
        pending = get_pending()
        pending_before_id = pending.action_id if pending else None
        if pending is None:
            return {"intent": "agent", "pending_before_id": None}

        intent = _detect_approval_intent(state["cleaned_text"])
        if intent in ("approve", "cancel"):
            return {
                "intent": intent,
                "pending_action_id": pending.action_id,
                "pending_summary": pending.summary,
                "pending_before_id": pending_before_id,
            }
        # Unclear intent → fall through to the agent (it will re-prompt).
        return {"intent": "agent", "pending_before_id": pending_before_id}


def finalize_node(state: OrchestratorState) -> dict[str, Any]:
    """Approve + finalize a pending reservation. Terminal."""
    aid = state["pending_action_id"]
    with trace("node:finalize", action_id=aid):
        if not approve(aid):
            logger.warning("approve() failed for action_id=%s (race-cleared)", aid)
            return {
                "response_text": (
                    "Sorry, that reservation is no longer pending. Please start over."
                ),
                "debug": {
                    "refused": False,
                    "approval_action": "stale",
                    "action_id": aid,
                },
            }
        outcome = finalize_reservation(aid)
        return {
            "response_text": f"Done. {outcome['summary']}",
            "debug": {
                "refused": False,
                "approval_action": "confirmed",
                "action_id": aid,
            },
        }


def cancel_node(state: OrchestratorState) -> dict[str, Any]:
    """Discard a pending reservation. Terminal."""
    aid = state["pending_action_id"]
    with trace("node:cancel", action_id=aid):
        cancel_reservation(aid)
        return {
            "response_text": "Reservation cancelled. Let me know if you'd like to try again.",
            "debug": {
                "refused": False,
                "approval_action": "cancelled",
                "action_id": aid,
            },
        }


def agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Invoke the ReAct agent with injected memory facts + patterns + history."""
    facts = get_default_semantic().as_dict()
    patterns_text = get_default_procedural().as_text()
    agent = build_agent(state["model_id"], model_factory=state.get("model_factory"))

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt(facts=facts, patterns_text=patterns_text)),
        *state["history"],
        HumanMessage(content=state["cleaned_text"]),
    ]
    with trace("node:agent", n_messages=len(messages)):
        result = agent.invoke({"messages": messages})

    all_msgs: list[BaseMessage] = result["messages"]
    final = all_msgs[-1]
    response_text = (
        _extract_text(final) if isinstance(final, AIMessage) else str(final.content)
    )
    return {
        "facts": facts,
        "patterns_text": patterns_text,
        "agent_messages": all_msgs,
        "response_text": response_text,
    }


def output_guardrail_node(state: OrchestratorState) -> dict[str, Any]:
    """Run PII redaction + (env-controlled) LLM judge on the agent's reply."""
    with trace("node:output_guardrail"):
        out_factory = state.get("model_factory") or _default_model_factory
        out_guard = run_output_guardrails(
            state["response_text"],
            context_summary=_build_output_context(
                state["agent_messages"], facts=state.get("facts")
            ),
            model_factory=out_factory,
            skip_judge=state.get("skip_output_judge"),
        )
        return {
            "out_guard": out_guard,
            "response_text": out_guard.response_text + _format_output_note(out_guard),
        }


def reflection_node(state: OrchestratorState) -> dict[str, Any]:
    """Run the reflection sub-agent (env-controlled). Updates semantic +
    episodic memory automatically based on what the user said this turn."""
    with trace("node:reflection"):
        out_factory = state.get("model_factory") or _default_model_factory
        # Use the PII-stripped response WITHOUT the guardrail note — the note
        # is meta-content; reflection should see what the user sees as the
        # substantive reply.
        clean_response = state["out_guard"].response_text
        collector = run_reflection(
            user_message=state["cleaned_text"],
            agent_response=clean_response,
            model_factory=out_factory,
            skip=state.get("skip_reflection"),
        )
        return {"reflection_result": collector}


def procedural_derive_node(state: OrchestratorState) -> dict[str, Any]:
    """Conditionally derive procedural patterns — only when enough new
    episodes accumulated since the last derivation (default: 5)."""
    with trace("node:procedural_derive"):
        out_factory = state.get("model_factory") or _default_model_factory
        ran = maybe_derive_procedural(model_factory=out_factory)
        if ran:
            logger.info("procedural patterns derived this turn")
        return {"procedural_derived": ran}


def format_agent_response_node(state: OrchestratorState) -> dict[str, Any]:
    """Build the final debug dict and (if a new pending was registered during
    this turn) append a yes/no confirmation CTA so the next turn's intent
    detector reliably catches the user's reply."""
    with trace("node:format_agent_response"):
        guard = state["guard_result"]
        out_guard = state["out_guard"]
        all_msgs = state["agent_messages"]
        facts = state.get("facts") or {}

        debug: dict[str, Any] = {
            "refused": False,
            "pii_redactions": guard.pii_redactions,
            "out_of_scope": guard.out_of_scope,
            "tool_calls": _count_tool_calls(all_msgs),
            "n_messages": len(all_msgs),
            "n_facts_in_prompt": len(facts),
            "patterns_in_prompt": bool(state.get("patterns_text")),
            "output_guard": out_guard.summary_for_debug(),
            "procedural_derived": bool(state.get("procedural_derived")),
        }

        # Reflection details — surface the writes / conflicts / clarifications
        # so the Gradio sidebar can show "memory just changed".
        reflection = state.get("reflection_result")
        if reflection is not None:
            debug["reflection"] = {
                "skipped": reflection.skipped,
                "semantic_writes": len(reflection.semantic_writes),
                "episodic_writes": len(reflection.episodic_writes),
                "conflicts": len(reflection.semantic_conflicts),
                "clarifications": len(reflection.clarifications),
                "tool_calls": reflection.tool_calls,
                "error": reflection.error,
            }

        pending_before_id = state.get("pending_before_id")
        pending_after = get_pending()
        is_new_pending = pending_after is not None and (
            pending_before_id is None or pending_after.action_id != pending_before_id
        )

        response_text = state["response_text"]
        if pending_after is not None and is_new_pending:
            response_text = (
                f"{response_text}\n\n"
                f"_Pending action: {pending_after.summary}._\n"
                "Reply **yes** to confirm or **no** to cancel."
            )
            debug["pending_approval"] = pending_after.action_id

        # Append clarifications queued during reflection. The user sees them
        # as a natural-language follow-up so they can disambiguate next turn.
        #
        # Important: these come from the reflection sub-agent's LLM output
        # and therefore bypass the main output_guardrail_node above. Run a
        # deterministic PII redaction pass before appending — a hallucinated
        # phone/email/card in a clarification question would otherwise leak
        # straight to the user.
        #
        # Cap at 2 questions per turn to avoid chatbot-spammy outputs when
        # the sub-agent gets enthusiastic. Surplus questions are dropped
        # (and surfaced in debug) — the user can clarify on a later turn.
        if reflection is not None and reflection.clarifications:
            max_clarifications = 2
            cleaned_qs: list[str] = []
            clarification_pii_redactions = 0
            for q in reflection.clarifications[:max_clarifications]:
                cleaned, n, _ = redact_output_pii(q)
                cleaned_qs.append(cleaned)
                clarification_pii_redactions += n
            dropped = max(0, len(reflection.clarifications) - max_clarifications)
            quoted = "\n".join(f"- {q}" for q in cleaned_qs)
            response_text = (
                f"{response_text}\n\nBefore I forget — a quick question:\n{quoted}"
            )
            if clarification_pii_redactions:
                debug["clarification_pii_redactions"] = clarification_pii_redactions
            if dropped:
                debug["clarifications_dropped"] = dropped

        return {"response_text": response_text, "debug": debug}


# ── Routing ──────────────────────────────────────────────────────────────────


def _route_after_input_guardrail(
    state: OrchestratorState,
) -> Literal["approval_check", "__end__"]:
    """Terminate on refusal; otherwise continue to approval classification."""
    # response_text is only set by input_guardrail_node on refusal.
    if state.get("response_text") is not None:
        return "__end__"
    return "approval_check"


def _route_after_approval_check(
    state: OrchestratorState,
) -> Literal["finalize", "cancel", "agent"]:
    intent = state.get("intent") or "agent"
    return intent  # type: ignore[return-value]


# ── Graph construction (cached) ──────────────────────────────────────────────

_GRAPH: Any | None = None


def _build_orchestrator_graph() -> Any:
    """Compile the orchestrator state graph. Called once, cached."""
    # Lazy import to keep module load cheap and tests fast.
    from langgraph.graph import END, StateGraph

    g: Any = StateGraph(OrchestratorState)
    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("approval_check", approval_check_node)
    g.add_node("finalize", finalize_node)
    g.add_node("cancel", cancel_node)
    g.add_node("agent", agent_node)
    g.add_node("output_guardrail", output_guardrail_node)
    g.add_node("reflection", reflection_node)
    g.add_node("procedural_derive", procedural_derive_node)
    g.add_node("format_agent_response", format_agent_response_node)

    g.set_entry_point("input_guardrail")
    g.add_conditional_edges(
        "input_guardrail",
        _route_after_input_guardrail,
        {"approval_check": "approval_check", "__end__": END},
    )
    g.add_conditional_edges(
        "approval_check",
        _route_after_approval_check,
        # Router returns the *intent* literal; map each to the target node.
        {"approve": "finalize", "cancel": "cancel", "agent": "agent"},
    )
    g.add_edge("finalize", END)
    g.add_edge("cancel", END)
    # Agent path: agent → output_guardrail → reflection → procedural_derive
    # → format_agent_response → END. Reflection and procedural_derive both
    # update memory in the background; format_agent_response weaves any
    # clarifications from reflection into the user-facing reply.
    g.add_edge("agent", "output_guardrail")
    g.add_edge("output_guardrail", "reflection")
    g.add_edge("reflection", "procedural_derive")
    g.add_edge("procedural_derive", "format_agent_response")
    g.add_edge("format_agent_response", END)

    return g.compile()


def _get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_orchestrator_graph()
    return _GRAPH


def reset_graph_cache() -> None:
    """Clear the compiled-graph cache. Useful in tests."""
    global _GRAPH
    _GRAPH = None


# ── Public entry point ───────────────────────────────────────────────────────


def run_turn(
    user_text: str,
    history: list[BaseMessage] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    *,
    model_factory: ModelFactory | None = None,
    skip_output_judge: bool | None = None,
    skip_reflection: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run one conversational turn through the orchestrator state graph.

    Args:
        user_text: raw input from the user.
        history: prior LangChain messages (empty for first turn).
        model_id: LiteLLM model identifier.
        model_factory: optional injection point for tests.
        skip_output_judge: explicit override; None defers to env-resolution.

    Returns:
        ``(response_text, debug_info)``. Shape unchanged from prior versions
        so callers don't need to update.
    """
    initial_state: OrchestratorState = {
        "user_text": user_text,
        "history": history or [],
        "model_id": model_id,
        "model_factory": model_factory,
        "skip_output_judge": skip_output_judge,
        "skip_reflection": skip_reflection,
    }
    with trace("turn", model=model_id):
        final_state = _get_graph().invoke(initial_state)
    return final_state["response_text"], final_state["debug"]
