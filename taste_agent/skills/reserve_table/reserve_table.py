"""reserve_table skill — drives a browser sub-agent to fill a reservation form.

The public ``run`` function is what the SKILL.md loader exposes to the
orchestrator agent. ``_run_impl`` is the testable inner function that accepts
a backend and model factory.

The skill never actually clicks the final submit. It stops at
``request_user_approval`` and returns a pending status. Finalization is the
orchestrator's job, gated by ``taste_agent.guardrails.action``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

from taste_agent.browser.backend import BrowserBackend, MockBrowserBackend
from taste_agent.browser.parser_cache import (
    ActionTrace,
    get_trace,
    has_trace,
    host_of,
    save_trace,
)
from taste_agent.browser.sub_agent import run_browser_subagent
from taste_agent.guardrails.action import (
    consume,
    gate_action,
    get,
    register_pending,
)
from taste_agent.logging_ import get_logger, trace

logger = get_logger(__name__)

# Default selector for the final submit button. Real Playwright + Phase 4 will
# either rely on cached selectors or have the sub-agent report it explicitly.
_DEFAULT_SUBMIT_SELECTOR = "button.confirm-reservation"

# Module-level backend default. The orchestrator (or tests) can swap it via
# ``set_default_backend``. Single-process demo; Phase 3 will scope per session.
_DEFAULT_BACKEND: BrowserBackend | None = None


def set_default_backend(backend: BrowserBackend) -> None:
    """Override the module-level backend used by ``run`` / ``finalize_reservation``."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = backend


def _get_default_backend() -> BrowserBackend:
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = MockBrowserBackend()
    return _DEFAULT_BACKEND


def _default_model_factory(model_id: str) -> BaseChatModel:
    """Same lazy-LiteLLM factory pattern as the main orchestrator."""
    from langchain_litellm import ChatLiteLLM

    return ChatLiteLLM(model=model_id, temperature=0.2)


def _format_goal(
    *,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
) -> str:
    phone_part = f", phone {contact_phone}" if contact_phone else ""
    return (
        f"Reserve a table at {place_name}. The reservation page is at "
        f"{reservation_url}. Details: date {date}, time {time}, party size "
        f"{party_size}, name {contact_name}{phone_part}. Fill the form and "
        f"call request_user_approval when the form is ready to submit. Do not "
        f"click the final submit button."
    )


def _format_summary(
    *,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
) -> str:
    phone_part = f", phone {contact_phone}" if contact_phone else ""
    host = host_of(reservation_url)
    return (
        f"Reserve at {place_name} (via {host}): {date} {time}, party of {party_size}, "
        f"name {contact_name}{phone_part}"
    )


_REPLAYABLE_ACTIONS = {"navigate", "click", "fill", "wait_for", "dom_snapshot"}


def _replay_cached(
    *,
    cached_trace: ActionTrace,
    backend: BrowserBackend,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
) -> dict[str, Any]:
    """Replay a cached action sequence.

    Phase 2 replays verbatim — the summary returned to the user reflects the
    *new* arguments so they can detect mismatches and cancel. Phase 4 will
    parameterize cached traces so a single trace works for any date/party.

    Refuses to replay if the cache contains an unknown action — forces
    re-discovery via the sub-agent path rather than risk a partial submit.
    """
    with trace("skill:reserve_table:replay_cached", n_actions=len(cached_trace)):
        # Pre-flight: any unknown action means the cache is stale / corrupt.
        # Better to fall back to a fresh sub-agent run than risk a partial form.
        for action_name, _ in cached_trace:
            if action_name not in _REPLAYABLE_ACTIONS:
                logger.error(
                    "unknown cached action %r in trace for %s — refusing to replay",
                    action_name,
                    reservation_url,
                )
                return {
                    "status": "failed",
                    "error": f"cached trace contains unknown action {action_name!r}; "
                    "cache may be stale — clear and retry",
                }

        for action_name, args in cached_trace:
            if action_name == "navigate":
                backend.navigate(str(args.get("url", "")))
            elif action_name == "click":
                backend.click(str(args.get("selector", "")))
            elif action_name == "fill":
                backend.fill(str(args.get("selector", "")), str(args.get("value", "")))
            elif action_name == "wait_for":
                timeout_ms = args.get("timeout_ms", 5000)
                backend.wait_for(
                    str(args.get("selector", "")),
                    timeout_ms=int(timeout_ms) if isinstance(timeout_ms, int) else 5000,
                )
            elif action_name == "dom_snapshot":
                backend.dom_snapshot(args.get("selector"))  # type: ignore[arg-type]

        summary = _format_summary(
            place_name=place_name,
            reservation_url=reservation_url,
            date=date,
            time=time,
            party_size=party_size,
            contact_name=contact_name,
            contact_phone=contact_phone,
        )
        action_id = register_pending(
            tool_name="confirm_reservation",
            summary=summary,
            args={"source": "cached", "submit_selector": _DEFAULT_SUBMIT_SELECTOR},
        )
        return {
            "status": "pending_approval",
            "action_id": action_id,
            "summary": summary,
            "source": "cached",
        }


def _run_impl(
    *,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
    backend: BrowserBackend,
    model_factory: Callable[[str], BaseChatModel],
) -> dict[str, Any]:
    with trace(
        "skill:reserve_table",
        place=place_name,
        url=reservation_url,
        cached=has_trace(reservation_url),
    ):
        # Defense in depth: forbid the submit selector at the backend level
        # *before* any LLM-driven activity touches the browser. If the
        # sub-agent goes off-prompt, the backend itself refuses the click.
        # The forbid is lifted only by ``finalize_reservation`` (after the
        # deterministic gate passes) or ``cancel_reservation``.
        backend.forbidden_selectors.add(_DEFAULT_SUBMIT_SELECTOR)

        cached = get_trace(reservation_url)
        if cached is not None:
            return _replay_cached(
                cached_trace=cached,
                backend=backend,
                place_name=place_name,
                reservation_url=reservation_url,
                date=date,
                time=time,
                party_size=party_size,
                contact_name=contact_name,
                contact_phone=contact_phone,
            )

        goal = _format_goal(
            place_name=place_name,
            reservation_url=reservation_url,
            date=date,
            time=time,
            party_size=party_size,
            contact_name=contact_name,
            contact_phone=contact_phone,
        )

        try:
            result = run_browser_subagent(
                goal=goal, backend=backend, model_factory=model_factory
            )
        except Exception:
            # Sub-agent crashed — drop the forbid so the backend is reusable.
            backend.forbidden_selectors.discard(_DEFAULT_SUBMIT_SELECTOR)
            raise

        # We don't pin to a specific id here; the sub-agent is the only thing
        # that registers a pending action in this code path.
        pending_after = None
        from taste_agent.guardrails.action import get_pending

        pending_after = get_pending()
        if pending_after is None:
            logger.warning("sub-agent finished without registering approval")
            backend.forbidden_selectors.discard(_DEFAULT_SUBMIT_SELECTOR)
            return {
                "status": "failed",
                "error": "sub-agent finished without registering approval",
                "actions": result.get("actions", []),
            }

        actions = result.get("actions", [])
        save_trace(reservation_url, actions)

        return {
            "status": "pending_approval",
            "action_id": pending_after.action_id,
            "summary": pending_after.summary,
            "source": "agentic",
            "n_actions": len(actions),
        }


def run(
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str = "",
) -> dict[str, Any]:
    """Drive a browser sub-agent to fill a reservation form. STOPS before submit.

    Returns:
        Dict with ``status`` ("pending_approval" or "failed"). When pending,
        also includes ``action_id`` and ``summary`` for the user-approval flow.
    """
    return _run_impl(
        place_name=place_name,
        reservation_url=reservation_url,
        date=date,
        time=time,
        party_size=party_size,
        contact_name=contact_name,
        contact_phone=contact_phone,
        backend=_get_default_backend(),
        model_factory=_default_model_factory,
    )


def finalize_reservation(
    action_id: str,
    backend: BrowserBackend | None = None,
    submit_selector: str = _DEFAULT_SUBMIT_SELECTOR,
) -> dict[str, Any]:
    """Click the final submit button. Gated by the action guardrail.

    Raises ``PermissionError`` if ``action_id`` has not been user-approved.
    Looks up the approval by ``action_id`` (not by "most recent pending") so a
    concurrent registration cannot trick us into displaying the wrong summary.
    Consumes the pending action on success.
    """
    with trace("finalize_reservation", action_id=action_id):
        gate_action(action_id, tool_name="confirm_reservation")
        bk = backend or _get_default_backend()
        # Look up by id, not "latest pending" — concurrent registrations are
        # possible and the gate validated *this* id, so use *this* summary.
        approval = get(action_id)
        summary = approval.summary if approval else "(unknown)"
        # Lift the forbid for this specific selector now that the gate passed,
        # then perform the irreversible action.
        bk.forbidden_selectors.discard(submit_selector)
        bk.click(submit_selector)
        consume(action_id)
        logger.info("reservation finalized: %s", summary)
        return {"status": "confirmed", "action_id": action_id, "summary": summary}


def cancel_reservation(
    action_id: str,
    backend: BrowserBackend | None = None,
    submit_selector: str = _DEFAULT_SUBMIT_SELECTOR,
) -> dict[str, Any]:
    """Discard a pending reservation without submitting.

    Lifts the backend's submit-forbid so the backend is reusable for a future
    reservation attempt.
    """
    with trace("cancel_reservation", action_id=action_id):
        bk = backend or _get_default_backend()
        bk.forbidden_selectors.discard(submit_selector)
        consume(action_id)
        logger.info("reservation cancelled: %s", action_id)
        return {"status": "cancelled", "action_id": action_id}
