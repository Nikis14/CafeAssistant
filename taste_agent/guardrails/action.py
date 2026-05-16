"""Action guardrail: deterministic confirm-gate on irreversible tool calls.

This is the star teaching moment for guardrails in the seminar. Contrast with
``taste_agent.guardrails.input``:

  - **Input guardrail** uses regex + (in Phase 4) LLM judges. It catches *fuzzy*
    classes of input — PII patterns, prompt-injection heuristics, off-scope
    requests. False positives and false negatives are acceptable.

  - **Action guardrail** (this module) uses set membership. It blocks
    *irreversible* actions until the user has explicitly approved a specific
    action_id. The check is black-and-white and NEVER delegated to an LLM. A
    model that *thinks* the user approved can still be wrong; the gate is the
    only source of truth.

The lesson: LLM-judges for ambiguity; deterministic gates for irreversibility.

Approval state is process-global for the demo (single-user Gradio session).
Phase 3 will scope this per session when memory and multi-user concerns enter.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from taste_agent.logging_ import get_logger

logger = get_logger(__name__)


@dataclass
class ApprovalRequest:
    """A pending action awaiting user approval."""

    action_id: str
    tool_name: str
    summary: str
    args: dict[str, Any] = field(default_factory=dict)


_PENDING: dict[str, ApprovalRequest] = {}
_APPROVED: set[str] = set()


def register_pending(tool_name: str, summary: str, args: dict[str, Any] | None = None) -> str:
    """Register a pending action and return the assigned ``action_id``."""
    action_id = uuid.uuid4().hex[:8]
    req = ApprovalRequest(
        action_id=action_id,
        tool_name=tool_name,
        summary=summary,
        args=args or {},
    )
    _PENDING[action_id] = req
    logger.info("action pending approval: tool=%s id=%s", tool_name, action_id)
    return action_id


def get_pending() -> ApprovalRequest | None:
    """Return the most-recent pending approval, if any."""
    if not _PENDING:
        return None
    # Insertion order is preserved; return the latest entry
    return next(reversed(_PENDING.values()))


def get(action_id: str) -> ApprovalRequest | None:
    """Return the pending approval with this id, or None if not registered.

    Prefer this over ``get_pending()`` when you have an explicit id — e.g.,
    inside ``finalize_reservation`` — so concurrent registrations don't cause
    you to read the wrong summary.
    """
    return _PENDING.get(action_id)


def approve(action_id: str) -> bool:
    """Mark ``action_id`` approved. Returns True on success, False if unknown."""
    if action_id not in _PENDING:
        logger.warning("approve called with unknown action_id=%s", action_id)
        return False
    _APPROVED.add(action_id)
    logger.info("action approved: id=%s", action_id)
    return True


def is_approved(action_id: str) -> bool:
    return action_id in _APPROVED


def consume(action_id: str) -> None:
    """Remove an action from pending and approved sets (call after execution)."""
    _PENDING.pop(action_id, None)
    _APPROVED.discard(action_id)


def gate_action(action_id: str, tool_name: str) -> None:
    """Raise ``PermissionError`` if ``action_id`` has not been user-approved.

    Call this *inside* any tool that performs an irreversible action. The check
    is deterministic — never delegated to an LLM judgment.
    """
    if action_id not in _PENDING:
        raise PermissionError(
            f"Action '{tool_name}' [{action_id}] has no pending approval request — "
            "it must be registered via request_user_approval first."
        )
    if not is_approved(action_id):
        raise PermissionError(
            f"Action '{tool_name}' [{action_id}] requires user approval before executing."
        )
    logger.info("action gate passed: tool=%s id=%s", tool_name, action_id)


def reset_action_state() -> None:
    """Clear all pending and approved actions. Test-only."""
    _PENDING.clear()
    _APPROVED.clear()
