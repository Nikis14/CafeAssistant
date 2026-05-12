"""Tests for the action guardrail — deterministic confirm-gate."""

from __future__ import annotations

import pytest

from taste_agent.guardrails.action import (
    approve,
    consume,
    gate_action,
    get,
    get_pending,
    is_approved,
    register_pending,
)


def test_register_pending_returns_short_action_id():
    aid = register_pending("confirm_reservation", "test summary")
    assert isinstance(aid, str)
    assert len(aid) == 8


def test_get_pending_returns_none_when_empty():
    assert get_pending() is None


def test_get_pending_returns_latest():
    register_pending("confirm_reservation", "first")
    a2 = register_pending("confirm_reservation", "second")
    pending = get_pending()
    assert pending is not None
    assert pending.action_id == a2
    assert pending.summary == "second"


def test_approve_unknown_action_id_returns_false():
    assert approve("nonexistent") is False


def test_approve_known_action_id_returns_true():
    aid = register_pending("confirm_reservation", "x")
    assert approve(aid) is True
    assert is_approved(aid) is True


def test_gate_action_raises_without_registration():
    with pytest.raises(PermissionError, match="no pending approval"):
        gate_action("ghost-id", "confirm_reservation")


def test_gate_action_raises_when_not_yet_approved():
    aid = register_pending("confirm_reservation", "x")
    with pytest.raises(PermissionError, match="requires user approval"):
        gate_action(aid, "confirm_reservation")


def test_gate_action_passes_after_approval():
    aid = register_pending("confirm_reservation", "x")
    approve(aid)
    # Should NOT raise
    gate_action(aid, "confirm_reservation")


def test_consume_clears_pending_and_approved():
    aid = register_pending("confirm_reservation", "x")
    approve(aid)
    consume(aid)
    assert get_pending() is None
    assert is_approved(aid) is False


def test_action_state_is_isolated_per_test():
    # Relies on the autouse reset fixture in conftest.
    assert get_pending() is None


# ── get(action_id) lookup ────────────────────────────────────────────────────


def test_get_by_action_id_returns_specific_approval():
    aid_a = register_pending("confirm_reservation", "Reserve at A")
    aid_b = register_pending("confirm_reservation", "Reserve at B")
    # get_pending returns LATEST (B); get(aid_a) returns specifically A
    assert get(aid_a).summary == "Reserve at A"
    assert get(aid_b).summary == "Reserve at B"
    assert get_pending().summary == "Reserve at B"


def test_get_returns_none_for_unknown_action_id():
    assert get("nonexistent") is None
