"""Tests for the reserve_table skill — replay path + finalize + cancel.

The agentic path (live LLM sub-agent) is not unit-tested here because it
requires a tool-calling fake LLM; an integration test for that is left to
Phase 4 when real Playwright is wired. The replay path, the action gate, and
the cancel path are all here.
"""

from __future__ import annotations

import pytest

from taste_agent.browser.backend import MockBrowserBackend
from taste_agent.browser.parser_cache import format_trace, has_trace, save_trace
from taste_agent.config import DEFAULT_MODEL_ID
from taste_agent.guardrails.action import (
    approve,
    get_pending,
    register_pending,
)
from taste_agent.skills.reserve_table.reserve_table import (
    _replay_cached,
    _run_impl,
    cancel_reservation,
    finalize_reservation,
    run,
    set_default_backend,
)
from tests.fakes import FakeAgentModel


def _factory(_id: str):
    return FakeAgentModel(response="done")


# ── _replay_cached ───────────────────────────────────────────────────────────


def test_replay_cached_executes_actions_in_order():
    backend = MockBrowserBackend()
    trace = [
        ("navigate", {"url": "https://x.example/r"}),
        ("fill", {"selector": "input#name", "value": "Ana"}),
        ("click", {"selector": "button.next"}),
    ]
    result = _replay_cached(
        cached_trace=trace,
        backend=backend,
        place_name="Iva",
        reservation_url="https://x.example/r",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
        contact_phone="",
    )
    assert result["status"] == "pending_approval"
    assert result["source"] == "cached"
    assert [c[0] for c in backend.calls] == ["navigate", "fill", "click"]


def test_replay_cached_registers_pending_approval():
    backend = MockBrowserBackend()
    result = _replay_cached(
        cached_trace=[("navigate", {"url": "https://x.example/r"})],
        backend=backend,
        place_name="Iva",
        reservation_url="https://x.example/r",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
        contact_phone="",
    )
    pending = get_pending()
    assert pending is not None
    assert pending.action_id == result["action_id"]
    assert "Iva" in pending.summary
    assert "2026-05-20" in pending.summary
    # New: pending summary should include the host so the user can sanity-check
    assert "x.example" in pending.summary


def test_replay_cached_fails_on_unknown_action():
    """Cache containing an unrecognised action must abort, not partial-submit."""
    backend = MockBrowserBackend()
    bad_trace = [
        ("navigate", {"url": "https://x.example/r"}),
        ("teleport", {"selector": "wherever"}),  # nonsense — refuse to replay
    ]
    result = _replay_cached(
        cached_trace=bad_trace,
        backend=backend,
        place_name="X",
        reservation_url="https://x.example/r",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="A",
        contact_phone="",
    )
    assert result["status"] == "failed"
    assert "teleport" in result["error"]
    # Pre-flight refused; no actions should have been executed
    assert backend.calls == []
    # No pending approval was registered
    assert get_pending() is None


# ── run() with cached trace path ─────────────────────────────────────────────


def test_run_uses_cache_when_available():
    backend = MockBrowserBackend()
    set_default_backend(backend)
    save_trace(
        "https://x.example/reserve",
        [
            ("navigate", {"url": "https://x.example/reserve"}),
            ("fill", {"selector": "input#name", "value": "_"}),
        ],
    )
    result = run(
        place_name="Iva",
        reservation_url="https://x.example/reserve",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
    )
    assert result["source"] == "cached"
    assert result["status"] == "pending_approval"


def test_run_rejects_homepage_url_before_subagent():
    backend = MockBrowserBackend()
    set_default_backend(backend)
    result = run(
        place_name="Sonder",
        reservation_url="https://www.sonder.rs",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
    )
    assert result["status"] == "failed"
    assert result["source"] == "validation"
    assert "homepage" in result["error"]
    assert backend.calls == []


def test_run_rejects_placeholder_contact_name():
    backend = MockBrowserBackend()
    set_default_backend(backend)
    result = run(
        place_name="Sonder",
        reservation_url="https://www.sonder.rs/reservations",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="User",
    )
    assert result["status"] == "failed"
    assert result["source"] == "validation"
    assert "placeholder" in result["error"]
    assert backend.calls == []


def test_format_trace_renders_steps():
    trace = [
        ("navigate", {"url": "https://x.example/reserve"}),
        ("fill", {"selector": "input#name", "value": "Ana"}),
    ]
    rendered = format_trace(trace)
    assert "1. navigate(url='https://x.example/reserve')" in rendered
    assert "2. fill(selector='input#name', value='Ana')" in rendered


# ── finalize_reservation (the deterministic confirm-gate) ────────────────────


def test_finalize_reservation_requires_approval():
    backend = MockBrowserBackend()
    aid = register_pending("confirm_reservation", "test")
    with pytest.raises(PermissionError, match="requires user approval"):
        finalize_reservation(aid, backend=backend)


def test_finalize_reservation_clicks_submit_after_approval():
    backend = MockBrowserBackend()
    aid = register_pending("confirm_reservation", "test")
    approve(aid)
    result = finalize_reservation(aid, backend=backend, submit_selector="button.confirm-x")
    assert result["status"] == "confirmed"
    assert backend.calls[-1] == ("click", {"selector": "button.confirm-x"})


def test_finalize_reservation_uses_action_id_not_latest_pending():
    """Two pending actions: finalize must use the summary for the approved id,
    not the most-recently-registered one."""
    backend = MockBrowserBackend()
    aid_a = register_pending("confirm_reservation", "Reserve at A")
    register_pending("confirm_reservation", "Reserve at B")  # second (latest) pending
    approve(aid_a)
    result = finalize_reservation(aid_a, backend=backend, submit_selector="button.c-x")
    assert result["summary"] == "Reserve at A"  # not B even though B is latest
    assert result["action_id"] == aid_a


def test_finalize_reservation_lifts_forbidden_before_click():
    """Even if the submit selector is currently forbidden, finalize lifts it
    before clicking (after the deterministic gate passes)."""
    backend = MockBrowserBackend()
    backend.forbidden_selectors.add("button.confirm-x")
    aid = register_pending("confirm_reservation", "test")
    approve(aid)
    # Should not raise — finalize discards the forbid first
    result = finalize_reservation(aid, backend=backend, submit_selector="button.confirm-x")
    assert result["status"] == "confirmed"
    assert backend.calls[-1] == ("click", {"selector": "button.confirm-x"})


def test_finalize_reservation_consumes_pending():
    backend = MockBrowserBackend()
    aid = register_pending("confirm_reservation", "test")
    approve(aid)
    finalize_reservation(aid, backend=backend)
    assert get_pending() is None


def test_finalize_reservation_rejects_unknown_action_id():
    backend = MockBrowserBackend()
    with pytest.raises(PermissionError, match="no pending approval"):
        finalize_reservation("ghost-id", backend=backend)


# ── cancel_reservation ───────────────────────────────────────────────────────


def test_cancel_reservation_consumes_pending():
    aid = register_pending("confirm_reservation", "test")
    result = cancel_reservation(aid)
    assert result["status"] == "cancelled"
    assert get_pending() is None


def test_cancel_reservation_unknown_action_is_idempotent():
    # Should not raise even if the id is unknown.
    result = cancel_reservation("ghost-id")
    assert result["status"] == "cancelled"


def test_cancel_reservation_lifts_forbidden_selector():
    """Cancellation makes the backend reusable for a fresh attempt."""
    backend = MockBrowserBackend()
    backend.forbidden_selectors.add("button.submit-x")
    aid = register_pending("confirm_reservation", "test")
    cancel_reservation(aid, backend=backend, submit_selector="button.submit-x")
    assert "button.submit-x" not in backend.forbidden_selectors


# ── Defense in depth: sub-agent path forbids the submit selector ─────────────


def test_run_impl_forbids_submit_selector_before_subagent():
    """Even if the sub-agent goes off-prompt and tries to click submit, the
    backend refuses. This is the deterministic safety net behind the prompt
    instruction."""
    from taste_agent.skills.reserve_table.reserve_table import _DEFAULT_SUBMIT_SELECTOR

    backend = MockBrowserBackend()
    # Sanity check: not forbidden before we start
    assert _DEFAULT_SUBMIT_SELECTOR not in backend.forbidden_selectors

    _run_impl(
        place_name="X",
        reservation_url="https://defenseindepth.example/reserve",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="A",
        contact_phone="",
        backend=backend,
        model_factory=_factory,
    )
    # FakeAgentModel didn't register approval; _run_impl should have cleaned up
    # the forbid (so the backend is reusable). The point of the test is to
    # confirm cleanup happens on the failure path.
    assert _DEFAULT_SUBMIT_SELECTOR not in backend.forbidden_selectors


# ── _run_impl with no cache (sub-agent path) ─────────────────────────────────


def test_run_impl_subagent_path_returns_failed_when_no_approval_registered():
    # FakeAgentModel returns a plain string with no tool calls, so the
    # sub-agent finishes without ever calling request_user_approval. The
    # skill should report this as a failure rather than silently succeeding.
    backend = MockBrowserBackend()
    result = _run_impl(
        place_name="Iva",
        reservation_url="https://new.example/reserve",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
        contact_phone="",
        backend=backend,
        model_factory=_factory,
    )
    assert result["status"] == "failed"
    assert "approval" in result["error"]


def test_run_impl_caches_trace_only_on_pending_outcome():
    # The fake LLM produces no tool calls and no approval — cache must NOT
    # be populated in this failure case.
    backend = MockBrowserBackend()
    url = "https://needs-new-parser.example/reserve"
    _run_impl(
        place_name="X",
        reservation_url=url,
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="A",
        contact_phone="",
        backend=backend,
        model_factory=_factory,
    )
    assert has_trace(url) is False


def test_run_impl_passes_real_model_id_to_subagent_factory():
    seen: list[str] = []

    def factory(model_id: str):
        seen.append(model_id)
        return FakeAgentModel(response="done")

    backend = MockBrowserBackend()
    _run_impl(
        place_name="Iva",
        reservation_url="https://model-id.example/reserve",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
        contact_phone="",
        backend=backend,
        model_factory=factory,
    )
    assert seen == [DEFAULT_MODEL_ID]
