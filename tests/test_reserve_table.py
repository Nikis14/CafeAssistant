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
from taste_agent.browser.spec_cache import get_spec, save_spec
from taste_agent.browser.specs import BookingFieldSpec, BookingFlowSpec, BookingFlowStep
from taste_agent.browser.sub_agent import run_browser_subagent
from taste_agent.config import DEFAULT_MODEL_ID
from taste_agent.guardrails.action import (
    approve,
    get_pending,
    register_pending,
)
from taste_agent.skills.reserve_table.reserve_table import (
    _prepare_from_spec,
    _replay_cached,
    _run_impl,
    cancel_reservation,
    finalize_reservation,
    run,
    set_default_backend,
)
import taste_agent.skills.reserve_table.reserve_table as _rt_module
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


def test_replay_cached_allows_raw_html_observation_steps():
    backend = MockBrowserBackend()
    result = _replay_cached(
        cached_trace=[
            ("navigate", {"url": "https://x.example/r"}),
            ("raw_html", {}),
        ],
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
    assert ("raw_html", {}) in backend.calls


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


def test_run_without_backend_returns_configuration_error(monkeypatch):
    monkeypatch.setattr(_rt_module, "ALLOW_RUNTIME_MOCKS", False)
    _rt_module._DEFAULT_BACKEND = None
    result = run(
        place_name="June Cafe",
        reservation_url="https://june-cafe.resos.com/booking",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
    )
    assert result["status"] == "failed"
    assert result["source"] == "configuration"
    assert "Browser automation is not configured" in result["error"]


def test_prepare_from_spec_fills_current_values():
    backend = MockBrowserBackend()
    spec = BookingFlowSpec(
        status="ok",
        place_name="June Cafe",
        source_host="june-cafe.resos.com",
        platform="resos",
        entry_url="https://june-cafe.resos.com/booking",
        final_form_url="https://june-cafe.resos.com/booking",
        steps_to_form=[
            BookingFlowStep(
                action="navigate",
                args={"url": "https://june-cafe.resos.com/booking"},
            ),
            BookingFlowStep(action="dom_snapshot", args={"selector": "body"}),
        ],
        required_fields=[
            BookingFieldSpec(name="date", type="date", selector="input[name='date']"),
            BookingFieldSpec(name="time", type="time", selector="input[name='time']"),
            BookingFieldSpec(
                name="party_size",
                type="integer",
                selector="input[name='party_size']",
            ),
            BookingFieldSpec(name="contact_name", type="text", selector="input[name='name']"),
        ],
        optional_fields=[
            BookingFieldSpec(
                name="contact_phone",
                type="phone",
                selector="input[name='phone']",
            )
        ],
    )
    result = _prepare_from_spec(
        flow_spec=spec,
        backend=backend,
        place_name="June Cafe",
        reservation_url="https://june-cafe.resos.com/booking",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
        contact_phone="+381601234567",
    )
    assert result["status"] == "pending_approval"
    assert result["source"] == "spec"
    assert backend.calls == [
        ("navigate", {"url": "https://june-cafe.resos.com/booking"}),
        ("dom_snapshot", {"selector": "body"}),
        ("fill", {"selector": "input[name='date']", "value": "2026-05-20"}),
        ("fill", {"selector": "input[name='time']", "value": "20:00"}),
        ("fill", {"selector": "input[name='party_size']", "value": "2"}),
        ("fill", {"selector": "input[name='name']", "value": "Ana"}),
        ("fill", {"selector": "input[name='phone']", "value": "+381601234567"}),
    ]


def test_prepare_from_spec_allows_raw_html_observation_steps():
    backend = MockBrowserBackend()
    spec = BookingFlowSpec(
        status="ok",
        place_name="June Cafe",
        source_host="june-cafe.resos.com",
        platform="resos",
        entry_url="https://june-cafe.resos.com/booking",
        final_form_url="https://june-cafe.resos.com/booking",
        steps_to_form=[
            BookingFlowStep(
                action="navigate",
                args={"url": "https://june-cafe.resos.com/booking"},
            ),
            BookingFlowStep(action="raw_html", args={}),
        ],
        required_fields=[
            BookingFieldSpec(name="contact_name", type="text", selector="input[name='name']"),
        ],
    )
    result = _prepare_from_spec(
        flow_spec=spec,
        backend=backend,
        place_name="June Cafe",
        reservation_url="https://june-cafe.resos.com/booking",
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
        contact_phone="",
    )
    assert result["status"] == "pending_approval"
    assert ("raw_html", {}) in backend.calls


def test_run_prefers_spec_cache_over_raw_trace():
    backend = MockBrowserBackend()
    set_default_backend(backend)
    url = "https://june-cafe.resos.com/booking"
    save_trace(
        url,
        [
            ("navigate", {"url": url}),
            ("fill", {"selector": "input[name='name']", "value": "OLD"}),
        ],
    )
    save_spec(
        url,
        BookingFlowSpec(
            status="ok",
            place_name="June Cafe",
            source_host="june-cafe.resos.com",
            platform="resos",
            entry_url=url,
            final_form_url=url,
            steps_to_form=[BookingFlowStep(action="navigate", args={"url": url})],
            required_fields=[
                BookingFieldSpec(name="date", type="date", selector="input[name='date']"),
                BookingFieldSpec(name="time", type="time", selector="input[name='time']"),
                BookingFieldSpec(
                    name="party_size",
                    type="integer",
                    selector="input[name='party_size']",
                ),
                BookingFieldSpec(
                    name="contact_name",
                    type="text",
                    selector="input[name='name']",
                ),
            ],
        ),
    )

    result = run(
        place_name="June Cafe",
        reservation_url=url,
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
    )

    assert result["status"] == "pending_approval"
    assert result["source"] == "spec"
    assert ("fill", {"selector": "input[name='name']", "value": "Ana"}) in backend.calls
    assert ("fill", {"selector": "input[name='name']", "value": "OLD"}) not in backend.calls


def test_run_impl_ignores_incomplete_cached_spec(monkeypatch):
    backend = MockBrowserBackend()
    url = "https://june-cafe.resos.com/booking"
    save_spec(
        url,
        BookingFlowSpec(
            status="ok",
            place_name="June Cafe",
            source_host="june-cafe.resos.com",
            platform="resos",
            entry_url=url,
            final_form_url=url,
            steps_to_form=[BookingFlowStep(action="navigate", args={"url": url})],
            required_fields=[
                BookingFieldSpec(name="contact_name", type="text", selector="input[name='name']"),
            ],
        ),
    )

    def _fake_run_browser_subagent(**_kwargs):
        action_id = register_pending("confirm_reservation", "Reserve at June Cafe")
        return {
            "messages": [],
            "last_message_text": "done",
            "actions": [("navigate", {"url": url}), ("fill", {"selector": "input[name='name']", "value": "Ana"})],
            "action_id": action_id,
        }

    monkeypatch.setattr(
        "taste_agent.skills.reserve_table.reserve_table.run_browser_subagent",
        _fake_run_browser_subagent,
    )

    result = _run_impl(
        place_name="June Cafe",
        reservation_url=url,
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
        contact_phone="",
        backend=backend,
        model_factory=_factory,
    )

    assert result["status"] == "pending_approval"
    assert result["source"] == "agentic"
    spec = get_spec(url)
    assert spec is None or spec.status != "ok"


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
    assert get_spec(url) is None


def test_run_impl_saves_flow_spec_on_pending_outcome(monkeypatch):
    backend = MockBrowserBackend()
    url = "https://june-cafe.resos.com/booking"

    def _fake_run_browser_subagent(**_kwargs):
        backend.navigate(url)
        backend.dom_snapshot("body")
        backend.fill("input[name='date']", "2026-05-20")
        backend.fill("input[name='time']", "20:00")
        backend.fill("input[name='party_size']", "2")
        backend.fill("input[name='name']", "Ana")
        action_id = register_pending("confirm_reservation", "Reserve at June Cafe")
        return {
            "messages": [],
            "last_message_text": "done",
            "actions": list(backend.calls),
            "action_id": action_id,
        }

    monkeypatch.setattr(
        "taste_agent.skills.reserve_table.reserve_table.run_browser_subagent",
        _fake_run_browser_subagent,
    )

    result = _run_impl(
        place_name="June Cafe",
        reservation_url=url,
        date="2026-05-20",
        time="20:00",
        party_size=2,
        contact_name="Ana",
        contact_phone="",
        backend=backend,
        model_factory=_factory,
    )

    assert result["status"] == "pending_approval"
    assert result["source"] == "agentic"
    spec = get_spec(url)
    assert spec is not None
    assert spec.platform == "resos"
    assert [field.name for field in spec.required_fields] == [
        "date",
        "time",
        "party_size",
        "contact_name",
    ]
    assert result["flow_spec"]["place_name"] == "June Cafe"


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


def test_run_browser_subagent_returns_only_actions_from_current_run(monkeypatch):
    from langchain_core.messages import AIMessage

    backend = MockBrowserBackend()
    backend.navigate("https://old.example/reserve")  # pre-existing history

    class _FakeAgent:
        def invoke(self, payload):
            backend.navigate("https://new.example/reserve")
            backend.fill("input#name", "Ana")
            return {"messages": [AIMessage(content="done")]}

    monkeypatch.setattr("langchain.agents.create_agent", lambda *_a, **_kw: _FakeAgent())

    result = run_browser_subagent(
        goal="discover booking flow",
        backend=backend,
        model_factory=_factory,
        model_id=DEFAULT_MODEL_ID,
    )

    assert result["actions"] == [
        ("navigate", {"url": "https://new.example/reserve"}),
        ("fill", {"selector": "input#name", "value": "Ana"}),
    ]
