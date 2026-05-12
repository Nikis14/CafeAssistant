"""Tests for the MockBrowserBackend."""

from __future__ import annotations

from taste_agent.browser.backend import MockBrowserBackend


def test_navigate_records_call_and_sets_url():
    b = MockBrowserBackend()
    b.navigate("https://example.com/reserve")
    assert b.current_url() == "https://example.com/reserve"
    assert b.calls == [("navigate", {"url": "https://example.com/reserve"})]


def test_click_records_call():
    b = MockBrowserBackend()
    b.click("button.submit")
    assert b.calls == [("click", {"selector": "button.submit"})]


def test_fill_records_selector_and_value():
    b = MockBrowserBackend()
    b.fill("input#name", "Ana")
    assert b.calls == [("fill", {"selector": "input#name", "value": "Ana"})]


def test_wait_for_records_call_with_default_timeout():
    b = MockBrowserBackend()
    b.wait_for("form")
    assert b.calls == [("wait_for", {"selector": "form", "timeout_ms": 5000})]


def test_screenshot_returns_bytes():
    b = MockBrowserBackend()
    result = b.screenshot()
    assert isinstance(result, bytes)
    assert b.calls == [("screenshot", {})]


def test_dom_snapshot_returns_default_when_no_dom_set():
    b = MockBrowserBackend()
    b.navigate("https://x.example/")
    dom = b.dom_snapshot()
    assert "<empty" in dom


def test_dom_snapshot_returns_programmed_dom_for_url():
    b = MockBrowserBackend()
    b.set_dom("https://x.example/r", "<form>fixed</form>")
    b.navigate("https://x.example/r")
    assert b.dom_snapshot() == "<form>fixed</form>"


def test_expect_dom_one_shot_overrides_then_reverts():
    b = MockBrowserBackend()
    b.set_dom("https://x.example/", "<base>")
    b.navigate("https://x.example/")
    b.expect_dom("<after-click>")
    assert b.dom_snapshot() == "<after-click>"
    # Next call reverts to the URL default
    assert b.dom_snapshot() == "<base>"


def test_calls_record_in_order():
    b = MockBrowserBackend()
    b.navigate("https://x.example/")
    b.fill("input#a", "1")
    b.click("button#submit")
    names = [c[0] for c in b.calls]
    assert names == ["navigate", "fill", "click"]


# ── Defense in depth: forbidden_selectors blocks irreversible clicks ─────────


def test_forbidden_selectors_starts_empty():
    b = MockBrowserBackend()
    assert b.forbidden_selectors == set()


def test_click_raises_permission_error_for_forbidden_selector():
    import pytest

    b = MockBrowserBackend()
    b.forbidden_selectors.add("button.confirm-reservation")
    with pytest.raises(PermissionError, match="forbidden"):
        b.click("button.confirm-reservation")


def test_forbidden_click_does_not_record_call():
    """If the click is refused, the call is *not* added to .calls — otherwise
    a sub-agent could 'launder' a forbidden action by calling it."""
    import pytest

    b = MockBrowserBackend()
    b.forbidden_selectors.add("button.x")
    with pytest.raises(PermissionError):
        b.click("button.x")
    assert b.calls == []


def test_non_forbidden_clicks_still_work_when_set_is_populated():
    b = MockBrowserBackend()
    b.forbidden_selectors.add("button.x")
    b.click("button.different")  # should not raise
    assert b.calls == [("click", {"selector": "button.different"})]
