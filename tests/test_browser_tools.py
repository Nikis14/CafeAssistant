"""Tests for the browser tools factory + the request_user_approval tool."""

from __future__ import annotations

from taste_agent.browser.backend import MockBrowserBackend
from taste_agent.browser.tools import build_browser_tools, make_request_approval_tool
from taste_agent.guardrails.action import get_pending


def _by_name(tools, name):
    return next(t for t in tools if t.name == name)


def test_build_browser_tools_returns_five_tools():
    tools = build_browser_tools(MockBrowserBackend())
    names = {t.name for t in tools}
    assert names == {
        "browser_navigate",
        "browser_click",
        "browser_fill",
        "browser_wait_for",
        "browser_dom_snapshot",
    }


def test_browser_navigate_tool_drives_backend():
    backend = MockBrowserBackend()
    tools = build_browser_tools(backend)
    nav = _by_name(tools, "browser_navigate")
    nav.invoke({"url": "https://x.example/reserve"})
    assert backend.calls == [("navigate", {"url": "https://x.example/reserve"})]


def test_browser_click_tool_drives_backend():
    backend = MockBrowserBackend()
    tools = build_browser_tools(backend)
    _by_name(tools, "browser_click").invoke({"selector": "button.book"})
    assert backend.calls == [("click", {"selector": "button.book"})]


def test_browser_fill_tool_drives_backend():
    backend = MockBrowserBackend()
    tools = build_browser_tools(backend)
    _by_name(tools, "browser_fill").invoke({"selector": "input#name", "value": "Ana"})
    assert backend.calls == [("fill", {"selector": "input#name", "value": "Ana"})]


def test_browser_dom_snapshot_tool_returns_dom():
    backend = MockBrowserBackend()
    backend.set_dom("https://x.example/r", "<form>booking</form>")
    backend.navigate("https://x.example/r")
    tools = build_browser_tools(backend)
    dom = _by_name(tools, "browser_dom_snapshot").invoke({"selector": "body"})
    assert dom == "<form>booking</form>"


def test_request_user_approval_tool_registers_pending():
    tool = make_request_approval_tool()
    result = tool.invoke({"summary": "Reserve table at Iva, 2 people, May 20 20:00"})
    assert result.startswith("approval_pending:")
    action_id = result.split(":", 1)[1]
    pending = get_pending()
    assert pending is not None
    assert pending.action_id == action_id
    assert pending.summary == "Reserve table at Iva, 2 people, May 20 20:00"


def test_tools_share_one_backend_instance():
    # All tools built by one call to the factory point at the same backend.
    backend = MockBrowserBackend()
    tools = build_browser_tools(backend)
    _by_name(tools, "browser_navigate").invoke({"url": "https://x.example/r"})
    _by_name(tools, "browser_click").invoke({"selector": ".x"})
    assert len(backend.calls) == 2


def test_browser_click_tool_refuses_forbidden_selector():
    """The reviewer's key concern: even if the sub-agent's LLM emits a tool
    call attempting to click the submit selector, the backend refuses. This
    test exercises the tool surface — the path a misbehaving sub-agent would
    take."""
    import pytest

    backend = MockBrowserBackend()
    backend.forbidden_selectors.add("button.confirm-reservation")
    tools = build_browser_tools(backend)
    click_tool = _by_name(tools, "browser_click")

    # The tool wraps backend.click; the PermissionError bubbles up.
    with pytest.raises(PermissionError, match="forbidden"):
        click_tool.invoke({"selector": "button.confirm-reservation"})

    # No record of the forbidden click — important so the cached parser trace
    # never contains a forbidden action.
    assert backend.calls == []
