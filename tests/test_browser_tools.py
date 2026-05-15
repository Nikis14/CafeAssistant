"""Tests for the browser tools factory + the request_user_approval tool."""

from __future__ import annotations

from taste_agent.browser.backend import MockBrowserBackend
from taste_agent.browser.tools import (
    build_browser_tools,
    make_request_approval_tool,
)
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
        "browser_page_context",
    }


def test_browser_navigate_tool_drives_backend():
    backend = MockBrowserBackend()
    tools = build_browser_tools(backend)
    nav = _by_name(tools, "browser_navigate")
    nav.invoke({"url": "https://x.example/reserve"})
    assert backend.calls == [("navigate", {"url": "https://x.example/reserve"})]


def test_browser_navigate_tool_returns_block_message_on_permission_error():
    class _BlockedBackend(MockBrowserBackend):
        def navigate(self, url: str) -> None:
            raise PermissionError("blocked cross-host jump")

    backend = _BlockedBackend()
    tools = build_browser_tools(backend)
    nav = _by_name(tools, "browser_navigate")
    result = nav.invoke({"url": "https://blocked.example/"})
    assert "blocked navigation" in result
    assert backend.calls == []


def test_browser_click_tool_drives_backend():
    backend = MockBrowserBackend()
    tools = build_browser_tools(backend)
    _by_name(tools, "browser_click").invoke({"selector": "button.book"})
    assert backend.calls == [("click", {"selector": "button.book"})]


def test_browser_click_tool_returns_timeout_message_on_hidden_selector():
    class _TimeoutBackend(MockBrowserBackend):
        def click(self, selector: str) -> None:
            raise TimeoutError("selector matched 1 element, but none were visible")

    backend = _TimeoutBackend()
    tools = build_browser_tools(backend)
    result = _by_name(tools, "browser_click").invoke({"selector": "div[data-test='date']"})
    assert "could not click" in result
    assert "none were visible" in result


def test_browser_click_tool_returns_error_message_on_invalid_selector():
    class _InvalidSelectorBackend(MockBrowserBackend):
        def click(self, selector: str) -> None:
            raise ValueError("not a valid selector")

    backend = _InvalidSelectorBackend()
    tools = build_browser_tools(backend)
    result = _by_name(tools, "browser_click").invoke({"selector": "button:contains('2')"})
    assert "could not click" in result
    assert "not a valid selector" in result


def test_browser_fill_tool_drives_backend():
    backend = MockBrowserBackend()
    tools = build_browser_tools(backend)
    _by_name(tools, "browser_fill").invoke({"selector": "input#name", "value": "Ana"})
    assert backend.calls == [("fill", {"selector": "input#name", "value": "Ana"})]


def test_browser_fill_tool_returns_timeout_message_on_hidden_selector():
    class _TimeoutBackend(MockBrowserBackend):
        def fill(self, selector: str, value: str) -> None:
            raise TimeoutError("selector matched 1 element, but none were visible")

    backend = _TimeoutBackend()
    tools = build_browser_tools(backend)
    result = _by_name(tools, "browser_fill").invoke({"selector": "input[name=name]", "value": "Ana"})
    assert "could not fill" in result
    assert "none were visible" in result


def test_browser_wait_for_tool_returns_error_message_on_invalid_selector():
    class _InvalidSelectorBackend(MockBrowserBackend):
        def wait_for(self, selector: str, timeout_ms: int = 5000) -> None:
            raise ValueError("not a valid selector")

    backend = _InvalidSelectorBackend()
    tools = build_browser_tools(backend)
    result = _by_name(tools, "browser_wait_for").invoke({"selector": "button:contains('2')"})
    assert "could not wait for" in result
    assert "not a valid selector" in result


def test_browser_page_context_returns_error_message_when_raw_html_fails():
    class _BrokenBackend(MockBrowserBackend):
        def raw_html(self) -> str:
            raise RuntimeError("page crashed")

    backend = _BrokenBackend()
    tools = build_browser_tools(backend)
    result = _by_name(tools, "browser_page_context").invoke({})
    assert result == "could not read page html: page crashed"


def test_browser_page_context_tool_returns_raw_html():
    backend = MockBrowserBackend()
    html = "<html><body><h1>June Cafe</h1><a href='/booking'>Reserve here</a></body></html>"
    backend.set_dom(
        "https://x.example/r",
        html,
    )
    backend.navigate("https://x.example/r")
    tools = build_browser_tools(backend)
    payload = _by_name(tools, "browser_page_context").invoke({})
    assert payload == html
    assert ("raw_html", {}) in backend.calls


def test_browser_wait_for_tool_returns_timeout_message_on_missing_selector():
    class _TimeoutBackend(MockBrowserBackend):
        def wait_for(self, selector: str, timeout_ms: int = 5000) -> None:
            raise TimeoutError("not found")

    backend = _TimeoutBackend()
    tools = build_browser_tools(backend)
    result = _by_name(tools, "browser_wait_for").invoke({"selector": "iframe", "timeout_ms": 1234})
    assert result == "selector iframe did not appear within 1234ms"


def test_request_user_approval_tool_registers_pending():
    tool = make_request_approval_tool()
    result = tool.invoke({"summary": "Reserve table at Iva, 2 people, May 20 20:00"})
    assert result.startswith("approval_pending:")
    action_id = result.split(":", 1)[1]
    pending = get_pending()
    assert pending is not None
    assert pending.action_id == action_id
    assert pending.summary == "Reserve table at Iva, 2 people, May 20 20:00"


def test_request_user_approval_tool_stores_submit_selector_when_provided():
    tool = make_request_approval_tool()
    result = tool.invoke(
        {
            "summary": "Reserve table at Iva, 2 people, May 20 20:00",
            "submit_selector": "button[type='submit']",
        }
    )
    assert result.startswith("approval_pending:")
    pending = get_pending()
    assert pending is not None
    assert pending.args["submit_selector"] == "button[type='submit']"


def test_request_user_approval_tool_stores_recovery_metadata():
    tool = make_request_approval_tool()
    tool.invoke(
        {
            "summary": "Reserve table at Iva, 2 people, May 20 20:00",
            "submit_selector": "button[type='submit']",
            "place_name": "Iva",
            "reservation_url": "https://iva.example/booking",
        }
    )
    pending = get_pending()
    assert pending is not None
    assert pending.args["place_name"] == "Iva"
    assert pending.args["reservation_url"] == "https://iva.example/booking"


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
