"""Browser tools — atomic actions exposed to the sub-agent.

A factory function `build_browser_tools(backend)` returns a list of
LangChain StructuredTools bound to a specific backend. Each tool corresponds
1:1 with a BrowserBackend verb; the agent's tool call IS one action in the
JSON-DSL we teach.

Why a factory and not module-level @tool: the backend is per-session (per
reservation flow). Module-level tools would force a singleton backend.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool, tool

from taste_agent.browser.backend import BrowserBackend
from taste_agent.guardrails.action import register_pending
from taste_agent.logging_ import get_logger, trace

logger = get_logger(__name__)


def build_browser_tools(backend: BrowserBackend) -> list[StructuredTool]:
    """Return the tool list to give a browser-driving sub-agent."""

    @tool
    def browser_navigate(url: str) -> str:
        """Open the given URL in the browser. Returns a confirmation string."""
        with trace("tool:browser_navigate", url=url):
            backend.navigate(url)
            return f"navigated to {url}"

    @tool
    def browser_click(selector: str) -> str:
        """Click the element matching the CSS selector."""
        with trace("tool:browser_click", selector=selector):
            backend.click(selector)
            return f"clicked {selector}"

    @tool
    def browser_fill(selector: str, value: str) -> str:
        """Type ``value`` into the form input matching the CSS selector."""
        with trace("tool:browser_fill", selector=selector):
            backend.fill(selector, value)
            return f"filled {selector} with {value!r}"

    @tool
    def browser_wait_for(selector: str, timeout_ms: int = 5000) -> str:
        """Wait until the element matching the selector appears (default 5s)."""
        with trace("tool:browser_wait_for", selector=selector):
            backend.wait_for(selector, timeout_ms=timeout_ms)
            return f"selector {selector} is present"

    @tool
    def browser_dom_snapshot(selector: str = "body") -> str:
        """Return a simplified DOM snippet for the given selector (default body).

        The agent uses this to see what's currently on the page before deciding
        the next action.
        """
        with trace("tool:browser_dom_snapshot", selector=selector):
            return backend.dom_snapshot(selector)

    return [
        browser_navigate,
        browser_click,
        browser_fill,
        browser_wait_for,
        browser_dom_snapshot,
    ]


def make_request_approval_tool() -> StructuredTool:
    """Return the ``request_user_approval`` tool.

    The tool registers a pending action with the action guardrail and returns
    the ``action_id``. The sub-agent should call this when the reservation
    form is fully filled and the only remaining step is the final submit.
    """

    @tool
    def request_user_approval(summary: str) -> str:
        """Request user approval for an irreversible action.

        Args:
            summary: short human-readable description of what is about to happen.
                Example: "Reserve table at Iva for 2026-05-20 20:00, party of 2,
                name: Nikolai S."

        Returns:
            A string of the form ``approval_pending:<action_id>``. The action
            will not execute until the user explicitly approves via the chat.
        """
        with trace("tool:request_user_approval"):
            action_id = register_pending(tool_name="confirm_reservation", summary=summary)
            return f"approval_pending:{action_id}"

    return request_user_approval
