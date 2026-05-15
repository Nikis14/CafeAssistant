"""Browser tools — atomic actions exposed to the sub-agent."""

from __future__ import annotations

from langchain_core.tools import StructuredTool, tool

from taste_agent.browser.backend import BrowserBackend
from taste_agent.guardrails.action import register_pending
from taste_agent.logging_ import debug_enter, debug_exit, get_logger, trace

logger = get_logger(__name__)


def build_browser_tools(backend: BrowserBackend) -> list[StructuredTool]:
    """Return the tool list to give a browser-driving sub-agent."""

    @tool
    def browser_navigate(url: str) -> str:
        """Open the given URL in the browser. Returns a confirmation string."""
        debug_enter("browser_navigate", url=url)
        with trace("tool:browser_navigate", url=url):
            try:
                backend.navigate(url)
            except PermissionError as e:
                logger.info("browser_navigate blocked: %s", e)
                result = f"blocked navigation: {e}"
                debug_exit("browser_navigate", result=result)
                return result
            except Exception as e:
                result = f"could not navigate to {url}: {e}"
                debug_exit("browser_navigate", result=result)
                return result
            result = f"navigated to {url}"
            debug_exit("browser_navigate", result=result)
            return result

    @tool
    def browser_click(selector: str) -> str:
        """Click the element matching the CSS selector."""
        debug_enter("browser_click", selector=selector)
        with trace("tool:browser_click", selector=selector):
            try:
                backend.click(selector)
            except PermissionError:
                raise
            except Exception as e:
                result = f"could not click {selector}: {e}"
                debug_exit("browser_click", result=result)
                return result
            result = f"clicked {selector}"
            debug_exit("browser_click", result=result)
            return result

    @tool
    def browser_fill(selector: str, value: str) -> str:
        """Type ``value`` into the form input matching the CSS selector."""
        debug_enter("browser_fill", selector=selector, value=value)
        with trace("tool:browser_fill", selector=selector):
            try:
                backend.fill(selector, value)
            except Exception as e:
                result = f"could not fill {selector}: {e}"
                debug_exit("browser_fill", result=result)
                return result
            result = f"filled {selector} with {value!r}"
            debug_exit("browser_fill", result=result)
            return result

    @tool
    def browser_wait_for(selector: str, timeout_ms: int = 5000) -> str:
        """Wait until the element matching the selector appears (default 5s)."""
        debug_enter("browser_wait_for", selector=selector, timeout_ms=timeout_ms)
        with trace("tool:browser_wait_for", selector=selector):
            try:
                backend.wait_for(selector, timeout_ms=timeout_ms)
            except TimeoutError:
                result = f"selector {selector} did not appear within {timeout_ms}ms"
                debug_exit("browser_wait_for", result=result)
                return result
            except Exception as e:
                result = f"could not wait for {selector}: {e}"
                debug_exit("browser_wait_for", result=result)
                return result
            result = f"selector {selector} is present"
            debug_exit("browser_wait_for", result=result)
            return result

    @tool
    def browser_page_context() -> str:
        """Return the current rendered page as raw HTML."""
        debug_enter("browser_page_context")
        with trace("tool:browser_page_context"):
            try:
                result = backend.raw_html()
            except Exception as e:
                result = f"could not read page html: {e}"
            debug_exit("browser_page_context", result=result)
            return result

    return [
        browser_navigate,
        browser_click,
        browser_fill,
        browser_wait_for,
        browser_page_context,
    ]


def make_request_approval_tool() -> StructuredTool:
    """Return the ``request_user_approval`` tool.

    The tool registers a pending action with the action guardrail and returns
    the ``action_id``. The sub-agent should call this when the reservation
    form is fully filled and the only remaining step is the final submit.
    """

    @tool
    def request_user_approval(
        summary: str,
        submit_selector: str = "",
        place_name: str = "",
        reservation_url: str = "",
    ) -> str:
        """Request user approval for an irreversible action.

        Args:
            summary: short human-readable description of what is about to happen.
                Example: "Reserve table at Iva for 2026-05-20 20:00, party of 2,
                name: Nikolai S."
            submit_selector: selector for the final irreversible submit control.
                Pass this when the final button is known so finalization can
                click the exact control that was discovered.
            place_name: human-readable place name for recovery flows.
            reservation_url: current booking URL for recovery flows.

        Returns:
            A string of the form ``approval_pending:<action_id>``. The action
            will not execute until the user explicitly approves via the chat.
        """
        debug_enter(
            "request_user_approval",
            summary=summary,
            submit_selector=submit_selector,
            place_name=place_name,
            reservation_url=reservation_url,
        )
        with trace("tool:request_user_approval"):
            args: dict[str, str] = {}
            if submit_selector:
                args["submit_selector"] = submit_selector
            if place_name:
                args["place_name"] = place_name
            if reservation_url:
                args["reservation_url"] = reservation_url
            action_id = register_pending(
                tool_name="confirm_reservation",
                summary=summary,
                args=args,
            )
            result = f"approval_pending:{action_id}"
            debug_exit("request_user_approval", result=result)
            return result

    return request_user_approval
