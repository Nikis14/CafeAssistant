"""Browser backend abstraction.

The browser sub-agent talks to a `BrowserBackend`, never to Playwright
directly. Two implementations live behind the protocol:

- ``MockBrowserBackend``: programmable, deterministic, used in tests and as a
  scaffold for the seminar demo when Playwright is not installed.
- ``PlaywrightBrowserBackend``: real Playwright driver. Phase 4 enables it
  for the live demo on real reservation sites.

Keeping the protocol thin (5 verbs + 2 readers) makes it small enough to walk
through in a lecture slide and easy to mock for tests.
"""

from __future__ import annotations

from typing import Protocol


class BrowserBackend(Protocol):
    """Minimal surface a browser-driving agent needs.

    Each method represents one atomic action the agent can take. The JSON-DSL
    we teach in the seminar is simply the sequence of these method calls,
    serialized.

    ``forbidden_selectors`` is defense-in-depth for the action guardrail.
    Even if a sub-agent goes off-prompt and tries to click an irreversible
    target (e.g., a final submit button) before user approval, the backend
    itself refuses the click. Real Playwright implementations should enforce
    the same contract.
    """

    forbidden_selectors: set[str]

    def navigate(self, url: str) -> None: ...
    def click(self, selector: str) -> None: ...
    def fill(self, selector: str, value: str) -> None: ...
    def wait_for(self, selector: str, timeout_ms: int = 5000) -> None: ...
    def screenshot(self) -> bytes: ...
    def dom_snapshot(self, selector: str | None = None) -> str: ...
    def current_url(self) -> str: ...


class MockBrowserBackend:
    """In-memory browser that records every call.

    Test setup pattern:

        backend = MockBrowserBackend()
        backend.set_dom("https://x.example/reserve", "<form>...</form>")
        backend.expect_dom("<form ... ready for submit>")  # one-shot override
        # ... drive the agent ...
        assert backend.calls == [
            ("navigate", {"url": "https://x.example/reserve"}),
            ("fill", {"selector": "input#date", "value": "2026-05-20"}),
            ...
        ]
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.forbidden_selectors: set[str] = set()
        self._url: str = ""
        self._dom_by_url: dict[str, str] = {}
        self._next_dom: str | None = None

    # ── BrowserBackend surface ──

    def navigate(self, url: str) -> None:
        self.calls.append(("navigate", {"url": url}))
        self._url = url

    def click(self, selector: str) -> None:
        # Defense in depth: the action guardrail registers irreversible
        # selectors here. The check is deterministic — no LLM in the path.
        if selector in self.forbidden_selectors:
            raise PermissionError(
                f"Clicking {selector!r} is forbidden in this session — "
                "irreversible action requires user approval via the action guardrail."
            )
        self.calls.append(("click", {"selector": selector}))

    def fill(self, selector: str, value: str) -> None:
        self.calls.append(("fill", {"selector": selector, "value": value}))

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> None:
        self.calls.append(("wait_for", {"selector": selector, "timeout_ms": timeout_ms}))

    def screenshot(self) -> bytes:
        self.calls.append(("screenshot", {}))
        return b"<fake-png>"

    def dom_snapshot(self, selector: str | None = None) -> str:
        self.calls.append(("dom_snapshot", {"selector": selector}))
        if self._next_dom is not None:
            result = self._next_dom
            self._next_dom = None
            return result
        return self._dom_by_url.get(self._url, "<empty></empty>")

    def current_url(self) -> str:
        return self._url

    # ── Test helpers ──

    def set_dom(self, url: str, dom: str) -> None:
        """Pre-program the DOM the backend will return when at ``url``."""
        self._dom_by_url[url] = dom

    def expect_dom(self, dom: str) -> None:
        """Return ``dom`` from the *next* dom_snapshot call only, then revert."""
        self._next_dom = dom
