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

from concurrent.futures import ThreadPoolExecutor
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
    def raw_html(self) -> str: ...
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

    def raw_html(self) -> str:
        self.calls.append(("raw_html", {}))
        if self._next_dom is not None:
            result = self._next_dom
            self._next_dom = None
            return result
        return self._dom_by_url.get(self._url, "<empty></empty>")

    # ── Test helpers ──

    def set_dom(self, url: str, dom: str) -> None:
        """Pre-program the DOM the backend will return when at ``url``."""
        self._dom_by_url[url] = dom

    def expect_dom(self, dom: str) -> None:
        """Return ``dom`` from the *next* dom_snapshot call only, then revert."""
        self._next_dom = dom


class PlaywrightBrowserBackend:
    """Thin synchronous Playwright-backed browser backend for production use."""

    def __init__(self, *, headless: bool = True) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.forbidden_selectors: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="taste-browser")
        self._headless = headless
        self._executor.submit(self._init_playwright).result()

    def _init_playwright(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Playwright is not installed. Install the 'playwright' package and browser binaries."
            ) from e

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page()

    def _run(self, fn, *args):
        return self._executor.submit(fn, *args).result()

    def _settle_page(self) -> None:
        self._page.wait_for_load_state("load", timeout=5000)
        try:
            self._page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            # Some sites keep background requests open; best-effort settle only.
            pass

    def _first_visible_locator(self, selector: str):
        locator = self._page.locator(selector)
        count = locator.count()
        if count == 0:
            raise TimeoutError(f"no elements matched selector {selector!r}")
        for idx in range(count):
            candidate = locator.nth(idx)
            if candidate.is_visible():
                return candidate
        raise TimeoutError(
            f"selector {selector!r} matched {count} element(s), but none were visible"
        )

    def _click_impl(self, selector: str) -> None:
        self._first_visible_locator(selector).click()

    def _fill_impl(self, selector: str, value: str) -> None:
        target = self._first_visible_locator(selector)
        target.wait_for(state="visible", timeout=5000)
        target.fill(value)

    def _wait_for_impl(self, selector: str, timeout_ms: int) -> None:
        self._first_visible_locator(selector).wait_for(timeout=timeout_ms)

    def _dom_snapshot_impl(self, selector: str) -> str:
        return self._first_visible_locator(selector).inner_html()

    def navigate(self, url: str) -> None:
        self.calls.append(("navigate", {"url": url}))
        self._run(self._navigate_impl, url)

    def _navigate_impl(self, url: str) -> None:
        self._page.goto(url, wait_until="domcontentloaded")
        self._settle_page()

    def click(self, selector: str) -> None:
        if selector in self.forbidden_selectors:
            raise PermissionError(
                f"Clicking {selector!r} is forbidden in this session — "
                "irreversible action requires user approval via the action guardrail."
            )
        self.calls.append(("click", {"selector": selector}))
        self._run(self._click_impl, selector)

    def fill(self, selector: str, value: str) -> None:
        self.calls.append(("fill", {"selector": selector, "value": value}))
        self._run(self._fill_impl, selector, value)

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> None:
        self.calls.append(("wait_for", {"selector": selector, "timeout_ms": timeout_ms}))
        self._run(self._wait_for_impl, selector, timeout_ms)

    def screenshot(self) -> bytes:
        self.calls.append(("screenshot", {}))
        return self._run(self._page.screenshot)

    def dom_snapshot(self, selector: str | None = None) -> str:
        self.calls.append(("dom_snapshot", {"selector": selector}))
        target = selector or "body"
        return self._run(self._dom_snapshot_impl, target)

    def raw_html(self) -> str:
        self.calls.append(("raw_html", {}))
        return self._run(self._raw_html_impl)

    def _raw_html_impl(self) -> str:
        self._settle_page()
        return self._page.content()

    def current_url(self) -> str:
        return self._run(lambda: self._page.url)

    def close(self) -> None:
        self._run(self._page.close)
        self._run(self._browser.close)
        self._run(self._pw.stop)
        self._executor.shutdown(wait=True)
