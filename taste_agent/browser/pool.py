"""Bounded pool of independent Playwright backends.

Playwright's sync API uses a per-thread greenlet dispatcher, so a single
``Browser``/``Page`` can only be driven from the thread that created it.
Parallel booking flows therefore require multiple independent backends, each
with its own Chromium process and its own executor thread. This pool owns N
of them and hands them out per flow; cookie/state reset happens automatically
on release.

Two borrow styles:

- ``acquire()`` context manager for one-shot use (e.g. booking-flow discovery
  that completes within a single function call).
- ``checkout()`` / ``checkin()`` for multi-turn flows where a backend must
  outlive one call (e.g. ``reserve_table.run`` fills the form, then a later
  ``finalize_reservation`` clicks submit on the same prepared page).
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterator

from taste_agent.browser.backend import PlaywrightBrowserBackend
from taste_agent.logging_ import get_logger

logger = get_logger(__name__)


class BrowserBackendPool:
    """Bounded pool of independent Playwright backends."""

    def __init__(self, size: int, *, headless: bool = True) -> None:
        if size < 1:
            raise ValueError(f"pool size must be >= 1, got {size}")
        self._free: queue.Queue[PlaywrightBrowserBackend] = queue.Queue(maxsize=size)
        self._all: list[PlaywrightBrowserBackend] = []
        self._closed = False
        self._lock = threading.Lock()
        try:
            for slot_id in range(size):
                backend = PlaywrightBrowserBackend(headless=headless)
                backend.slot_id = slot_id
                self._all.append(backend)
                self._free.put_nowait(backend)
        except Exception:
            # Roll back any partially-initialized backends so we don't leak
            # Chromium processes if construction fails halfway through.
            for backend in self._all:
                with contextlib.suppress(Exception):
                    backend.close()
            self._all.clear()
            raise
        logger.info("browser backend pool initialized with %d slot(s)", size)

    @property
    def size(self) -> int:
        return len(self._all)

    def checkout(self, timeout: float | None = None) -> PlaywrightBrowserBackend:
        """Borrow a backend. Caller is responsible for calling ``checkin``."""
        if self._closed:
            raise RuntimeError("pool is closed")
        backend = self._free.get(timeout=timeout)
        logger.debug("pool: checked out slot %s", backend.slot_id)
        return backend

    def checkin(self, backend: PlaywrightBrowserBackend, *, reset: bool = True) -> None:
        """Return a previously checked-out backend to the pool.

        ``reset=True`` (default) disposes the backend's BrowserContext so the
        next borrower starts with a clean cookie jar and localStorage.
        """
        if self._closed:
            with contextlib.suppress(Exception):
                backend.close()
            return
        if reset:
            try:
                backend.reset_context()
            except Exception as e:
                logger.warning(
                    "pool: reset_context failed on slot %s: %s; closing and dropping",
                    backend.slot_id,
                    e,
                )
                with contextlib.suppress(Exception):
                    backend.close()
                return
        self._free.put(backend)
        logger.debug("pool: checked in slot %s", backend.slot_id)

    @contextlib.contextmanager
    def acquire(self, timeout: float | None = None) -> Iterator[PlaywrightBrowserBackend]:
        """One-shot borrow with automatic checkin on exit."""
        backend = self.checkout(timeout=timeout)
        try:
            yield backend
        finally:
            self.checkin(backend)

    def close(self) -> None:
        """Tear down every backend in the pool. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for backend in self._all:
            with contextlib.suppress(Exception):
                backend.close()


# ── Module-level singleton ──────────────────────────────────────────────────

_BROWSER_POOL: BrowserBackendPool | None = None


def init_browser_pool(size: int, *, headless: bool = True) -> BrowserBackendPool:
    """Initialize the process-wide pool. Closes any existing pool first."""
    global _BROWSER_POOL
    if _BROWSER_POOL is not None:
        _BROWSER_POOL.close()
    _BROWSER_POOL = BrowserBackendPool(size=size, headless=headless)
    return _BROWSER_POOL


def get_browser_pool() -> BrowserBackendPool | None:
    """Return the current pool, or None if it hasn't been initialized."""
    return _BROWSER_POOL


def close_browser_pool() -> None:
    """Close and clear the process-wide pool. Useful for tests."""
    global _BROWSER_POOL
    if _BROWSER_POOL is not None:
        _BROWSER_POOL.close()
        _BROWSER_POOL = None
