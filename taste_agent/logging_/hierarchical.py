"""Hierarchical (tree-shaped) console logger.

Use `trace("name", **extra)` as a context manager around each LangGraph node,
skill entry function, and non-trivial tool. Logs emitted inside the block are
indented one level deeper, producing a console tree alongside LangSmith traces.
"""

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_DEFAULT_LOGGER_NAME = "taste_agent"
_DEBUG_ENV_VAR = "TASTE_AGENT_DEBUG"

# Tree-drawing characters (ASCII; CLAUDE.md forbids emoji)
_INDENT = "   "
_BRANCH = "|- "

_depth: ContextVar[int] = ContextVar("trace_depth", default=0)


def make_prefix(depth: int) -> str:
    """Render the tree prefix for a log line at the given indentation depth."""
    if depth <= 0:
        return ""
    return _INDENT * (depth - 1) + _BRANCH


class HierarchicalFormatter(logging.Formatter):
    """Formatter that prepends a tree prefix based on the current trace depth."""

    def format(self, record: logging.LogRecord) -> str:
        depth = getattr(record, "trace_depth", _depth.get())
        prefix = make_prefix(depth)
        ts = self.formatTime(record, datefmt="%H:%M:%S")
        return f"[{ts}] {record.levelname:<7} {prefix}{record.getMessage()}"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger. With no name, returns the package root logger."""
    return logging.getLogger(name or _DEFAULT_LOGGER_NAME)


def _debug_enabled_from_env() -> bool:
    value = os.getenv(_DEBUG_ENV_VAR, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def configure_logging(level: int | None = None) -> None:
    """Wire the hierarchical formatter to stdout once. Idempotent."""
    root = logging.getLogger(_DEFAULT_LOGGER_NAME)
    if root.handlers:
        return
    if level is None:
        level = logging.DEBUG if _debug_enabled_from_env() else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(HierarchicalFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


@contextmanager
def trace(name: str, **extra: object) -> Iterator[logging.Logger]:
    """Log the entry and increase indentation depth for nested logs.

    Why: LangSmith captures the call tree, but we also want a readable
    indented trace in the console for live demos and local debugging.
    """
    logger = get_logger()
    extras_str = " ".join(f"{k}={v}" for k, v in extra.items())
    msg = f"{name} {extras_str}".rstrip()
    logger.info(msg)
    token = _depth.set(_depth.get() + 1)
    try:
        yield logger
    finally:
        _depth.reset(token)


def current_depth() -> int:
    """Return the current trace depth (useful for tests)."""
    return _depth.get()
