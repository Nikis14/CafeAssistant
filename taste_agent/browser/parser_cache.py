"""In-memory cache of successful reservation action traces, keyed by host.

After a sub-agent successfully fills a reservation form on a new site, the
action sequence is stored here. Subsequent reservations on the same host can
replay the cached sequence (the "scripted" mode of reserve_table) instead of
spawning a fresh LLM-driven sub-agent. The teaching moment is the duality:
first run discovers the parser, later runs reuse it.

Phase 2: in-memory only. Phase 4 persists to ``parsers_cache/<host>.json``.
"""

from __future__ import annotations

from urllib.parse import urlparse

from taste_agent.logging_ import get_logger

logger = get_logger(__name__)

# Action trace shape: list of (action_name, args_dict)
ActionTrace = list[tuple[str, dict[str, object]]]

_CACHE: dict[str, ActionTrace] = {}


def host_of(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url


def format_trace(trace: ActionTrace) -> str:
    """Render a compact multi-line action trace for logs/debug."""
    if not trace:
        return "(no actions)"

    lines: list[str] = []
    for i, (action_name, args) in enumerate(trace, start=1):
        rendered_args = ", ".join(f"{k}={v!r}" for k, v in sorted(args.items()))
        if rendered_args:
            lines.append(f"{i}. {action_name}({rendered_args})")
        else:
            lines.append(f"{i}. {action_name}()")
    return "\n".join(lines)


def save_trace(url: str, trace: ActionTrace) -> None:
    host = host_of(url)
    _CACHE[host] = list(trace)
    logger.info("cached parser trace for host=%s (%d actions)", host, len(trace))
    logger.info("parser trace steps for host=%s\n%s", host, format_trace(trace))


def get_trace(url: str) -> ActionTrace | None:
    return _CACHE.get(host_of(url))


def has_trace(url: str) -> bool:
    return host_of(url) in _CACHE


def clear_cache() -> None:
    """Test-only."""
    _CACHE.clear()
