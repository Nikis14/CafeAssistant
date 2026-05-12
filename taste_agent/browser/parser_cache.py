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


def save_trace(url: str, trace: ActionTrace) -> None:
    host = host_of(url)
    _CACHE[host] = list(trace)
    logger.info("cached parser trace for host=%s (%d actions)", host, len(trace))


def get_trace(url: str) -> ActionTrace | None:
    return _CACHE.get(host_of(url))


def has_trace(url: str) -> bool:
    return host_of(url) in _CACHE


def clear_cache() -> None:
    """Test-only."""
    _CACHE.clear()
