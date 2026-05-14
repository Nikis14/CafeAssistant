"""In-memory cache of discovered booking-flow specs, keyed by host."""

from __future__ import annotations

from taste_agent.browser.parser_cache import host_of
from taste_agent.browser.specs import BookingFlowSpec
from taste_agent.logging_ import get_logger

logger = get_logger(__name__)

_CACHE: dict[str, BookingFlowSpec] = {}


def save_spec(url: str, spec: BookingFlowSpec) -> None:
    host = host_of(url)
    _CACHE[host] = spec
    logger.info(
        "cached booking flow spec for host=%s (%d required fields)",
        host,
        len(spec.required_fields),
    )


def get_spec(url: str) -> BookingFlowSpec | None:
    return _CACHE.get(host_of(url))


def delete_spec(url: str) -> None:
    _CACHE.pop(host_of(url), None)


def has_spec(url: str) -> bool:
    return host_of(url) in _CACHE


def clear_spec_cache() -> None:
    """Test-only."""
    _CACHE.clear()
