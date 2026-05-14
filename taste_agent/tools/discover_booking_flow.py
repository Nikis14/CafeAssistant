"""Tool wrapper for booking-flow discovery."""

from __future__ import annotations

from langchain_core.tools import tool

from taste_agent.logging_ import debug_enter, debug_exit
from taste_agent.skills.reserve_table.reserve_table import discover_booking_flow as _discover


@tool
def discover_booking_flow(place_name: str, reservation_url: str) -> dict[str, object]:
    """Discover how a site's booking flow works before collecting user values.

    Use this after you have a grounded candidate page URL for a specific place
    but before asking the user for booking details. The URL can be a booking
    page, official site, menu page, or another strong entry point discovered
    from search results. The tool explores the page, infers which fields are
    required, and caches a reusable booking flow spec.
    """
    debug_enter(
        "discover_booking_flow",
        place_name=place_name,
        reservation_url=reservation_url,
    )
    result = _discover(place_name=place_name, reservation_url=reservation_url)
    debug_exit("discover_booking_flow", result=result)
    return result
