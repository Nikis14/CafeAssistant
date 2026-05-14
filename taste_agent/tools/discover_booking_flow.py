"""Tool wrapper for booking-flow discovery."""

from __future__ import annotations

from langchain_core.tools import tool

from taste_agent.skills.reserve_table.reserve_table import discover_booking_flow as _discover


@tool
def discover_booking_flow(place_name: str, reservation_url: str) -> dict[str, object]:
    """Discover how a site's booking flow works before collecting user values.

    Use this after you have a grounded reservation page URL but before asking
    the user for booking details. It explores the page, infers which fields
    are required, and caches a reusable booking flow spec.
    """
    return _discover(place_name=place_name, reservation_url=reservation_url)
