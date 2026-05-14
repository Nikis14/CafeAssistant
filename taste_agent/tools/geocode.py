"""geocode tool — Phase 1 mock.

Phase 4 swaps this for OSM Nominatim (free, no key) or Foursquare's geocoding.
"""

from __future__ import annotations

from langchain_core.tools import tool

from taste_agent.config import ALLOW_RUNTIME_MOCKS
from taste_agent.logging_ import debug_enter, debug_exit, get_logger, trace

logger = get_logger(__name__)

_MOCK_PLACES: dict[str, dict[str, float | str]] = {
    "belgrade": {"lat": 44.787, "lng": 20.457, "normalized_name": "Belgrade, Serbia"},
    "istanbul": {"lat": 41.008, "lng": 28.978, "normalized_name": "Istanbul, Turkey"},
    "stari grad": {"lat": 44.819, "lng": 20.456, "normalized_name": "Stari Grad, Belgrade"},
    "savamala": {"lat": 44.812, "lng": 20.453, "normalized_name": "Savamala, Belgrade"},
    "vracar": {"lat": 44.798, "lng": 20.479, "normalized_name": "Vračar, Belgrade"},
}


@tool
def geocode(location: str) -> dict[str, float | str]:
    """Resolve a place name to coordinates and a normalized display name.

    Args:
        location: place or neighborhood name. Case-insensitive.

    Returns:
        Dict with keys ``lat``, ``lng``, ``normalized_name``. Returns a Belgrade
        fallback for unknown locations (Phase 1 behavior — Phase 4 returns an
        explicit "not found" status).
    """
    debug_enter("geocode", location=location)
    with trace("tool:geocode", location=location):
        if not ALLOW_RUNTIME_MOCKS:
            result: dict[str, float | str] = {
                "lat": 0.0,
                "lng": 0.0,
                "normalized_name": location,
                "status": "error",
                "reason": "Geocoding unavailable: no live geocoder configured.",
            }
            debug_exit("geocode", result=result)
            return result
        key = location.lower().strip()
        result = _MOCK_PLACES.get(
            key,
            {"lat": 44.787, "lng": 20.457, "normalized_name": location},
        )
        logger.debug("geocode %r -> %s", location, result)
        debug_exit("geocode", result=result)
        return result
