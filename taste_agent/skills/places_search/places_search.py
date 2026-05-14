"""places_search skill — Foursquare-backed with a mock fallback.

Production path: Foursquare Places API v3, gated by ``FOURSQUARE_API_KEY``.
When the key is set, ``run`` issues a real lookup and returns live results.
When the key is unset (tests, offline demos), the same Belgrade mock fixtures
ship as in Phase 1. The skill's public contract is the same in both modes.

The two paths share the same ``PlaceResult`` schema so the agent prompt and
downstream guardrails don't need to know which backend ran.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, TypedDict

from pydantic import BaseModel

from taste_agent.logging_ import get_logger, trace

logger = get_logger(__name__)

_FOURSQUARE_KEY_ENV = "FOURSQUARE_API_KEY"
_FOURSQUARE_SEARCH_URL = "https://api.foursquare.com/v3/places/search"


class PlaceFixture(TypedDict):
    """Mock-data shape. Keeps the type narrow so we don't need runtime asserts."""

    name: str
    address: str
    tags: list[str]
    review: str
    website_url: str
    reservation_url: str
    phone: str
    maps_url: str


class PlaceResult(BaseModel):
    name: str
    address: str
    reason: str
    review_snippet: str | None = None
    website_url: str = ""
    reservation_url: str = ""
    phone: str = ""
    maps_url: str = ""
    source: str = "unknown"
    status: str = "ok"


def _sentinel_result(*, location: str, reason: str) -> list[dict[str, object]]:
    return [
        PlaceResult(
            name="",
            address=location,
            reason=reason,
            review_snippet=None,
            source="error",
            status="error",
        ).model_dump()
    ]


# Static fixtures (Phase 1). Used as the fallback when no Foursquare key is set
# AND in tests. Tags are loose and overlap intentionally.
_MOCK_DATA: list[PlaceFixture] = [
    {
        "name": "Kafeterija",
        "address": "Cara Lazara 12, Belgrade",
        "tags": ["coffee", "cafe", "café", "specialty", "quiet", "wifi"],
        "review": "Best flat white in Belgrade. Calm during weekdays, busy on weekends.",
        "website_url": "",
        "reservation_url": "",
        "phone": "",
        "maps_url": "",
    },
    {
        "name": "Koffein",
        "address": "Resavska 22, Belgrade",
        "tags": ["coffee", "cafe", "cappuccino", "specialty", "roastery"],
        "review": "Serious cappuccino program, beans roasted in-house.",
        "website_url": "",
        "reservation_url": "",
        "phone": "",
        "maps_url": "",
    },
    {
        "name": "Iva New Balkan Cuisine",
        "address": "Dobračina 56, Belgrade",
        "tags": ["restaurant", "fine-dining", "balkan", "tasting-menu", "vegetarian", "quiet"],
        "review": "Tasting menu, intimate space, strong vegetarian options.",
        "website_url": "",
        "reservation_url": "",
        "phone": "",
        "maps_url": "",
    },
    {
        "name": "Ambar",
        "address": "Karađorđeva 2-4, Belgrade",
        "tags": ["restaurant", "balkan", "small-plates", "view", "danube"],
        "review": "Modern Balkan small plates with a Danube view.",
        "website_url": "",
        "reservation_url": "",
        "phone": "",
        "maps_url": "",
    },
    {
        "name": "Salon 1905",
        "address": "Kralja Petra 19, Belgrade",
        "tags": ["restaurant", "fine-dining", "european", "wine"],
        "review": "Refined European menu, deep wine list.",
        "website_url": "",
        "reservation_url": "",
        "phone": "",
        "maps_url": "",
    },
    {
        "name": "Pržionica D59B",
        "address": "Dobračina 59B, Belgrade",
        "tags": ["coffee", "cafe", "specialty", "roastery", "third-wave"],
        "review": "Third-wave roaster; pour-overs are the move.",
        "website_url": "",
        "reservation_url": "",
        "phone": "",
        "maps_url": "",
    },
]


# ── Foursquare backend ───────────────────────────────────────────────────────


def _format_foursquare_address(loc: dict[str, Any]) -> str:
    """Render a Foursquare ``location`` block as a single-line address."""
    parts = [
        loc.get("address"),
        loc.get("locality"),
        loc.get("country"),
    ]
    return ", ".join(p for p in parts if isinstance(p, str) and p)


def _string_field(raw: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str):
            return value
    return ""


def _foursquare_search(
    query: str, location: str, max_results: int, api_key: str
) -> list[dict[str, object]]:
    """Call Foursquare Places v3. Raises ``urllib.error.URLError`` on failure."""
    params = urllib.parse.urlencode(
        {
            "query": query,
            "near": location,
            "limit": min(max_results, 50),
            "categories": "13000",  # Food & Beverage top-level category
        }
    )
    req = urllib.request.Request(
        f"{_FOURSQUARE_SEARCH_URL}?{params}",
        headers={
            "Authorization": api_key,
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.load(resp)

    raw_results = payload.get("results", []) if isinstance(payload, dict) else []
    formatted: list[dict[str, object]] = []
    for r in raw_results[:max_results]:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", ""))
        address = _format_foursquare_address(r.get("location", {}) or {})
        website_url = _string_field(r, "website", "website_url")
        phone = _string_field(r, "tel", "phone")
        maps_url = _string_field(r, "link", "maps_url")
        reservation_url = _string_field(r, "reservation_url", "booking_url")
        # Phase 4 ships without per-place reviews from Foursquare (would need a
        # second API call per result). Surfacing the categories as a reason
        # keeps the result structure populated.
        categories = r.get("categories", []) or []
        cat_names = [c.get("name") for c in categories if isinstance(c, dict)]
        reason = (
            f"Foursquare match: {', '.join(c for c in cat_names if c)}"
            if cat_names
            else "Foursquare match"
        )
        formatted.append(
            PlaceResult(
                name=name,
                address=address,
                reason=reason,
                review_snippet=None,
                website_url=website_url,
                reservation_url=reservation_url,
                phone=phone,
                maps_url=maps_url,
                source="foursquare",
                status="ok",
            ).model_dump()
        )
    return formatted


# ── Mock backend (Phase 1) ───────────────────────────────────────────────────


def _score(query: str, place: PlaceFixture) -> int:
    """Number of tag hits in the query. Higher is better."""
    q = query.lower()
    return sum(1 for tag in place["tags"] if tag in q)


def _mock_search(
    query: str, location: str, max_results: int
) -> list[dict[str, object]]:
    logger.debug("mock places search: query=%r location=%r", query, location)
    scored: list[tuple[int, PlaceFixture]] = [(_score(query, p), p) for p in _MOCK_DATA]
    generic = any(w in query.lower() for w in ("place", "where", "somewhere"))
    if not generic:
        scored = [(s, p) for s, p in scored if s > 0]
    scored.sort(key=lambda x: -x[0])

    results: list[dict[str, object]] = []
    for score, place in scored[:max_results]:
        matched_tags = [t for t in place["tags"] if t in query.lower()]
        reason = f"Matches: {', '.join(matched_tags)}" if matched_tags else "General fit"
        results.append(
            PlaceResult(
                name=place["name"],
                address=place["address"],
                reason=reason,
                review_snippet=place["review"],
                website_url=place["website_url"],
                reservation_url=place["reservation_url"],
                phone=place["phone"],
                maps_url=place["maps_url"],
                source="mock",
                status="ok",
            ).model_dump()
        )
        logger.debug("candidate score=%d name=%s", score, place["name"])
    return results


# ── Public entry point ───────────────────────────────────────────────────────


def run(query: str, location: str = "Belgrade", max_results: int = 5) -> list[dict[str, object]]:
    """Search for places matching ``query`` in ``location``.

    If ``FOURSQUARE_API_KEY`` is set, queries Foursquare Places v3.
    Otherwise (or on any API error), falls back to the Belgrade mock data so
    tests and offline demos keep working.

    Args:
        query: free-form description of what the user wants.
        location: city or neighborhood; default Belgrade.
        max_results: how many results to return at most.

    Returns:
        List of result dicts with keys: name, address, reason, review_snippet.
    """
    with trace("skill:places_search", query=query, location=location):
        api_key = os.environ.get(_FOURSQUARE_KEY_ENV)
        if api_key:
            # Production path: real Foursquare. If the API call fails, return
            # a non-empty sentinel result rather than the Belgrade mock — the
            # mock data is Belgrade-only, so quietly substituting it for a
            # query about Istanbul (or any other city) would surface fabricated
            # results.
            try:
                results = _foursquare_search(query, location, max_results, api_key)
                logger.info(
                    "places_search via Foursquare returned %d result(s)", len(results)
                )
                if results:
                    return results
                return _sentinel_result(
                    location=location,
                    reason="No matching places were returned by the upstream Places API.",
                )
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as e:
                logger.warning(
                    "Foursquare search failed: %s; returning sentinel result (mock would be Belgrade-only)",
                    e,
                )
                return _sentinel_result(
                    location=location,
                    reason=f"Places search unavailable: upstream Places API failed ({e}).",
                )

        # Offline / demo path: no key set → Belgrade-only mock data is OK
        # because there's no expectation of real coverage.
        results = _mock_search(query, location, max_results)
        logger.info("places_search via mock returned %d result(s)", len(results))
        if results:
            return results
        return _sentinel_result(
            location=location,
            reason="No matching places found in the local fallback dataset.",
        )
