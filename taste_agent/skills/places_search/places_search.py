"""places_search skill — Phase 1 mock implementation.

Phase 4 will replace `_MOCK_DATA` and the scoring with Foursquare API calls
plus Tavily web search and memory-based ranking. The public `run` signature
is stable so the orchestrator never has to change.
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel

from taste_agent.logging_ import get_logger, trace

logger = get_logger(__name__)


class PlaceFixture(TypedDict):
    """Mock-data shape. Keeps the type narrow so we don't need runtime asserts."""

    name: str
    address: str
    tags: list[str]
    review: str


class PlaceResult(BaseModel):
    name: str
    address: str
    reason: str
    review_snippet: str | None = None


# Static fixtures for Phase 1. Each entry has tags the scorer matches against
# the user's query. Tags are loose and overlap intentionally.
_MOCK_DATA: list[PlaceFixture] = [
    {
        "name": "Kafeterija",
        "address": "Cara Lazara 12, Belgrade",
        "tags": ["coffee", "cafe", "café", "specialty", "quiet", "wifi"],
        "review": "Best flat white in Belgrade. Calm during weekdays, busy on weekends.",
    },
    {
        "name": "Koffein",
        "address": "Resavska 22, Belgrade",
        "tags": ["coffee", "cafe", "cappuccino", "specialty", "roastery"],
        "review": "Serious cappuccino program, beans roasted in-house.",
    },
    {
        "name": "Iva New Balkan Cuisine",
        "address": "Dobračina 56, Belgrade",
        "tags": ["restaurant", "fine-dining", "balkan", "tasting-menu", "vegetarian", "quiet"],
        "review": "Tasting menu, intimate space, strong vegetarian options.",
    },
    {
        "name": "Ambar",
        "address": "Karađorđeva 2-4, Belgrade",
        "tags": ["restaurant", "balkan", "small-plates", "view", "danube"],
        "review": "Modern Balkan small plates with a Danube view.",
    },
    {
        "name": "Salon 1905",
        "address": "Kralja Petra 19, Belgrade",
        "tags": ["restaurant", "fine-dining", "european", "wine"],
        "review": "Refined European menu, deep wine list.",
    },
    {
        "name": "Pržionica D59B",
        "address": "Dobračina 59B, Belgrade",
        "tags": ["coffee", "cafe", "specialty", "roastery", "third-wave"],
        "review": "Third-wave roaster; pour-overs are the move.",
    },
]


def _score(query: str, place: PlaceFixture) -> int:
    """Number of tag hits in the query. Higher is better."""
    q = query.lower()
    return sum(1 for tag in place["tags"] if tag in q)


def run(query: str, location: str = "Belgrade", max_results: int = 5) -> list[dict[str, object]]:
    """Search for places matching ``query`` in ``location``.

    Args:
        query: free-form description of what the user wants.
        location: city or neighborhood; default Belgrade.
        max_results: how many results to return at most.

    Returns:
        List of result dicts with keys: name, address, reason, review_snippet.
    """
    with trace("skill:places_search", query=query, location=location):
        logger.debug("searching mock data: query=%r location=%r", query, location)

        scored: list[tuple[int, PlaceFixture]] = [(_score(query, p), p) for p in _MOCK_DATA]
        # Drop zero-score entries unless the query is generic ("place", "cafe", "where")
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
                ).model_dump()
            )
            logger.debug("candidate score=%d name=%s", score, place["name"])

        logger.info("places_search returned %d result(s)", len(results))
        return results
