"""Parallel place discovery: structured places + web enrichment merged.

This tool is the default discovery path for venue recommendations. It runs the
structured places API search and the web enrichment path in parallel, then merges the
results into one normalized candidate list.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from langchain_core.tools import tool

from taste_agent.logging_ import debug_enter, debug_exit, get_logger, trace
from taste_agent.skills.places_search.places_search import run as places_search_run
from taste_agent.tools.place_web_fallback import place_web_enrichment

logger = get_logger(__name__)


class _DiscoveryState(TypedDict, total=False):
    query: str
    location: str
    max_results: int
    fetch_results: int
    places_results: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    merged_results: list[dict[str, Any]]


def _places_node(state: _DiscoveryState) -> dict[str, Any]:
    with trace("tool:place_discovery:places", query=state["query"][:80]):
        results = places_search_run(
            state["query"],
            state["location"],
            state["fetch_results"],
        )
    return {"places_results": results}


def _web_node(state: _DiscoveryState) -> dict[str, Any]:
    with trace("tool:place_discovery:web", query=state["query"][:80]):
        results = place_web_enrichment.invoke(
            {
                "query": state["query"],
                "location": state["location"],
                "max_results": state["fetch_results"],
            }
        )
    return {"web_results": results}


def _normalize_name(name: str) -> str:
    lowered = name.lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _is_error_result(item: dict[str, Any]) -> bool:
    return item.get("status") == "error" or item.get("source") == "error"


def _merge_reason(places_reason: str, web_reason: str) -> str:
    if places_reason and web_reason:
        return f"{places_reason} Web: {web_reason}"
    return places_reason or web_reason


def _query_terms(query: str) -> set[str]:
    return {token for token in re.findall(r"\w+", query.lower()) if len(token) >= 3}


def _score_candidate(candidate: dict[str, Any], *, query: str) -> tuple[int, int]:
    haystacks = [
        str(candidate.get("name", "")),
        str(candidate.get("reason", "")),
        str(candidate.get("review_snippet", "")),
        str(candidate.get("address", "")),
    ]
    text = " ".join(haystacks).lower()
    score = 0
    for term in _query_terms(query):
        if term in text:
            score += 3

    source = str(candidate.get("source", ""))
    if source == "places+web":
        score += 6
    elif source == "web_enrichment":
        score += 3
    elif source == "foursquare":
        score += 2

    lowered_query = query.lower()
    if any(
        token in lowered_query
        for token in ("coffee", "cafe", "café", "espresso", "cappuccino", "roastery")
    ):
        if any(
            token in text
            for token in (
                "coffee",
                "cafe",
                "café",
                "espresso",
                "cappuccino",
                "roastery",
                "specialty",
            )
        ):
            score += 5
        if "hotel" in text:
            score -= 6

    if candidate.get("website_url"):
        score += 1
    if candidate.get("maps_url"):
        score += 1
    if candidate.get("review_snippet"):
        score += 1

    return score, len(str(candidate.get("reason", "")))


def _summarize_error_results(
    places_results: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
    *,
    location: str,
) -> list[dict[str, Any]]:
    places_reason = ""
    web_reason = ""
    if places_results and _is_error_result(places_results[0]):
        places_reason = str(places_results[0].get("reason", "")).strip()
    if web_results and _is_error_result(web_results[0]):
        web_reason = str(web_results[0].get("reason", "")).strip()

    reason = _merge_reason(places_reason, web_reason)
    if not reason:
        reason = "Place discovery returned no usable results."

    return [
        {
            "name": "",
            "address": location,
            "reason": reason,
            "review_snippet": None,
            "website_url": "",
            "reservation_url": "",
            "phone": "",
            "maps_url": "",
            "source": "error",
            "status": "error",
        }
    ]


def _merge_results(
    places_results: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
    *,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    places_ok = [r for r in places_results if not _is_error_result(r) and r.get("name")]
    web_ok = [r for r in web_results if not _is_error_result(r) and r.get("name")]

    if not places_ok and not web_ok:
        fallback_location = ""
        if places_results:
            fallback_location = str(places_results[0].get("address", "")).strip()
        if not fallback_location and web_results:
            fallback_location = str(web_results[0].get("address", "")).strip()
        return _summarize_error_results(
            places_results,
            web_results,
            location=fallback_location,
        )

    web_by_name = {_normalize_name(r["name"]): r for r in web_ok}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for place in places_ok:
        key = _normalize_name(str(place.get("name", "")))
        web_match = web_by_name.get(key)
        if web_match is None:
            for candidate_key, candidate in web_by_name.items():
                if key and (key in candidate_key or candidate_key in key):
                    web_match = candidate
                    break
        merged_item = dict(place)
        if web_match is not None:
            merged_item["reason"] = _merge_reason(
                str(place.get("reason", "")),
                str(web_match.get("reason", "")),
            )
            if not merged_item.get("review_snippet"):
                merged_item["review_snippet"] = web_match.get("review_snippet")
            if not merged_item.get("website_url"):
                merged_item["website_url"] = web_match.get("website_url", "")
            if not merged_item.get("maps_url"):
                merged_item["maps_url"] = web_match.get("maps_url", "")
            merged_item["source"] = "places+web"
        seen.add(key)
        merged.append(merged_item)

    for web_item in web_ok:
        key = _normalize_name(str(web_item.get("name", "")))
        if key and key in seen:
            continue
        merged.append(dict(web_item))

    merged.sort(key=lambda item: _score_candidate(item, query=query), reverse=True)
    return merged[:max_results]


def _merge_node(state: _DiscoveryState) -> dict[str, Any]:
    merged = _merge_results(
        state.get("places_results", []),
        state.get("web_results", []),
        query=state["query"],
        max_results=state["max_results"],
    )
    return {"merged_results": merged}


def _run_parallel(state: _DiscoveryState) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        places_future = pool.submit(_places_node, state)
        web_future = pool.submit(_web_node, state)
        return {
            **places_future.result(),
            **web_future.result(),
        }


_GRAPH: Any | None = None


def _get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        from langgraph.graph import END, StateGraph

        g: Any = StateGraph(_DiscoveryState)
        g.add_node("parallel_search", _run_parallel)
        g.add_node("merge", _merge_node)
        g.set_entry_point("parallel_search")
        g.add_edge("parallel_search", "merge")
        g.add_edge("merge", END)
        _GRAPH = g.compile()
    return _GRAPH


def reset_graph_cache() -> None:
    global _GRAPH
    _GRAPH = None


@tool
def place_discovery(
    query: str, location: str = "Belgrade", max_results: int = 8
) -> list[dict[str, Any]]:
    """Discover restaurant/cafe candidates by combining structured places and web.

    Use this as the default venue-discovery tool when the user wants
    recommendations. It returns one merged normalized candidate list.
    """
    debug_enter("place_discovery", query=query, location=location, max_results=max_results)
    initial_state: _DiscoveryState = {
        "query": query,
        "location": location,
        "max_results": max_results,
        "fetch_results": max(max_results * 2, 12),
    }
    with trace("tool:place_discovery", query=query[:80], location=location):
        final_state = _get_graph().invoke(initial_state)
    results = final_state["merged_results"]
    logger.info(
        "place_discovery returned %d merged result(s) for %r in %r",
        len(results),
        query[:60],
        location,
    )
    debug_exit("place_discovery", result=results)
    return results
