"""LangGraph-backed web enrichment for place discovery.

This tool enriches place discovery with open-web evidence. It runs a small
workflow:

  1. Web search for relevant sources
  2. Extract normalized candidate venues from the snippets
  3. Return a place-like result schema the main agent can cite
"""

from __future__ import annotations

from contextvars import ContextVar, Token
import json
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError, field_validator

from taste_agent.config import DEFAULT_MODEL_ID
from taste_agent.logging_ import debug_enter, debug_exit, get_logger, trace
from taste_agent.tools.web_search import _do_search

logger = get_logger(__name__)
_current_model_id: ContextVar[str] = ContextVar(
    "place_web_enrichment_model_id", default=DEFAULT_MODEL_ID
)


class _EnrichmentState(TypedDict, total=False):
    query: str
    location: str
    max_results: int
    search_queries: list[str]
    raw_results: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    extract_error: str


class _CandidatePayload(BaseModel):
    model_config = {"strict": True}

    class _Candidate(BaseModel):
        name: str
        reason: str
        review_snippet: str | None = None
        neighborhood: str | None = None
        website_url: str | None = None
        maps_url: str | None = None
        evidence_url: str | None = None

        @field_validator("website_url", "maps_url", "evidence_url", mode="before")
        @classmethod
        def _normalize_optional_urls(cls, value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            return str(value)

    candidates: list[_Candidate] = Field(default_factory=list)


def _chat_model_kwargs(model_id: str) -> dict[str, Any]:
    normalized = model_id.lower()
    if normalized.startswith("openai/gpt-5"):
        return {}
    return {"temperature": 0.2}


def set_current_model_id(model_id: str) -> Token[str]:
    """Bind the active UI-selected model for web enrichment extraction."""
    return _current_model_id.set(model_id)


def reset_current_model_id(token: Token[str]) -> None:
    _current_model_id.reset(token)


def _get_current_model_id() -> str:
    return _current_model_id.get()


def _build_search_query(query: str, location: str) -> str:
    return f"best {query} in {location}"


def _build_search_queries(query: str, location: str) -> list[str]:
    base = query.strip()
    normalized = re.sub(r"\s+", " ", base).strip()
    variants = [
        _build_search_query(normalized, location),
        f"{normalized} cafe {location}",
        f"{normalized} brunch {location}",
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for variant in variants:
        lowered = variant.lower().strip()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(variant)
    return unique


def _dedupe_raw_results(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for result in raw_results:
        url = str(result.get("url", "")).strip().lower()
        title = str(result.get("title", "")).strip().lower()
        key = (url, title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _search_node(state: _EnrichmentState) -> dict[str, Any]:
    search_queries = _build_search_queries(state["query"], state["location"])
    merged_results: list[dict[str, Any]] = []
    error_results: list[dict[str, Any]] = []
    per_query_max = max(state["max_results"], 5)
    for search_query in search_queries:
        with trace("tool:place_web_enrichment:search", query=search_query[:80]):
            raw_results = _do_search(search_query, max_results=per_query_max)
        if raw_results and raw_results[0].get("status") == "error":
            error_results.extend(raw_results[:1])
            continue
        merged_results.extend(raw_results)
    deduped = _dedupe_raw_results(merged_results)
    if deduped:
        deduped = deduped[: max(state["max_results"] * 3, state["max_results"])]
    else:
        deduped = error_results[:1]
    return {"search_queries": search_queries, "raw_results": deduped}


def _parse_candidate_payload(raw: str) -> _CandidatePayload:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in web enrichment extractor output")
    return _CandidatePayload.model_validate(json.loads(text[start : end + 1]))


def _extract_candidates(
    raw_results: list[dict[str, Any]],
    *,
    query: str,
    location: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """LLM-based candidate extraction over snippet results.

    Kept as a standalone function so tests can monkey-patch it directly.
    """
    if not raw_results:
        return []
    if raw_results and raw_results[0].get("status") == "error":
        return []

    from langchain_litellm import ChatLiteLLM

    model_id = _get_current_model_id()
    llm = ChatLiteLLM(model=model_id, **_chat_model_kwargs(model_id))
    prompt = (
        "You are extracting restaurant/cafe candidates from web search snippets.\n"
        "Only use places explicitly mentioned in the snippets. Do not invent URLs, "
        "addresses, or booking links. Return strict JSON with shape:\n"
        '{"candidates":[{"name":"...","reason":"...","review_snippet":"...","neighborhood":"...",'
        '"website_url":"...","maps_url":"...","evidence_url":"..."}]}\n'
        f"Target query: {query}\n"
        f"Target location: {location}\n"
        f"Return at most {max_results} candidates.\n\n"
        f"Web results:\n{json.dumps(raw_results, ensure_ascii=False)}"
    )
    with trace("tool:place_web_enrichment:extract", model=model_id):
        raw = llm.invoke([HumanMessage(content=prompt)])
    content = raw.content if isinstance(raw.content, str) else str(raw.content)
    payload = _parse_candidate_payload(content)
    return [c.model_dump() for c in payload.candidates[:max_results]]


def _extract_node(state: _EnrichmentState) -> dict[str, Any]:
    try:
        candidates = _extract_candidates(
            state["raw_results"],
            query=state["query"],
            location=state["location"],
            max_results=state["max_results"],
        )
        if not candidates:
            return {
                "candidates": [],
                "extract_error": (
                    "Web enrichment extractor returned no candidates from the available web results."
                ),
            }
    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        logger.warning("place_web_enrichment extractor parse failed: %s", e)
        return {
            "candidates": [],
            "extract_error": "Web enrichment extractor returned an invalid payload.",
        }
    except Exception as e:  # pragma: no cover
        logger.warning("place_web_enrichment extractor failed: %s", e)
        return {
            "candidates": [],
            "extract_error": "Web enrichment extractor failed.",
        }
    return {"candidates": candidates, "extract_error": ""}


def _finalize_node(state: _EnrichmentState) -> dict[str, Any]:
    raw_results = state.get("raw_results") or []
    candidates = state.get("candidates") or []
    if candidates:
        results = []
        for candidate in candidates:
            neighborhood = candidate.get("neighborhood") or state["location"]
            reason = candidate.get("reason") or "Web-sourced recommendation"
            evidence_url = candidate.get("evidence_url", "")
            if evidence_url:
                reason = f"{reason} Source: {evidence_url}"
            results.append(
                {
                    "name": candidate.get("name", ""),
                    "address": neighborhood,
                    "reason": reason,
                    "review_snippet": candidate.get("review_snippet"),
                    "website_url": candidate.get("website_url") or "",
                    "reservation_url": "",
                    "phone": "",
                    "maps_url": candidate.get("maps_url") or "",
                    "source": "web_enrichment",
                    "status": "ok",
                }
            )
        return {"candidates": results}

    if raw_results and raw_results[0].get("status") == "error":
        return {
            "candidates": [
                {
                    "name": "",
                    "address": state["location"],
                    "reason": raw_results[0].get("content", "Web enrichment unavailable."),
                    "review_snippet": None,
                    "website_url": "",
                    "reservation_url": "",
                    "phone": "",
                    "maps_url": "",
                    "source": "error",
                    "status": "error",
                }
            ]
        }

    extract_error = str(state.get("extract_error", "")).strip()
    return {
        "candidates": [
            {
                "name": "",
                "address": state["location"],
                "reason": extract_error or "Web enrichment returned no candidates.",
                "review_snippet": None,
                "website_url": "",
                "reservation_url": "",
                "phone": "",
                "maps_url": "",
                "source": "error",
                "status": "error",
            }
        ]
    }


_GRAPH: Any | None = None


def _get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        from langgraph.graph import END, StateGraph

        g: Any = StateGraph(_EnrichmentState)
        g.add_node("search", _search_node)
        g.add_node("extract", _extract_node)
        g.add_node("finalize", _finalize_node)
        g.set_entry_point("search")
        g.add_edge("search", "extract")
        g.add_edge("extract", "finalize")
        g.add_edge("finalize", END)
        _GRAPH = g.compile()
    return _GRAPH


def reset_graph_cache() -> None:
    global _GRAPH
    _GRAPH = None


@tool("place_web_enrichment")
def place_web_enrichment(
    query: str, location: str = "Belgrade", max_results: int = 5
) -> list[dict[str, Any]]:
    """Find place candidates and supporting evidence from web sources.

    Use this to enrich or broaden venue discovery with web evidence such as
    review pages, menu pages, official/social links, and related sources.
    Returns normalized place-like result objects.
    """
    debug_enter(
        "place_web_enrichment",
        query=query,
        location=location,
        max_results=max_results,
    )
    initial_state: _EnrichmentState = {
        "query": query,
        "location": location,
        "max_results": max_results,
    }
    with trace("tool:place_web_enrichment", query=query[:80], location=location):
        final_state = _get_graph().invoke(initial_state)
    results = final_state["candidates"]
    logger.info(
        "place_web_enrichment returned %d candidate(s) for %r in %r",
        len(results),
        query[:60],
        location,
    )
    debug_exit("place_web_enrichment", result=results)
    return results


place_web_fallback = place_web_enrichment
