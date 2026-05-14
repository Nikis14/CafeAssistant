"""LangGraph-backed fallback place discovery over web_search results.

This tool is meant for place discovery when the primary structured Places API
is unavailable or too sparse. It runs a small workflow:

  1. Web search for relevant sources
  2. Extract normalized candidate venues from the snippets
  3. Return a place-like result schema the main agent can cite
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from taste_agent.config import DEFAULT_MODEL_ID
from taste_agent.logging_ import get_logger, trace
from taste_agent.tools.web_search import _do_search

logger = get_logger(__name__)


class _FallbackState(TypedDict, total=False):
    query: str
    location: str
    max_results: int
    search_query: str
    raw_results: list[dict[str, Any]]
    candidates: list[dict[str, Any]]


class _CandidatePayload(BaseModel):
    model_config = {"strict": True}

    class _Candidate(BaseModel):
        name: str
        reason: str
        review_snippet: str | None = None
        neighborhood: str | None = None
        website_url: str = ""
        maps_url: str = ""
        evidence_url: str = ""

    candidates: list[_Candidate] = Field(default_factory=list)


def _chat_model_kwargs(model_id: str) -> dict[str, Any]:
    normalized = model_id.lower()
    if normalized.startswith("openai/gpt-5"):
        return {}
    return {"temperature": 0.2}


def _build_search_query(query: str, location: str) -> str:
    return f"best {query} in {location}"


def _search_node(state: _FallbackState) -> dict[str, Any]:
    search_query = _build_search_query(state["query"], state["location"])
    with trace("tool:place_web_fallback:search", query=search_query[:80]):
        raw_results = _do_search(search_query, max_results=state["max_results"])
    return {"search_query": search_query, "raw_results": raw_results}


def _parse_candidate_payload(raw: str) -> _CandidatePayload:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in fallback extractor output")
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

    llm = ChatLiteLLM(model=DEFAULT_MODEL_ID, **_chat_model_kwargs(DEFAULT_MODEL_ID))
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
    with trace("tool:place_web_fallback:extract", model=DEFAULT_MODEL_ID):
        raw = llm.invoke([HumanMessage(content=prompt)])
    content = raw.content if isinstance(raw.content, str) else str(raw.content)
    payload = _parse_candidate_payload(content)
    return [c.model_dump() for c in payload.candidates[:max_results]]


def _extract_node(state: _FallbackState) -> dict[str, Any]:
    try:
        candidates = _extract_candidates(
            state["raw_results"],
            query=state["query"],
            location=state["location"],
            max_results=state["max_results"],
        )
    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        logger.warning("place_web_fallback extractor parse failed: %s", e)
        candidates = []
    except Exception as e:  # pragma: no cover
        logger.warning("place_web_fallback extractor failed: %s", e)
        candidates = []
    return {"candidates": candidates}


def _finalize_node(state: _FallbackState) -> dict[str, Any]:
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
                    "website_url": candidate.get("website_url", ""),
                    "reservation_url": "",
                    "phone": "",
                    "maps_url": candidate.get("maps_url", ""),
                    "source": "web_fallback",
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
                    "reason": raw_results[0].get("content", "Web fallback unavailable."),
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

    return {
        "candidates": [
            {
                "name": "",
                "address": state["location"],
                "reason": "Web fallback found no reliable place candidates.",
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

        g: Any = StateGraph(_FallbackState)
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


@tool
def place_web_fallback(
    query: str, location: str = "Belgrade", max_results: int = 5
) -> list[dict[str, Any]]:
    """Find place candidates from web sources when structured place search fails.

    Returns normalized place-like result objects so the main agent can cite
    web-backed recommendations without exposing raw provider errors.
    """
    initial_state: _FallbackState = {
        "query": query,
        "location": location,
        "max_results": max_results,
    }
    with trace("tool:place_web_fallback", query=query[:80], location=location):
        final_state = _get_graph().invoke(initial_state)
    results = final_state["candidates"]
    logger.info(
        "place_web_fallback returned %d candidate(s) for %r in %r",
        len(results),
        query[:60],
        location,
    )
    return results
