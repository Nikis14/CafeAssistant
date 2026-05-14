"""web_search tool — Tavily-backed web search exposed to the orchestrator.

Why Tavily: it's an LLM-friendly search API that returns short, citation-ready
snippets rather than raw HTML, which keeps token budgets reasonable.

The seminar's MCP discussion: we ship the *direct* Tavily integration here
because (a) it works offline-of-Node.js, and (b) it keeps the test suite
hermetic. The MCP-equivalent pattern is documented at the bottom of this file
so students see exactly what would change to swap Tavily for an MCP server.

Behavior:
- Reads ``TAVILY_API_KEY`` from the environment. No key → returns ``[]`` and
  logs a warning. This lets the agent still run on machines without a key.
- Each result is ``{"title", "url", "content", "score"}``. ``score`` is
  Tavily's relevance score (0..1).
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

from taste_agent.logging_ import debug_enter, debug_exit, get_logger, trace

logger = get_logger(__name__)

_ENV_KEY = "TAVILY_API_KEY"


def _sentinel_result(query: str, message: str) -> list[dict[str, Any]]:
    """Return a non-empty result payload for no-result / error cases."""
    return [
        {
            "title": "Web search unavailable",
            "url": "",
            "content": f"{message} Query: {query}",
            "score": 0.0,
            "status": "error",
        }
    ]


def _do_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Synchronous search via the Tavily SDK.

    Kept as a free function so tests can monkey-patch it without touching the
    @tool wrapper, and so the @tool docstring stays concise for the LLM.
    """
    api_key = os.environ.get(_ENV_KEY)
    if not api_key:
        logger.warning("%s not set; web_search returning sentinel result", _ENV_KEY)
        return _sentinel_result(query, f"{_ENV_KEY} is not set.")

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
    )
    raw_results = response.get("results", []) if isinstance(response, dict) else []
    results = [
        {
            "title": str(r.get("title", "")),
            "url": str(r.get("url", "")),
            "content": str(r.get("content", "")),
            "score": float(r.get("score", 0.0)),
        }
        for r in raw_results
    ]
    if not results:
        return _sentinel_result(query, "No web search results were returned.")
    return results


@tool
def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the open web for up-to-date information.

    Use this for things the agent's static skills cannot answer: recent
    reviews, blog posts, news about a place, current opening-hours pages,
    food trends. Returns a list of ``{title, url, content, score}`` items
    ordered by relevance.

    Args:
        query: free-form search query.
        max_results: up to 5 by default; 10 if you need broader coverage.
    """
    debug_enter("web_search", query=query, max_results=max_results)
    with trace("tool:web_search", query=query[:80], max_results=max_results):
        results = _do_search(query, max_results=max_results)
        if not results:
            results = _sentinel_result(query, "No web search results were returned.")
        logger.info("web_search returned %d result(s) for %r", len(results), query[:60])
        debug_exit("web_search", result=results)
        return results


# ── MCP equivalent (for the lecture) ─────────────────────────────────────────
#
# To swap this direct-SDK integration for an MCP server (Tavily ships one as
# `tavily-mcp` on npm), the change is:
#
#     from langchain_mcp_adapters.client import MultiServerMCPClient
#
#     async def build_mcp_tools():
#         client = MultiServerMCPClient({
#             "tavily": {
#                 "command": "npx",
#                 "args": ["-y", "tavily-mcp"],
#                 "env": {"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},
#             }
#         })
#         return await client.get_tools()
#
# The agent ends up with the same conceptual tool ("search the web"), but now
# the protocol between us and the search provider is MCP, not Tavily's bespoke
# REST API. The agent doesn't care; LangChain wraps both into StructuredTools.
# The pedagogical point is: MCP standardizes the *integration* surface, not the
# *capability*. Swap the search provider, keep the agent code unchanged.
