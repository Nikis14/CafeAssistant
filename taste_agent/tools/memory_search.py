"""memory_search tool — vector-similarity search across episodic memory.

Used when the agent wants to recall past experiences ("the last time I was in
Vienna I had a great schnitzel — where was it?").
"""

from __future__ import annotations

from langchain_core.tools import tool

from taste_agent.logging_ import get_logger, trace
from taste_agent.memory import get_default_episodic

logger = get_logger(__name__)


def _sentinel_event(query: str) -> list[dict[str, object]]:
    return [
        {
            "status": "no_results",
            "query": query,
            "notes": "No matching episodic memory entries were found.",
        }
    ]


@tool
def memory_search(query: str, k: int = 5) -> list[dict[str, object]]:
    """Search the user's logged dining experiences by similarity to ``query``.

    Args:
        query: free-form description ("Italian dinner that surprised me").
        k: max results to return (default 5).

    Returns:
        List of event dicts with: place_name, notes, rating (if any),
        date (ISO), address, cuisine. Ordered by relevance.
    """
    with trace("tool:memory_search", query=query[:60], k=k):
        events = get_default_episodic().search(query, k=k)
        logger.debug("memory_search returned %d event(s)", len(events))
        if not events:
            return _sentinel_event(query)
        return [e.model_dump(exclude_none=True) for e in events]
