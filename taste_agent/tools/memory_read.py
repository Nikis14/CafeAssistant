"""memory_read tool — return the agent's current semantic knowledge of the user.

The orchestrator already injects facts into the system prompt at every turn,
so this tool is for cases where the agent wants to *re-read* during a long
turn (e.g., after a memorize call that wrote new facts the agent should
respect immediately).
"""

from __future__ import annotations

from langchain_core.tools import tool

from taste_agent.logging_ import debug_enter, debug_exit, get_logger, trace
from taste_agent.memory import get_default_semantic

logger = get_logger(__name__)


@tool
def memory_read() -> dict[str, str]:
    """Return all semantic facts currently known about the user.

    Returns a flat ``{key: value}`` dictionary. Examples of common keys:
    ``dietary``, ``city``, ``favorite_cuisine``, ``budget_pref``, ``ambience_pref``.
    Empty dict if nothing has been memorized yet.
    """
    debug_enter("memory_read")
    with trace("tool:memory_read"):
        facts = get_default_semantic().as_dict()
        logger.debug("memory_read returned %d fact(s)", len(facts))
        debug_exit("memory_read", result=facts)
        return facts
