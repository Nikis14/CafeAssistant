"""Memory write primitives.

The reflection sub-agent (``taste_agent.memory.reflection``) is the primary
caller. Each function does one focused thing: validate input, write to the
right store, detect conflicts, return a structured result the caller can
react to.

Previously this logic lived inside the ``memorize`` skill, which exposed it
as a tool to the orchestrator agent. We removed that surface — memory writes
now flow exclusively through reflection, which lets us keep the orchestrator
focused on responding and gives memory a single owner.
"""

from __future__ import annotations

from typing import Any

from taste_agent.logging_ import get_logger, trace
from taste_agent.memory import (
    EpisodicEvent,
    EpisodicMemory,
    SemanticMemory,
    get_default_episodic,
    get_default_semantic,
)
from taste_agent.memory.schemas import SemanticFact

logger = get_logger(__name__)


def write_semantic(
    key: str,
    value: str,
    *,
    confidence: float = 1.0,
    source: str = "inferred",
    store: SemanticMemory | None = None,
) -> dict[str, Any]:
    """Write a semantic fact with conflict detection.

    Returns a dict with:
        ``written`` (bool): whether the fact was persisted.
        ``conflict`` (dict | None): present when an existing explicit fact
            differs; caller should surface it (e.g., as a clarification).
    """
    sem = store or get_default_semantic()
    try:
        fact = SemanticFact(key=key, value=value, confidence=confidence, source=source)
    except (TypeError, ValueError) as e:
        logger.warning("invalid semantic fact %s=%s: %s", key, value, e)
        return {"written": False, "conflict": None, "error": str(e)}

    conflict = sem.detect_conflict(fact.key, fact.value)
    if conflict is not None:
        logger.info(
            "semantic conflict on %s: existing=%r proposed=%r — skipping write",
            fact.key,
            conflict.value,
            fact.value,
        )
        return {
            "written": False,
            "conflict": {
                "key": fact.key,
                "existing_value": conflict.value,
                "proposed_value": fact.value,
            },
        }

    sem.write(fact.key, fact.value, source=fact.source, confidence=fact.confidence)
    logger.info("semantic written: %s=%s (source=%s)", fact.key, fact.value, fact.source)
    return {"written": True, "conflict": None}


def write_episodic(
    place_name: str,
    notes: str,
    *,
    rating: int | None = None,
    date: str | None = None,
    address: str | None = None,
    cuisine: str | None = None,
    store: EpisodicMemory | None = None,
) -> dict[str, Any]:
    """Log an episodic event. Returns the doc id, or an error dict on validation failure."""
    epi = store or get_default_episodic()
    try:
        event = EpisodicEvent(
            place_name=place_name,
            notes=notes,
            rating=rating,
            date=date,
            address=address,
            cuisine=cuisine,
        )
    except (TypeError, ValueError) as e:
        logger.warning("invalid episodic event for %s: %s", place_name, e)
        return {"written": False, "error": str(e)}

    with trace("write_episodic", place=place_name):
        doc_id = epi.log(event)
    return {"written": True, "doc_id": doc_id, "place_name": place_name}
