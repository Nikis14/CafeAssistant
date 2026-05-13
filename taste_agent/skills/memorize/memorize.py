"""memorize skill — persist structured facts to semantic + episodic memory.

The orchestrator agent identifies memory-worthy content during conversation
and calls this skill with already-structured facts. The skill's job is to:

1. Write each semantic fact, detecting conflicts with existing explicit facts.
2. Write each episodic event to the vector store for later similarity recall.
3. Report back what was written and what was skipped.

LLM-based reflection (the skill itself calling an LLM to extract facts from
free-form text) is intentionally deferred to Phase 4 — the orchestrator is
already an LLM and is a better place to do that reasoning. See SKILL.md for
the input contract the agent sees.
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


def _run_impl(
    *,
    semantic_facts: list[dict[str, Any]] | None,
    episodic_events: list[dict[str, Any]] | None,
    semantic_store: SemanticMemory,
    episodic_store: EpisodicMemory,
) -> dict[str, Any]:
    """Inner entry point — accepts injected stores. The public ``run`` uses
    the module-level defaults."""

    semantic_written: list[dict[str, str]] = []
    semantic_conflicts: list[dict[str, str]] = []
    episodic_written: list[dict[str, str]] = []
    episodic_skipped: list[dict[str, str]] = []

    # No-op guard: a no-arg call is almost certainly an LLM mistake. Surface
    # it explicitly rather than returning a silent zero so regressions are
    # visible in the trace / debug panel.
    if not semantic_facts and not episodic_events:
        logger.warning("memorize called with no semantic_facts and no episodic_events")
        return {
            "warning": "no input — pass semantic_facts and/or episodic_events",
            "semantic_written": [],
            "semantic_conflicts": [],
            "episodic_written": [],
            "episodic_skipped": [],
            "total_written": 0,
        }

    with trace(
        "skill:memorize",
        n_semantic=len(semantic_facts or []),
        n_episodic=len(episodic_events or []),
    ):
        for raw in semantic_facts or []:
            try:
                fact = SemanticFact(**raw)
            except (TypeError, ValueError) as e:
                semantic_conflicts.append(
                    {"key": str(raw.get("key", "?")), "error": f"invalid fact: {e}"}
                )
                continue

            conflict = semantic_store.detect_conflict(fact.key, fact.value)
            if conflict is not None:
                semantic_conflicts.append(
                    {
                        "key": fact.key,
                        "existing_value": conflict.value,
                        "proposed_value": fact.value,
                    }
                )
                logger.info(
                    "semantic conflict on %s: existing=%r proposed=%r",
                    fact.key,
                    conflict.value,
                    fact.value,
                )
                continue

            semantic_store.write(
                fact.key,
                fact.value,
                source=fact.source,
                confidence=fact.confidence,
            )
            semantic_written.append({"key": fact.key, "value": fact.value})

        for raw in episodic_events or []:
            try:
                event = EpisodicEvent(**raw)
            except (TypeError, ValueError) as e:
                episodic_skipped.append(
                    {
                        "place_name": str(raw.get("place_name", "?")),
                        "reason": f"invalid event: {e}",
                    }
                )
                continue
            doc_id = episodic_store.log(event)
            episodic_written.append({"place_name": event.place_name, "doc_id": doc_id})

        total = len(semantic_written) + len(episodic_written)
        logger.info(
            "memorize complete: %d semantic, %d episodic, %d conflicts, %d skipped",
            len(semantic_written),
            len(episodic_written),
            len(semantic_conflicts),
            len(episodic_skipped),
        )

        return {
            "semantic_written": semantic_written,
            "semantic_conflicts": semantic_conflicts,
            "episodic_written": episodic_written,
            "episodic_skipped": episodic_skipped,
            "total_written": total,
        }


def run(
    semantic_facts: list[dict[str, Any]] | None = None,
    episodic_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist memory-worthy facts and events.

    Args:
        semantic_facts: durable user traits ({key, value, confidence?, source?}).
        episodic_events: dining experiences ({place_name, notes, rating?, date?, ...}).

    Returns:
        Report dict — see SKILL.md for the contract.
    """
    return _run_impl(
        semantic_facts=semantic_facts,
        episodic_events=episodic_events,
        semantic_store=get_default_semantic(),
        episodic_store=get_default_episodic(),
    )
