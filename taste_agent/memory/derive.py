"""Procedural-pattern derivation: a single LLM call over accumulated memory.

Triggered conditionally by the orchestrator (every N new episodes — see
``maybe_derive_procedural``). One LLM call, no agent loop. The LLM gets the
full semantic + episodic history and emits a JSON list of patterns;
``ProceduralMemory.replace_all`` wholesale-replaces the prior snapshot.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from taste_agent.config import DEFAULT_MODEL_ID
from taste_agent.logging_ import get_logger, trace
from taste_agent.memory import (
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
    get_default_episodic,
    get_default_procedural,
    get_default_semantic,
)
from taste_agent.memory.schemas import InferredPattern
from taste_agent.prompts import derive_patterns_prompt

logger = get_logger(__name__)

ModelFactory = Callable[[str], BaseChatModel]


class _DerivedPayload(BaseModel):
    """Strict schema for the derivation LLM's JSON output."""

    model_config = {"strict": True}

    class _PatternItem(BaseModel):
        text: str
        confidence: float = Field(default=1.0, ge=0.0, le=1.0)
        evidence_count: int = Field(default=1, ge=1)

    patterns: list[_PatternItem] = Field(default_factory=list)


def _parse_payload(raw: str) -> _DerivedPayload:
    """Parse the LLM's JSON output. Tolerates markdown fences + stray prose."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in derive output")
    return _DerivedPayload.model_validate(json.loads(text[start : end + 1]))


def _format_semantic_block(sem: SemanticMemory) -> str:
    facts = sem.all()
    if not facts:
        return "(no semantic facts)"
    return "\n".join(
        f"- {f.key}={f.value} (source={f.source}, confidence={f.confidence:.2f})"
        for f in facts
    )


def _format_episodic_block(epi: EpisodicMemory, max_events: int) -> str:
    """Render up to ``max_events`` recent episodes for the derivation prompt.

    The cap exists for prompt-token budgeting; for users with very long
    episodic histories, older episodes won't influence the derivation. The
    caller controls the cap via ``derive_patterns(max_events=...)``.
    """
    events = epi.list_recent(k=max_events)
    if not events:
        return "(no episodic events)"
    return "\n".join(
        f"- [{e.date or '?'}] {e.place_name} ({e.cuisine or '?'}): {e.notes}"
        + (f" rating={e.rating}" if e.rating is not None else "")
        for e in events
    )


def derive_patterns(
    *,
    semantic: SemanticMemory | None = None,
    episodic: EpisodicMemory | None = None,
    model_factory: ModelFactory,
    model_id: str = DEFAULT_MODEL_ID,
    max_events: int = 50,
) -> list[InferredPattern] | None:
    """Run the LLM derivation.

    Args:
        max_events: cap on episodes included in the prompt (for token budget).
            Patterns derived from a truncated window won't reflect older
            episodes; raise this for users with very long histories.

    Returns:
        - ``list[InferredPattern]`` on success (possibly empty if no patterns
          are well-supported — caller should treat as "valid result, no
          patterns" and replace its store accordingly).
        - ``None`` on LLM / parse / validation failure. Caller MUST preserve
          existing procedural memory in this case — silently writing ``[]``
          would wipe accumulated patterns on a single transient error.
    """
    sem = semantic or get_default_semantic()
    epi = episodic or get_default_episodic()
    semantic_block = _format_semantic_block(sem)
    episodic_block = _format_episodic_block(epi, max_events=max_events)

    with trace("derive_patterns", model=model_id):
        llm = model_factory(model_id)
        prompt = derive_patterns_prompt(
            semantic_block=semantic_block, episodic_block=episodic_block
        )
        try:
            raw = llm.invoke([HumanMessage(content=prompt)])
            content = raw.content if isinstance(raw.content, str) else str(raw.content)
            payload = _parse_payload(content)
        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            logger.warning("derive_patterns parse/validate failed: %s", e)
            return None
        except Exception as e:  # pragma: no cover
            logger.warning("derive_patterns LLM call failed: %s", e)
            return None

    now = datetime.now(timezone.utc)
    return [
        InferredPattern(
            text=p.text,
            confidence=p.confidence,
            evidence_count=p.evidence_count,
            derived_at=now,
        )
        for p in payload.patterns
    ]


def maybe_derive_procedural(
    *,
    model_factory: ModelFactory,
    episode_threshold: int = 5,
    semantic: SemanticMemory | None = None,
    episodic: EpisodicMemory | None = None,
    procedural: ProceduralMemory | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    max_events: int = 50,
) -> bool:
    """Run derivation only when enough new episodes accumulated since last time.

    Returns True if derivation ran (and procedural was replaced); False
    otherwise. Idempotent — calling on the same memory state twice
    short-circuits the second call.
    """
    epi = episodic or get_default_episodic()
    proc = procedural or get_default_procedural()

    current = epi.count()
    last = proc.last_derive_episode_count()
    if current - last < episode_threshold:
        return False

    logger.info(
        "deriving procedural patterns: current=%d last=%d delta=%d >= threshold=%d",
        current,
        last,
        current - last,
        episode_threshold,
    )
    patterns = derive_patterns(
        semantic=semantic,
        episodic=epi,
        model_factory=model_factory,
        model_id=model_id,
        max_events=max_events,
    )
    if patterns is None:
        # LLM / parse failure — DO NOT wipe existing patterns and DO NOT
        # advance the threshold counter. The next turn that crosses the
        # threshold (potentially the very next one) will retry.
        logger.warning(
            "derive_patterns failed; preserving existing %d pattern(s) and will retry on next threshold crossing",
            proc.count(),
        )
        return False
    proc.replace_all(patterns)
    proc.set_last_derive_episode_count(current)
    return True
