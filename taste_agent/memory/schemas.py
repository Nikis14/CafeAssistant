"""Pydantic schemas for memory layers.

Two kinds of long-term memory we expose to the agent:

- **Semantic**: stable facts about the user (vegetarian, lives in Belgrade,
  prefers quiet places). Keyed by ``key`` so updates overwrite cleanly.
- **Episodic**: time-stamped experiences (visited Iva on 2026-05-12, loved
  the gnocchi). Many entries per place; searched by vector similarity.

Procedural memory (learned patterns like "prefers Italian when stressed") is
deferred to Phase 4 — derivable from episodic and not pedagogically distinct.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SemanticFact(BaseModel):
    """A durable fact about the user."""

    key: str = Field(..., description="Normalized fact name, e.g. 'dietary' or 'city'.")
    value: str = Field(..., description="Fact value, e.g. 'vegetarian' or 'Belgrade'.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How sure we are. Explicit user statements = 1.0; inferred = lower.",
    )
    source: str = Field(
        default="explicit",
        description="'explicit' (user said it) or 'inferred' (agent deduced it).",
    )
    created_at: datetime | None = None
    updated_at: datetime | None = None


class EpisodicEvent(BaseModel):
    """A logged user experience — typically a dining visit."""

    place_name: str = Field(..., description="Name of the place.")
    notes: str = Field(..., description="Free-form description of the experience.")
    rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="User-supplied 1-5 rating, optional.",
    )
    date: str | None = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) of the experience; defaults to today.",
    )
    address: str | None = None
    cuisine: str | None = None


class InferredPattern(BaseModel):
    """A behavioral pattern derived from the user's episodic + semantic history.

    Distinct from ``SemanticFact``: facts are what the user explicitly stated;
    patterns are what we noticed about their behavior.
    """

    text: str = Field(..., description="Free-form pattern description.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How sure we are based on the evidence count and consistency.",
    )
    evidence_count: int = Field(
        default=1,
        ge=1,
        description="Number of episodes / facts this pattern was derived from.",
    )
    derived_at: datetime | None = None
