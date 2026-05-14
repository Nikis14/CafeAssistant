"""Input guardrails: PII redaction, prompt-injection detection, scope heuristic.

Phase 1 implementation is heuristic/regex-based on purpose — Phase 4 swaps the
scope check for an LLM judge and the injection check for a stronger classifier.
The shape of the public function `run_input_guardrails` will stay stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from taste_agent.logging_ import get_logger, trace

logger = get_logger(__name__)

# ── PII patterns ─────────────────────────────────────────────────────────────
# Production systems should use Microsoft Presidio or similar; regex is for the
# seminar so students can see the mechanism without an external service.
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Match phone-like sequences: 8-15 digits, optionally separated by a single
# space, dash, or paren character between digits. Excludes ISO-like dates such
# as ``2026-05-15`` which otherwise look phone-like to a naive regex.
PHONE_RE = re.compile(
    r"(?<!\d)(?!\d{4}-\d{2}-\d{2}\b)\+?(?:\d[\s\-()]?){7,14}\d(?!\d)"
)
CARD_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")

# ── Prompt-injection heuristics ──────────────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+|the\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(
        r"disregard\s+(?:your\s+|the\s+)?(?:prior\s+|previous\s+)?"
        r"(?:instructions|prompt|system)",
        re.IGNORECASE,
    ),
    re.compile(r"you\s+are\s+now\s+(?:a\s+|an\s+)?\w+", re.IGNORECASE),
    re.compile(r"system\s+prompt\s*:", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?(?:system\s+)?prompt", re.IGNORECASE),
]

# ── Scope heuristic ──────────────────────────────────────────────────────────
# Phase 1: any of these tokens marks the input as plausibly in-scope. Cold
# greetings ("hi") are *not* in-scope by this heuristic but we don't refuse —
# we only flag, the orchestrator decides how to respond.
_SCOPE_KEYWORDS: frozenset[str] = frozenset(
    {
        "restaurant",
        "cafe",
        "café",
        "bar",
        "pub",
        "food",
        "eat",
        "drink",
        "coffee",
        "cappuccino",
        "espresso",
        "latte",
        "tea",
        "lunch",
        "dinner",
        "breakfast",
        "brunch",
        "snack",
        "reserve",
        "reservation",
        "book",
        "table",
        "menu",
        "cuisine",
        "wine",
        "italian",
        "serbian",
        "french",
        "japanese",
        "balkan",
        "vegetarian",
        "vegan",
        "halal",
        "kosher",
        "place",
        "places",
        "near",
        "spot",
        "spots",
        "remember",
        "memorize",
        "prefer",
        "like",
        "love",
        "hate",
        "allergy",
        "allergic",
    }
)


@dataclass
class GuardrailResult:
    """Outcome of the input-guardrail pass.

    Fields:
        cleaned_text: input with PII tokens replaced ([EMAIL]/[PHONE]/[CARD]).
        pii_redactions: count of redactions made.
        injection_flagged: True if the input matches an injection heuristic.
        out_of_scope: True if no scope keyword was found.
        refusal_message: set only when the orchestrator should short-circuit
            and reply with this text instead of invoking the agent.
    """

    cleaned_text: str
    pii_redactions: int
    injection_flagged: bool
    out_of_scope: bool
    refusal_message: str | None = None


def redact_pii(text: str) -> tuple[str, int]:
    """Replace PII tokens with labels. Returns (cleaned, count)."""
    count = 0
    # Order matters: cards before phones (cards are also long digit runs)
    for pattern, label in ((CARD_RE, "[CARD]"), (EMAIL_RE, "[EMAIL]"), (PHONE_RE, "[PHONE]")):
        text, n = pattern.subn(label, text)
        count += n
    return text, count


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def check_scope(text: str) -> bool:
    """Return True if the input is *out of scope* (no scope keyword present)."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return not (words & _SCOPE_KEYWORDS)


def run_input_guardrails(text: str) -> GuardrailResult:
    """Apply all input guardrails to user text. See `GuardrailResult` for outputs."""
    with trace("guardrail:input"):
        cleaned, n = redact_pii(text)
        injection = detect_injection(cleaned)
        out_of_scope = check_scope(cleaned)

        refusal: str | None = None
        if injection:
            refusal = (
                "Your message looks like an attempt to override my instructions. "
                "Try asking me about a place to eat or drink instead."
            )

        logger.info(
            "input_guardrail pii_redacted=%d injection=%s out_of_scope=%s",
            n,
            injection,
            out_of_scope,
        )

        return GuardrailResult(
            cleaned_text=cleaned,
            pii_redactions=n,
            injection_flagged=injection,
            out_of_scope=out_of_scope,
            refusal_message=refusal,
        )
