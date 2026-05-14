"""Output guardrail: PII redaction (deterministic) + factuality / citation
judgement (LLM).

This completes the three-guardrail-surfaces story. Recap:

  - Input guardrail (`input.py`) — fuzzy classification before the LLM runs:
    PII redaction, prompt-injection heuristics, scope check.
  - Action guardrail (`action.py`) — deterministic confirm-gate on
    irreversible actions. Never LLM-based.
  - Output guardrail (this module) — runs AFTER the agent has produced a
    response. Two layers:
      1. PII regex (deterministic, fast) — strips any phone/email/card the
         agent might have leaked.
      2. LLM judge (optional, env-controlled) — verifies factuality (claims
         grounded in the conversation context) and citation hygiene (place
         names are real, not fabricated). Returns concerns; the caller
         decides whether to expose them to the user.

**Cost note**: with the judge on, every user turn costs *two* LLM calls
(the agent itself + this judge). For production, consider prompt caching on
the judge prompt template, batching, or running the judge sampled
(every Nth turn) rather than always-on. The seminar defaults to always-on so
students see the cost surface.

Judge model resolution (see ``resolve_judge_model_id``):
  1. ``TASTE_AGENT_SKIP_OUTPUT_JUDGE=1`` → skip
  2. ``TASTE_AGENT_JUDGE_MODEL_ID=<litellm_id>`` → use that model
  3. ``OPENAI_API_KEY`` set → ``openai/gpt-5-nano``
  4. ``ANTHROPIC_API_KEY`` set → ``anthropic/claude-haiku-4-5`` (fallback)
  5. Otherwise → skip (no key for the default judge)

Per CLAUDE.md, we keep this hand-rolled rather than wrapping NeMo / Guardrails
AI / LLM Guard so students see what those frameworks do under the hood.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from taste_agent.guardrails.input import CARD_RE, EMAIL_RE, PHONE_RE
from taste_agent.logging_ import get_logger, trace
from taste_agent.prompts import output_judge_prompt

logger = get_logger(__name__)

# Env vars controlling judge behavior
_JUDGE_SKIP_ENV = "TASTE_AGENT_SKIP_OUTPUT_JUDGE"
_JUDGE_MODEL_ENV = "TASTE_AGENT_JUDGE_MODEL_ID"
_DEFAULT_JUDGE_MODEL_ID = "openai/gpt-5-nano"
_DEFAULT_JUDGE_KEY_ENV = "OPENAI_API_KEY"
_FALLBACK_JUDGE_MODEL_ID = "anthropic/claude-haiku-4-5"
_FALLBACK_JUDGE_KEY_ENV = "ANTHROPIC_API_KEY"


def resolve_judge_model_id() -> str | None:
    """Pick the judge model id from environment, or return None to skip.

    Returns:
        A LiteLLM model id when the judge should run, or ``None`` when the
        judge should be silently skipped.
    """
    if os.environ.get(_JUDGE_SKIP_ENV) == "1":
        return None
    override = os.environ.get(_JUDGE_MODEL_ENV)
    if override:
        return override
    if os.environ.get(_DEFAULT_JUDGE_KEY_ENV):
        return _DEFAULT_JUDGE_MODEL_ID
    if os.environ.get(_FALLBACK_JUDGE_KEY_ENV):
        return _FALLBACK_JUDGE_MODEL_ID
    logger.info(
        "%s / %s not set and no %s override; skipping output judge",
        _DEFAULT_JUDGE_KEY_ENV,
        _FALLBACK_JUDGE_KEY_ENV,
        _JUDGE_MODEL_ENV,
    )
    return None


class _JudgePayload(BaseModel):
    """Strict schema for the LLM judge's JSON output.

    Pydantic refuses string-coerced bools (``"false"``) and non-list concerns,
    so schema drift fails loudly instead of silently misreporting a flagged
    response as passing.
    """

    model_config = {"strict": True}

    factuality_ok: bool
    factuality_concerns: list[str] = Field(default_factory=list)
    citation_ok: bool
    citation_concerns: list[str] = Field(default_factory=list)

# Markers used in the redacted output. Distinct from the input-side markers so
# a leak is visually obvious in the chat ("redacted" not "[EMAIL]") and the
# difference makes the bidirectional flow visible during the lecture.
_OUT_EMAIL_TOKEN = "[REDACTED-EMAIL]"
_OUT_PHONE_TOKEN = "[REDACTED-PHONE]"
_OUT_CARD_TOKEN = "[REDACTED-CARD]"

# Also strip the input-side tokens if they somehow leak back into the output.
_INPUT_TOKEN_RE = re.compile(r"\[(EMAIL|PHONE|CARD)\]")


@dataclass
class OutputGuardrailResult:
    """Outcome of the output-guardrail pass.

    ``response_text`` is always safe to render — PII has been stripped before
    return. ``factuality_concerns`` / ``citation_concerns`` are non-empty when
    the LLM judge flagged something; the orchestrator decides whether to
    surface them to the user.
    """

    response_text: str
    pii_leaked: int = 0
    pii_concerns: list[str] = field(default_factory=list)
    factuality_ok: bool = True
    factuality_concerns: list[str] = field(default_factory=list)
    citation_ok: bool = True
    citation_concerns: list[str] = field(default_factory=list)
    internal_error_rewritten: bool = False
    internal_error_concerns: list[str] = field(default_factory=list)
    judge_rewritten: bool = False
    judge_rewrite_reason: str | None = None
    judge_skipped: bool = False
    judge_error: str | None = None

    @property
    def has_concerns(self) -> bool:
        return bool(
            self.pii_concerns
            or self.factuality_concerns
            or self.citation_concerns
            or self.internal_error_concerns
        )

    def summary_for_debug(self) -> dict[str, object]:
        return {
            "pii_leaked": self.pii_leaked,
            "pii_concerns": self.pii_concerns,
            "factuality_ok": self.factuality_ok,
            "factuality_concerns": self.factuality_concerns,
            "citation_ok": self.citation_ok,
            "citation_concerns": self.citation_concerns,
            "internal_error_rewritten": self.internal_error_rewritten,
            "internal_error_concerns": self.internal_error_concerns,
            "judge_rewritten": self.judge_rewritten,
            "judge_rewrite_reason": self.judge_rewrite_reason,
            "judge_skipped": self.judge_skipped,
            "judge_error": self.judge_error,
        }


# ── PII redaction (deterministic) ────────────────────────────────────────────


def redact_output_pii(text: str) -> tuple[str, int, list[str]]:
    """Strip any PII tokens the agent leaked. Returns (cleaned, count, concerns)."""
    concerns: list[str] = []
    cleaned = text
    n_total = 0

    cleaned, n = CARD_RE.subn(_OUT_CARD_TOKEN, cleaned)
    if n:
        concerns.append(f"credit-card numbers ({n})")
        n_total += n
    cleaned, n = EMAIL_RE.subn(_OUT_EMAIL_TOKEN, cleaned)
    if n:
        concerns.append(f"email addresses ({n})")
        n_total += n
    cleaned, n = PHONE_RE.subn(_OUT_PHONE_TOKEN, cleaned)
    if n:
        concerns.append(f"phone numbers ({n})")
        n_total += n
    # Strip any input-side redaction markers the LLM might have parroted back.
    cleaned, n = _INPUT_TOKEN_RE.subn("[REDACTED]", cleaned)
    if n:
        concerns.append(f"input-redaction tokens leaked ({n})")
        n_total += n

    return cleaned, n_total, concerns


_INTERNAL_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bTAVILY_API_KEY\b"), "env-var leak"),
    (re.compile(r"\bANTHROPIC_API_KEY\b"), "env-var leak"),
    (re.compile(r"\bMISTRAL_API_KEY\b"), "env-var leak"),
    (re.compile(r"\bOPENAI_API_KEY\b"), "env-var leak"),
    (re.compile(r"\bHTTP Error \d+\b", re.IGNORECASE), "provider http error leak"),
    (re.compile(r"\bUnauthorized\b", re.IGNORECASE), "provider auth leak"),
    (re.compile(r"\bupstream Places API\b", re.IGNORECASE), "upstream provider leak"),
    (re.compile(r"\blive Places search\b", re.IGNORECASE), "internal search-path leak"),
    (re.compile(r"\bauthorization error\b", re.IGNORECASE), "provider auth leak"),
    (re.compile(r"\bprovider failure\b", re.IGNORECASE), "provider failure leak"),
    (re.compile(r"\bAPI key\b", re.IGNORECASE), "api-key leak"),
]


def rewrite_internal_error_leaks(text: str) -> tuple[str, bool, list[str]]:
    """Remove user-visible infrastructure/provider leakage from a reply.

    The goal is not to hide substantive lack-of-data; it is to avoid surfacing
    raw internal kitchen details like env vars, auth failures, and upstream
    provider names. We drop contaminated sentences and keep the useful rest.
    """
    concerns: list[str] = []
    hit = False

    segments = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    kept: list[str] = []
    for segment in segments:
        stripped = segment.strip()
        if not stripped:
            continue
        segment_has_internal = False
        for pattern, label in _INTERNAL_ERROR_PATTERNS:
            if pattern.search(stripped):
                concerns.append(label)
                segment_has_internal = True
                hit = True
        if not segment_has_internal:
            kept.append(stripped)

    if not hit:
        return text, False, []

    cleaned = "\n\n".join(kept).strip()
    if not cleaned:
        cleaned = (
            "I couldn’t confirm results from one of my search sources right now, "
            "but I can still help with other available information."
        )
    return cleaned, True, sorted(set(concerns))


# ── LLM judge for factuality + citation ──────────────────────────────────────

ModelFactory = Callable[[str], BaseChatModel]

# The judge prompt is loaded from ``taste_agent/prompts/output_judge.txt`` at
# render time — see ``output_judge_prompt()`` and the prompts/ README.


def _parse_judge_output(raw: str) -> dict[str, object]:
    """Parse the LLM judge's JSON. Tolerate ```json fences and stray prose."""
    text = raw.strip()
    # Strip triple-backtick fences (with or without language tag)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find the first balanced JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in judge output")
    return json.loads(text[start : end + 1])


def _run_judge(
    response_text: str,
    context_summary: str,
    factory: ModelFactory,
    model_id: str,
) -> tuple[_JudgePayload | None, str | None]:
    """Run the LLM judge. Returns (validated_judgement, error_message)."""
    with trace("guardrail:output:judge", model=model_id):
        llm = factory(model_id)
        prompt = output_judge_prompt(
            context=context_summary or "(no context provided)",
            response=response_text,
        )
        try:
            raw = llm.invoke([HumanMessage(content=prompt)])
            content = raw.content if isinstance(raw.content, str) else str(raw.content)
            parsed = _parse_judge_output(content)
            payload = _JudgePayload.model_validate(parsed)
            return payload, None
        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            logger.warning("judge output failed to parse/validate: %s", e)
            return None, f"parse-error: {e}"
        except Exception as e:  # pragma: no cover - any LLM API error
            logger.warning("judge LLM call failed: %s", e)
            return None, f"llm-error: {e}"


def _rewrite_on_judge_failure(
    text: str,
    *,
    factuality_ok: bool,
    citation_ok: bool,
) -> tuple[str, bool, str | None]:
    """Replace a judged-bad draft with a grounded fallback.

    The judge has already determined the draft overclaims or cites unsupported
    specifics. At that point the safe move is to stop the bad draft from
    reaching the user rather than merely annotate it in debug output.
    """
    if factuality_ok and citation_ok:
        return text, False, None
    if not factuality_ok:
        reason = "factuality"
    else:
        reason = "citation"
    rewritten = (
        "I couldn't verify enough of that reply from grounded results to say it "
        "confidently. I can keep checking with confirmed sources, or help with "
        "another place once I verify it."
    )
    return rewritten, True, reason


# ── Public entry point ───────────────────────────────────────────────────────


def run_output_guardrails(
    response_text: str,
    *,
    context_summary: str = "",
    model_factory: ModelFactory | None = None,
    judge_model_id: str | None = None,
    skip_judge: bool | None = None,
) -> OutputGuardrailResult:
    """Apply output guardrails.

    Args:
        response_text: the agent's draft reply to the user.
        context_summary: short summary of the conversation context the agent
            had access to (tool results, retrieved places, known facts from
            memory). Used to ground the factuality judgement.
        model_factory: ``(model_id) -> BaseChatModel`` for the judge. If
            None, the judge is skipped.
        judge_model_id: which model to use. If ``None``, resolved from env via
            ``resolve_judge_model_id()`` — typically Haiku when
            ``ANTHROPIC_API_KEY`` is set, else skip.
        skip_judge: explicit override. If ``None``, derived from env / judge
            model resolution.

    Returns:
        ``OutputGuardrailResult``. ``response_text`` has PII stripped; concerns
        are exposed for the orchestrator to surface.
    """
    # Resolve skip + model id once (single source of truth).
    if skip_judge is None or judge_model_id is None:
        env_model = resolve_judge_model_id()
        if skip_judge is None:
            skip_judge = env_model is None
        if judge_model_id is None:
            judge_model_id = env_model or _DEFAULT_JUDGE_MODEL_ID

    with trace("guardrail:output", skip_judge=skip_judge):
        cleaned, internal_rewritten, internal_concerns = rewrite_internal_error_leaks(
            response_text
        )
        cleaned, n_pii, pii_concerns = redact_output_pii(cleaned)

        if skip_judge or model_factory is None:
            return OutputGuardrailResult(
                response_text=cleaned,
                pii_leaked=n_pii,
                pii_concerns=pii_concerns,
                internal_error_rewritten=internal_rewritten,
                internal_error_concerns=internal_concerns,
                judge_skipped=True,
            )

        payload, err = _run_judge(cleaned, context_summary, model_factory, judge_model_id)
        if payload is None:
            return OutputGuardrailResult(
                response_text=cleaned,
                pii_leaked=n_pii,
                pii_concerns=pii_concerns,
                internal_error_rewritten=internal_rewritten,
                internal_error_concerns=internal_concerns,
                judge_skipped=False,
                judge_error=err,
            )

        final_text, judge_rewritten, judge_rewrite_reason = _rewrite_on_judge_failure(
            cleaned,
            factuality_ok=payload.factuality_ok,
            citation_ok=payload.citation_ok,
        )
        return OutputGuardrailResult(
            response_text=final_text,
            pii_leaked=n_pii,
            pii_concerns=pii_concerns,
            factuality_ok=payload.factuality_ok,
            factuality_concerns=list(payload.factuality_concerns),
            citation_ok=payload.citation_ok,
            citation_concerns=list(payload.citation_concerns),
            internal_error_rewritten=internal_rewritten,
            internal_error_concerns=internal_concerns,
            judge_rewritten=judge_rewritten,
            judge_rewrite_reason=judge_rewrite_reason,
            judge_skipped=False,
        )
