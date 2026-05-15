"""Prompt templates loaded from .txt files.

Why files, not inline Python strings:

- **Separation of concerns** — prompts are *content* (authored, tuned,
  versioned, often A/B-tested). Code is *control flow* (interpreted by the
  Python runtime). Mixing them makes both harder to evolve and harder to
  review.
- **LangSmith integration** — ``PromptTemplate.from_template`` shows up in
  trace views with its variables, so you can see *exactly* what the agent
  was prompted with on a given turn.
- **Templating discipline** — variable substitution is explicit
  (``{var}``); a missing variable raises at format time instead of
  silently rendering nothing.

Layout (this directory):

  __init__.py            (this module — loader + render helpers)
  orchestrator.txt       (the main agent's system prompt)
  browser_subagent.txt   (the reservation-flow sub-agent's system prompt)
  browser_discovery.txt  (the discovery-flow browser sub-agent prompt)
  output_judge.txt       (the output guardrail's factuality / citation judge)

All templates use f-string format (LangChain default). Literal curly braces
in a template (e.g. the JSON example inside ``output_judge.txt``) must be
escaped as ``{{`` and ``}}``.
"""

from __future__ import annotations

from datetime import datetime
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.prompts import PromptTemplate

from taste_agent.config import DEFAULT_TIMEZONE

_TEMPLATE_DIR = Path(__file__).parent


@cache
def load_template(name: str) -> PromptTemplate:
    """Load ``<name>.txt`` from this directory as a LangChain PromptTemplate.

    Cached: each template is read from disk once per process. Tests that
    edit templates between runs can call ``load_template.cache_clear()``.
    """
    path = _TEMPLATE_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8")
    return PromptTemplate.from_template(text, template_format="f-string")


# ── Helpers used by the system-prompt renderer ───────────────────────────────


def _render_facts(facts: dict[str, str] | None) -> str:
    """Render the memory-facts block. Empty string when no facts known."""
    if not facts:
        return ""
    lines = "\n".join(f"- {k}: {v}" for k, v in sorted(facts.items()))
    return (
        "\n\nWhat you know about the user (from prior conversations):\n"
        f"{lines}\n"
        "Use these facts to tailor recommendations. Do not parrot them back at "
        "the user unless they ask — just let them shape your choices."
    )


def _render_patterns(patterns_text: str | None) -> str:
    """Render the procedural-patterns block. Empty string when no patterns."""
    if not patterns_text:
        return ""
    return (
        "\n\nBehavioral patterns we have inferred about this user:\n"
        f"{patterns_text}\n"
        "Use these to bias your recommendations. Patterns are inferences, not "
        "rules — defer to anything the user explicitly says this turn."
    )


def _render_booking_context(booking_values: dict[str, str] | None) -> str:
    """Render known booking details already provided in this conversation."""
    if not booking_values:
        return ""
    labels = {
        "date": "Date",
        "time": "Time",
        "party_size": "Party size",
        "contact_name": "Reservation name",
        "contact_phone": "Contact phone",
    }
    ordered = ["date", "time", "party_size", "contact_name", "contact_phone"]
    lines = "\n".join(
        f"- {labels.get(key, key)}: {booking_values[key]}"
        for key in ordered
        if key in booking_values and booking_values[key]
    )
    if not lines:
        return ""
    return (
        "\n\nKnown booking details already provided earlier in this conversation:\n"
        f"{lines}\n"
        "Reuse these details when continuing a reservation flow. Ask only for "
        "the missing booking details."
    )


def _render_web_search_tool_section(include_web_search: bool) -> str:
    if not include_web_search:
        return ""
    return (
        "\n- web_search (tool): use this for explicit general web tasks like reviews, "
        "current hours, articles, or what people are saying about a place. Do not "
        "use it as the default venue-finding tool when `place_discovery` already fits."
    )


def _render_web_search_rule_section(include_web_search: bool) -> str:
    if not include_web_search:
        return ""
    return (
        "\n- Use `web_search` only when the user explicitly asks for general web evidence "
        "like reviews, current hours, articles, menus, or what people are saying."
    )


# ── Public render functions ──────────────────────────────────────────────────


def system_prompt(
    tz: str = DEFAULT_TIMEZONE,
    now: datetime | None = None,
    facts: dict[str, str] | None = None,
    patterns_text: str | None = None,
    booking_values: dict[str, str] | None = None,
    include_web_search: bool = False,
) -> str:
    """Render the orchestrator's system prompt for this turn.

    Args:
        tz: IANA timezone name (default Europe/Belgrade).
        now: override the timestamp (used in tests).
        facts: dict of known user facts; rendered as a trailing memory block.
        patterns_text: pre-rendered procedural patterns text (see
            ``ProceduralMemory.as_text``); rendered as a trailing block.
    """
    current = now if now is not None else datetime.now(ZoneInfo(tz))
    return load_template("orchestrator").format(
        timestamp=current.strftime("%Y-%m-%d %H:%M"),
        timezone=tz,
        web_search_tools_section=_render_web_search_tool_section(include_web_search),
        web_search_rule_section=_render_web_search_rule_section(include_web_search),
        facts_section=_render_facts(facts),
        patterns_section=_render_patterns(patterns_text),
        booking_context_section=_render_booking_context(booking_values),
    )


def subagent_prompt() -> str:
    """Render the browser sub-agent's system prompt (no variables)."""
    return load_template("browser_subagent").format()


def discovery_subagent_prompt() -> str:
    """Render the browser discovery sub-agent prompt (no variables)."""
    return load_template("browser_discovery").format()


def output_judge_prompt(*, context: str, response: str) -> str:
    """Render the output-guardrail judge prompt.

    Args:
        context: serialized conversation context (tool results, facts, etc.).
        response: the agent's draft reply being judged.
    """
    return load_template("output_judge").format(context=context, response=response)


def reflect_prompt() -> str:
    """Render the reflection sub-agent's system prompt (no variables — the
    turn data is passed via the HumanMessage to keep system/user separation
    clean)."""
    return load_template("reflect").format()


def derive_patterns_prompt(*, semantic_block: str, episodic_block: str) -> str:
    """Render the procedural-derivation prompt."""
    return load_template("derive_patterns").format(
        semantic_block=semantic_block or "(none)",
        episodic_block=episodic_block or "(none)",
    )
