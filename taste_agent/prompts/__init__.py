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


# ── Public render functions ──────────────────────────────────────────────────


def system_prompt(
    tz: str = DEFAULT_TIMEZONE,
    now: datetime | None = None,
    facts: dict[str, str] | None = None,
) -> str:
    """Render the orchestrator's system prompt for this turn.

    Args:
        tz: IANA timezone name (default Europe/Belgrade).
        now: override the timestamp (used in tests).
        facts: dict of known user facts; rendered as a trailing memory block.
    """
    current = now if now is not None else datetime.now(ZoneInfo(tz))
    city = tz.split("/")[-1].replace("_", " ")
    return load_template("orchestrator").format(
        timestamp=current.strftime("%Y-%m-%d %H:%M"),
        timezone=tz,
        city=city,
        facts_section=_render_facts(facts),
    )


def subagent_prompt() -> str:
    """Render the browser sub-agent's system prompt (no variables)."""
    return load_template("browser_subagent").format()


def output_judge_prompt(*, context: str, response: str) -> str:
    """Render the output-guardrail judge prompt.

    Args:
        context: serialized conversation context (tool results, facts, etc.).
        response: the agent's draft reply being judged.
    """
    return load_template("output_judge").format(context=context, response=response)
