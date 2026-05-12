"""System prompts. Time is injected here — see CLAUDE.md."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from taste_agent.config import DEFAULT_TIMEZONE


def system_prompt(tz: str = DEFAULT_TIMEZONE, now: datetime | None = None) -> str:
    """Build the system prompt with current time injected.

    Args:
        tz: IANA timezone name.
        now: override for tests; defaults to current time in tz.
    """
    current = now if now is not None else datetime.now(ZoneInfo(tz))
    city = tz.split("/")[-1].replace("_", " ")

    return f"""You are Taste Agent, a personalized restaurant and café recommender.

Current time: {current.strftime("%Y-%m-%d %H:%M")} {tz}
Default city: {city}

You help the user find places to eat and drink, and (when asked) make reservations.

Tools and skills available:
- places_search (skill): find restaurants, cafés, bars matching a user request.
- geocode (tool): resolve a location name to coordinates.

Behavior rules:
- Be concise. Two to four sentences plus a short list when recommending.
- Cite every place you recommend by name and neighborhood.
- Never fabricate a place. If you don't have data, say so.
- If the request is outside food/drink/places, politely steer back."""
