---
name: places_search
description: Use this skill when the user wants to find restaurants, cafés, bars, or other places to eat or drink. Handles location resolution, search across data sources, and ranking by user preferences. Inputs are a free-form query and optionally a location and max_results.
allowed-tools: [geocode, memory_read, memory_search]
---

# places_search

## When to use

Invoke this skill when the user is asking for a place recommendation. Triggers include:

- "Where can I find a good cappuccino in Belgrade?"
- "Best vegetarian restaurant near Stari Grad"
- "Quiet café with WiFi"
- "Somewhere for dinner tonight, budget ~€30 per person"

If the user is making a reservation, finding a place is a precursor — call this skill first to identify candidates, then proceed with the reservation flow.

## Inputs

- `query` (str, required): free-form description of what the user wants.
- `location` (str, optional): place name or neighborhood. Defaults to Belgrade.
- `max_results` (int, optional): default 8.

## What this skill does internally

1. Resolve location → coordinates (via the `geocode` tool when needed).
2. Search the places data source(s) for matching venues.
3. Filter against the user's known preferences (via `memory_read` / `memory_search`).
4. Rank by relevance + preference fit + (later) recency of positive experience.
5. Return a structured list of recommendations.

## Output shape

A list of objects, each with:

- `name`
- `address`
- `reason` (short, mentions any preference match)
- `review_snippet` (optional)
- `website_url` (optional; grounded only when known)
- `reservation_url` (optional; grounded only when known)
- `phone` (optional)
- `maps_url` (optional)
- `source` (`mock`, `foursquare`, or `error`)
- `status` (`ok`, `error`, or later richer states)

For booking flows, prefer grounded `reservation_url` when present. Do not invent it.

## Phase status

- Phase 1 (current): returns mock data so the end-to-end pipeline can be exercised.
- Phase 4: wires real Foursquare API + Tavily web search and the memory-based ranking.
