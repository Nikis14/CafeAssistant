---
name: memorize
description: >
  Use this skill when the conversation contains information worth remembering
  long-term about the user — preferences, dietary rules, allergies, location,
  taste profile, or dining experiences. Accepts two parallel lists,
  `semantic_facts` (durable user traits such as dietary=vegetarian) and
  `episodic_events` (logged dining experiences). Performs conflict detection
  on semantic writes and reports any conflicts back so the calling agent can
  decide whether to overwrite.
allowed-tools: []
---

# memorize

## When to use

Call this skill any time the user shares something the agent should remember across turns or sessions. Common triggers:

- **Explicit preference statements**: "I'm vegetarian", "I hate cilantro", "I prefer quiet places".
- **Casual mentions**: "btw I'm allergic to nuts", "I live in Belgrade", "tonight I'm meeting Marko".
- **Logged experiences**: "I just had dinner at Iva, the gnocchi was incredible", "Skip Salon 1905, the service was awful".
- **Negative feedback / corrections**: "actually I prefer pinot noir over cabernet now".

Do NOT call this skill for one-off conversational filler ("hi", "thanks", "yes please") or for the agent's own recommendations.

## Inputs

Pass either or both of:

- `semantic_facts` (list, optional): durable traits. Each item is a dict with:
  - `key` (required): normalized fact name. Prefer short, snake_case keys:
    `dietary`, `city`, `favorite_cuisine`, `budget_pref`, `ambience_pref`,
    `allergy`, `companion_<name>_pref`.
  - `value` (required): the fact value (free text).
  - `confidence` (optional, 0.0–1.0): default 1.0 for explicit statements; use lower (0.5–0.8) when inferring.

- `episodic_events` (list, optional): logged experiences. Each item is a dict with:
  - `place_name` (required)
  - `notes` (required): free-form summary of the experience.
  - `rating` (optional, 1–5)
  - `date` (optional, YYYY-MM-DD)
  - `address` (optional)
  - `cuisine` (optional)

## What this skill does internally

1. Writes each `semantic_fact` to the SQLite-backed semantic memory store.
   Conflicts (a different explicit value already exists for the same key) are
   detected but NOT auto-overwritten — they are reported back so the calling
   agent can ask the user to confirm an update.
2. Writes each `episodic_event` to the Chroma vector store, indexed by a
   text summary (`place_name: notes`) for later similarity retrieval via
   `memory_search`.
3. Returns a structured report of what was written and what was skipped.

## Output (pseudo-JSON)

- `semantic_written`: list of `{key, value}` actually persisted
- `semantic_conflicts`: list of `{key, existing_value, proposed_value}` skipped
  due to conflict — orchestrator should surface these to the user
- `episodic_written`: list of `{place_name, doc_id}` actually persisted
- `episodic_skipped`: list of `{place_name, reason}` skipped for any reason
- `total_written`: integer
