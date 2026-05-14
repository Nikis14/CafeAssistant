# Booking Flow Redesign

This document summarizes the recent booking-flow changes that split booking into a discovery phase and a preparation phase.

## Why this changed

The old reservation path mixed several responsibilities in one step:

- discovering whether a real online booking flow existed
- guessing which reservation fields a site required
- filling the form immediately with user values
- caching only a raw action trace

That caused a few concrete problems:

- the agent could ask for a static checklist before it actually knew what the site required
- cached traces could become opaque and hard to debug
- replay risked reusing stale filled values from an old run
- the assistant had no explicit artifact representing the learned booking flow

## New model

The booking stack is now split conceptually into:

1. `discover_booking_flow`
2. `reserve_table` preparation from a cached flow spec
3. `finalize_reservation`

### Discovery

Discovery is browser-driven, but it is intentionally non-destructive.

The new discovery sub-agent:

- navigates to a grounded booking URL
- clicks through to the booking form if needed
- inspects the DOM without filling user-specific values
- infers required and optional fields from the final form DOM
- caches the learned flow as structured data

Code:

- `taste_agent/tools/discover_booking_flow.py`
- `taste_agent/browser/sub_agent.py`
- `taste_agent/prompts/browser_discovery.txt`

### BookingFlowSpec

The learned artifact is `BookingFlowSpec`.

Code:

- `taste_agent/browser/specs.py`
- `taste_agent/browser/spec_cache.py`

It stores:

- host / platform metadata
- entry and final form URLs
- `steps_to_form`
- required fields
- optional fields
- selectors
- submit selector
- confidence / notes

This is the key boundary between “learn the site” and “instantiate the booking”.

### Preparation

Preparation now prefers the cached `BookingFlowSpec` over raw trace replay.

Instead of replaying old `fill(...)` calls with old values, the system:

- replays only the learned steps needed to reach the form
- injects the current booking values into the learned selectors
- stops before final submit
- registers `pending_approval`

Code:

- `taste_agent/skills/reserve_table/reserve_table.py`

The old raw trace cache still exists as a fallback, but spec-driven preparation is now the preferred path.

### Final submit

Final submit remains deterministic and approval-gated:

- `confirm_reservation` is registered via `request_user_approval`
- `finalize_reservation(action_id)` is the only place allowed to click submit

## Agent-facing improvements

### Discovery payload

`discover_booking_flow` no longer returns only raw spec JSON. It also returns:

- `required_fields`
- `optional_fields`
- `required_field_prompts`
- `optional_field_prompts`
- `requirements_summary`
- `next_step`

This makes the result directly usable by the main agent for follow-up questions.

### Reusing partial booking details

The orchestrator now extracts previously supplied booking details from earlier user turns and injects them into the system prompt.

Currently extracted:

- `date`
- `time`
- `party_size`
- `contact_name`
- `contact_phone`

This lets the assistant ask only for the still-missing details instead of restarting the booking checklist on every turn.

Code:

- `taste_agent/orchestrator.py`
- `taste_agent/prompts/__init__.py`
- `taste_agent/prompts/orchestrator.txt`

## Safety and correctness improvements

### Action trace isolation

Browser action recording is now scoped per run, so actions from an older reservation attempt are no longer leaked into a new cached recipe.

### Grounding

Booking calls are still validated before execution:

- reject obvious homepage URLs passed as reservation URLs
- reject placeholder contact names

### Internal UI behavior

The user message stays visible during assistant processing in Gradio because the UI now stages the user message before the assistant turn completes.

## Current limitations

This redesign is still an intermediate step, not the finished architecture.

Known limitations:

- field inference from discovery currently relies on DOM heuristics
- the main agent is prompted to use discovery-first behavior, but not all booking logic is orchestrator-enforced yet
- no explicit persistent per-conversation booking session object exists yet
- trace cache and spec cache are in-memory only

## Main files changed

- `taste_agent/browser/specs.py`
- `taste_agent/browser/spec_cache.py`
- `taste_agent/browser/sub_agent.py`
- `taste_agent/tools/discover_booking_flow.py`
- `taste_agent/skills/reserve_table/reserve_table.py`
- `taste_agent/orchestrator.py`
- `taste_agent/prompts/browser_discovery.txt`
- `taste_agent/prompts/orchestrator.txt`
- `taste_agent/prompts/__init__.py`

## Tests

Focused coverage now includes:

- `tests/test_discover_booking_flow.py`
- `tests/test_spec_cache.py`
- `tests/test_reserve_table.py`
- `tests/test_orchestrator.py`
