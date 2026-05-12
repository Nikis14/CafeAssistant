---
name: reserve_table
description: Use this skill when the user wants to make a reservation at a specific restaurant or café. The skill drives a browser sub-agent through the reservation form. It STOPS before the final submit and registers a pending approval — the user must confirm in chat before the reservation is actually placed. Inputs include place_name, reservation_url, date (YYYY-MM-DD), time (HH:MM), party_size, contact_name, and optional contact_phone.
allowed-tools: [browser_navigate, browser_click, browser_fill, browser_dom_snapshot, request_user_approval]
---

# reserve_table

## When to use

Invoke this skill when the user has expressed a clear intent to book a table or seat at a specific place, and you have (or can ask for) the required details:

- "Book me a table at Iva for tomorrow 8pm, 2 people."
- "Reserve at Koffein on Friday at 11am for 1."
- "Can you make a reservation at <place> on <date>?"

If any required detail is missing (date, time, party size, contact name), ask the user before calling this skill. Reservations are irreversible — the skill is designed to surface a confirmation prompt before any submit happens, but it should not be invoked with partial information that would force the sub-agent to guess.

## Inputs

- `place_name` (str, required): human-readable place name.
- `reservation_url` (str, required): full URL of the place's reservation page.
- `date` (str, required): YYYY-MM-DD.
- `time` (str, required): HH:MM in 24-hour clock.
- `party_size` (int, required): number of people, ≥1.
- `contact_name` (str, required): name on the reservation.
- `contact_phone` (str, optional): contact phone number.

## What this skill does internally

1. If the host has a cached parser trace, replay it with the new arguments.
2. Otherwise, spawn a browser sub-agent (ReAct loop) with browser tools:
   - The sub-agent navigates to `reservation_url`.
   - It reads the DOM via `browser_dom_snapshot` and decides which fields to fill.
   - It fills date, time, party size, name, phone.
   - When the form is ready, it calls `request_user_approval` with a summary.
   - It STOPS — it does NOT click the final submit button.
3. The trace is saved to the parser cache (per host).
4. The skill returns `{"status": "pending_approval", "action_id": ..., "summary": ...}`.

## What happens after this skill returns

The orchestrator detects the pending approval and surfaces the summary to the user via chat. If the user replies "yes" / "confirm" / "go ahead", the orchestrator passes the deterministic confirm-gate and calls `finalize_reservation`, which clicks the submit button. If the user replies "no" / "cancel", the orchestrator discards the pending action.

The action guardrail (`taste_agent.guardrails.action`) enforces this — the final submit cannot run unless `is_approved(action_id)` is true. The check is non-LLM-based.

## Output (pseudo-JSON)

- `status`: `"pending_approval"` or `"failed"`
- `action_id`: 8-char hex (present only when status is `"pending_approval"`)
- `summary`: human-readable reservation summary (present only when pending)
- `source`: `"agentic"` for a fresh sub-agent run, `"cached"` for a replayed trace
- `error`: error message (present only when status is `"failed"`)
