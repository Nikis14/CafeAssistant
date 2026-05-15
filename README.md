# Taste Agent

Taste Agent is a restaurant and café assistant with two main jobs:
- find relevant places using grounded search
- help complete reservation flows through a browser-backed booking skill

It runs as a Gradio app and keeps per-session memory about the user, such as durable preferences and past dining experiences.

## How It Works

- `app.py`: Gradio UI, session handling, memory panels, model selection.
- `taste_agent/orchestrator.py`: main LangGraph workflow for each turn.
- `taste_agent/tools/`: grounded tools such as `place_discovery`, `memory_read`, `memory_search`, and booking-flow discovery.
- `taste_agent/skills/reserve_table/`: reservation logic and final confirmation flow.
- `taste_agent/browser/`: Playwright-backed browser tools and sub-agent support.
- `taste_agent/memory/`: semantic, episodic, and procedural memory layers.
- `taste_agent/prompts/`: system prompts for the main agent, browser agent, and reflection flows.

## Turn Flow

For a normal user message, the system goes through:

1. input guardrail
2. approval check for pending irreversible actions
3. main agent execution
4. output guardrail
5. memory gating and reflection
6. procedural pattern derivation when enough new memory exists
7. final response formatting

The main agent receives:
- current conversation history
- known semantic facts about the user
- inferred behavioral patterns
- known booking details already provided in the current conversation

## Search And Booking

- Place recommendations use `place_discovery`, which combines place search and web enrichment, then ranks merged candidates.
- Reservations use `discover_booking_flow` plus `reserve_table`.
- Final reservation submission is guarded by an explicit user confirmation step.

## Run

```bash
cd production_system
cp .env.example .env
uv sync --extra dev
uv run playwright install chromium
uv run python app.py
```

Open `http://127.0.0.1:7860`.

Set at least one model provider key in `.env`. For reservation automation, Playwright plus the Chromium browser binary must be installed in the environment.

## Development

```bash
uv run pytest tests/ -q
uv run ruff check .
uv run ruff format .
```
