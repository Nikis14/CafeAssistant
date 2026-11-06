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
- `taste_agent/llm.py`: model layer — NeuralDeep / OpenAI via `ChatOpenAI` with a shared rate limiter, plus a LiteLLM Router "gateway" (several keys + OpenAI fallback).
- `taste_agent/guardrails/`: input (PII, injection, scope), output (PII/secret leaks, optional LLM judge), action (deterministic approval gate).
- `.github/workflows/ci.yml`: ruff + pytest on every push (all LLM calls mocked).

## Models

| Model id | What it is |
|---|---|
| `neuraldeep/qwen3.6-35b-a3b-noreason` | Default. Free tier: 20 requests/min per key |
| `gateway` | LiteLLM Router over every `NEURAL_DEEP_API_KEY*` key, falls back to `openai/gpt-5-nano` if `OPENAI_API_KEY` is set |
| `openai/gpt-5-nano` | OpenAI directly |

Memory reflection and the output LLM judge are off by default to keep one turn at ~2–4 model calls.

## Observability

With `LANGSMITH_TRACING=true` every turn is one LangSmith trace named `taste_agent_turn` with metadata `session_id`, `model_id`, `prompt_version` (hash of `prompts/orchestrator.txt`). Thumbs up/down in the chat is stored as `user_score` feedback on that trace.

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
uv sync --group dev
uv run python app.py
```

Open `http://127.0.0.1:7860`. Docker: `docker compose up --build`.

Set `NEURAL_DEEP_API_KEY` in `.env`. Without Foursquare / Tavily keys keep `TASTE_AGENT_ALLOW_RUNTIME_MOCKS=1`. For real reservation automation run `uv run playwright install chromium`.

## Development

```bash
uv run pytest tests/ -q
uv run ruff check .
uv run ruff format .
```
