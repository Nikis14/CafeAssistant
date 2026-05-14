# Taste Agent

Personalized restaurant and café recommender + reservation agent. Final capstone for the AI Agents course (Seminar 7: System Design, Production & Deployment).

## Setup

```
cd production_system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — at minimum set one provider key (ANTHROPIC_API_KEY recommended)
```

## Run

```
python app.py
```

Open http://127.0.0.1:7860.

## Development

```
ruff check .
ruff format .
pytest tests/ -v
```

See `CLAUDE.md` for project conventions (logging, testing, style).
See `docs/booking_flow_redesign.md` for the current reservation-flow architecture notes.

## Phase status

- [x] Phase 1 — Skeleton + happy path (input guardrail, orchestrator, mocked places_search, Gradio, hierarchical logger)
- [ ] Phase 2 — Reservation flow with browser sub-agent (JSON DSL + confirm-gate)
- [ ] Phase 3 — Memory layers (semantic + episodic)
- [ ] Phase 4 — Remaining intents, real Foursquare, Tavily MCP web search, output guardrail
- [ ] Phase 5 — Seminar notebook + lecture material
