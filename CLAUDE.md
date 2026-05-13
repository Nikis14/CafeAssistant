# CLAUDE.md — Taste Agent (Seminar 7 capstone)

Project-specific conventions for working on the Taste Agent codebase. Read this before making changes.

## Project context

Taste Agent is a personalized restaurant/café recommender + reservation agent. Final capstone of the AI Agents course (Seminar 7: System Design, Production & Deployment).

**Architecture**: agent-on-top. A LangGraph ReAct orchestrator is wrapped by deterministic guardrails (input → orchestrator → output). The orchestrator has access to:

- **Atomic tools** — `memory_read`, `memory_search`, `geocode`, `web_search` (via MCP), browser primitives, `request_user_approval`
- **Skills** — `places_search`, `memorize`, `reserve_table` — multi-step procedures using the Anthropic Agent Skills convention (SKILL.md folder + Python entry function)
- **Sub-agents** — browser reservation agent (JSON-DSL observe-act loop driving Playwright)

**Three guardrail surfaces**:
- Input: PII redaction, prompt-injection check, scope check
- Output: factuality, citation, PII-leak (LLM-judge)
- Action: deterministic confirm-gate on irreversible tool calls — **never LLM-based**

**Guardrail implementation policy**: hand-rolled (regex + dict + set) for pedagogy. Production swap-in candidates documented in `taste_agent/guardrails/__init__.py`: Presidio for PII, NeMo Guardrails / Guardrails AI / LLM Guard for full pipelines, Llama Guard for safety classification, LangChain `moderation` chain for OpenAI-flavored toxicity. Keep `action.py` hand-rolled regardless — the deterministic-vs-LLM-judge contrast is the centerpiece teaching moment.

**Memory layers**: thread (Gradio `gr.State`, per browser session), semantic (SQLite key-value, what the user explicitly stated), episodic (Chroma vector store, dining experiences), procedural (SQLite, behavioral patterns inferred every ~5 episodes). All four are scoped per session via a ContextVar.

**Per-turn cost**: with default settings, one user turn invokes ~3 LLM calls — the main agent, the output-guardrail judge (`TASTE_AGENT_SKIP_OUTPUT_JUDGE=1` to disable), and the reflection sub-agent that auto-updates memory (`TASTE_AGENT_SKIP_REFLECTION=1` to disable). Procedural derivation adds an occasional call (~0.2 amortized at the 5-episode threshold). Tests set both skip-flags via `conftest.py` so the suite costs nothing.

**Stack**: LangGraph + LangChain, LiteLLM for provider switching (Claude / GPT-5 family / Gemini / Mistral), LangSmith for tracing, Gradio for UI, Playwright for browser, Tavily MCP for web search.

**Deployment target**: localhost only for now. Cloud Run is handled separately.

## Development rules — do all of these before declaring work done

### Lint & format with ruff

```
ruff check production_system/
ruff format production_system/
```

If ruff isn't installed: `pip install ruff`. Config in `production_system/pyproject.toml`. Line length 100.

Fix all ruff errors before saying a task is complete. Don't use `# noqa` to silence rules without a reason.

### Write unit tests

- Framework: **pytest**.
- Tests live in `production_system/tests/`, mirroring the `taste_agent/` package layout.
- Every tool, skill, and guardrail gets at least one unit test in the same commit it's added.
- Mock LLM calls (`langchain_core.language_models.fake.FakeListChatModel` or similar) and external APIs (Foursquare, Tavily, Playwright). Never hit real services from a unit test.
- One integration test (`tests/test_integration.py`) runs a sample query end-to-end with everything mocked.
- Run: `pytest production_system/tests/ -v`.

A task is not "done" until tests pass.

### Use the hierarchical logger — never `print()`

- Import: `from taste_agent.logging_ import get_logger, trace`.
- Each LangGraph node, each skill entry function, and each non-trivial tool wraps its work:
  ```python
  with trace("node_name", extra={"k": v}):
      ...
  ```
- Levels:
  - DEBUG — tool inputs/outputs, internal state
  - INFO — node entry/exit, agent turn boundaries
  - WARNING — guardrail flags, retries
  - ERROR — failures
- `print()` in `taste_agent/` is a lint failure (ruff `T201`).

## Code style

- Type hints required on all public functions.
- `pydantic.BaseModel` for tool args, skill inputs, agent state.
- Skills follow the SKILL.md convention: a folder under `taste_agent/skills/<name>/` containing `SKILL.md` (YAML frontmatter + markdown) and `<name>.py` exposing a `run(...)` function.
- No emojis in code or comments.
- Comments only when the *why* is non-obvious; do not narrate *what*.

## When to stop

- Ruff is clean AND tests pass → task can be marked completed.
- Tests fail, or you hit a blocker you cannot resolve → keep the task `in_progress` and report what's blocking. Do not mark "completed" with failing tests.
- Don't amend prior commits to "fix" CI — make a new commit.
