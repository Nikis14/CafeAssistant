"""Browser sub-agent: a ReAct loop that drives a BrowserBackend to fill a form.

This is the JSON-DSL observe-act loop we teach in the seminar. The "DSL" is
the set of tool calls the LLM emits — `browser_navigate`, `browser_click`,
`browser_fill`, `browser_dom_snapshot` — each interpreted by the deterministic
backend. The agent decides one action per turn, observes the new page state,
decides again.

The sub-agent's job ends when it calls `request_user_approval`. The final
submit is NOT in its tool belt — that's the deterministic confirm-gate's job.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from taste_agent.browser.backend import BrowserBackend
from taste_agent.browser.tools import build_browser_tools, make_request_approval_tool
from taste_agent.logging_ import get_logger, trace

logger = get_logger(__name__)


_SUBAGENT_PROMPT = """You are a browser-automation sub-agent. Your job is to fill \
out a reservation form on a website.

Tools available:
- browser_navigate(url): open a URL.
- browser_click(selector): click an element by CSS selector.
- browser_fill(selector, value): fill a form field.
- browser_wait_for(selector): wait for an element to appear.
- browser_dom_snapshot(selector): see the current page (defaults to body).
- request_user_approval(summary): register the form-ready-to-submit state.

Process:
1. Navigate to the reservation URL.
2. Read the page DOM with `browser_dom_snapshot`.
3. Fill the form (date, time, party size, name, phone) one field at a time.
4. After every action, you may re-snapshot to verify.
5. When the form is FULLY filled and the only remaining step is the final \
submit button, STOP. Call `request_user_approval` with a clear human-readable \
summary (date, time, party size, name, phone if provided).
6. DO NOT click the final submit button yourself. The user must approve first; \
the actual submit happens elsewhere.
7. After `request_user_approval` returns, output one short confirmation \
sentence and stop.

Be terse. One tool call per turn. Take the obvious next action."""


def run_browser_subagent(
    goal: str,
    backend: BrowserBackend,
    model_factory: Callable[[str], BaseChatModel],
    model_id: str = "default",
) -> dict[str, Any]:
    """Run a ReAct sub-agent against ``backend`` to achieve ``goal``.

    The sub-agent will navigate, click, and fill — but is forbidden by prompt
    from clicking the final submit. It calls ``request_user_approval`` when
    the form is ready, which registers a pending action with the action
    guardrail. The orchestrator then handles approval and finalization.

    Args:
        goal: human-language description of what to do (URL, date, party, etc.).
        backend: a BrowserBackend implementation (mock or real Playwright).
        model_factory: callable that builds a chat model for the given model id.
        model_id: identifier passed to ``model_factory``.

    Returns:
        dict with ``messages`` (full message log), ``last_message_text``, and
        ``actions`` (the recorded backend calls if the backend tracks them).
    """
    tools = [*build_browser_tools(backend), make_request_approval_tool()]

    # Lazy import to keep test imports light.
    from langchain.agents import create_agent

    llm = model_factory(model_id)
    agent = create_agent(llm, tools)

    with trace("sub_agent:browser", goal=goal[:80]):
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=_SUBAGENT_PROMPT),
                    HumanMessage(content=goal),
                ]
            }
        )

    messages = result["messages"]
    last = messages[-1]
    last_content = last.content
    if isinstance(last_content, str):
        text = last_content
    elif isinstance(last_content, list):
        # Take only text blocks (same logic as orchestrator._extract_text)
        parts: list[str] = []
        for part in last_content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        text = "".join(parts)
    else:
        text = str(last_content)

    actions = getattr(backend, "calls", [])
    logger.info("sub-agent finished: %d messages, %d backend actions", len(messages), len(actions))

    return {
        "messages": messages,
        "last_message_text": text,
        "actions": list(actions),
    }
