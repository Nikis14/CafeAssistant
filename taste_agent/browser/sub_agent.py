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
from taste_agent.prompts import subagent_prompt

logger = get_logger(__name__)


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
                    SystemMessage(content=subagent_prompt()),
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
