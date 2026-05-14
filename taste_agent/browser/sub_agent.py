"""Browser sub-agent: a ReAct loop that drives a BrowserBackend to fill a form.

This is the JSON-DSL observe-act loop we teach in the seminar. The "DSL" is
the set of tool calls the LLM emits — `browser_navigate`, `browser_click`,
`browser_fill`, `browser_page_context` — each interpreted by the deterministic
backend. The agent decides one action per turn, observes the new page state,
decides again.

The sub-agent's job ends when it calls `request_user_approval`. The final
submit is NOT in its tool belt — that's the deterministic confirm-gate's job.
"""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import urlparse
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from taste_agent.browser.backend import BrowserBackend
from taste_agent.config import DEFAULT_MODEL_ID
from taste_agent.browser.tools import build_browser_tools, make_request_approval_tool
from taste_agent.logging_ import debug_enter, debug_exit, get_logger, trace
from taste_agent.prompts import discovery_subagent_prompt, subagent_prompt

logger = get_logger(__name__)


class _GroundedDiscoveryBackend:
    """Restrict discovery-mode free navigation to the grounded starting host.

    The discovery agent may still follow real links by clicking on the current
    page. What it must not do is invent sibling domains because they look
    semantically related to the venue name.
    """

    def __init__(self, backend: BrowserBackend, *, initial_url: str) -> None:
        self._backend = backend
        self._allowed_host = urlparse(initial_url).netloc.lower()
        self.forbidden_selectors = backend.forbidden_selectors

    def navigate(self, url: str) -> None:
        target_host = urlparse(url).netloc.lower()
        if target_host != self._allowed_host:
            raise PermissionError(
                "Discovery navigation must stay on the grounded starting host; "
                f"refusing cross-host jump to {url!r}."
            )
        self._backend.navigate(url)

    def click(self, selector: str) -> None:
        self._backend.click(selector)

    def fill(self, selector: str, value: str) -> None:
        self._backend.fill(selector, value)

    def wait_for(self, selector: str, timeout_ms: int = 5000) -> None:
        self._backend.wait_for(selector, timeout_ms=timeout_ms)

    def screenshot(self) -> bytes:
        return self._backend.screenshot()

    def dom_snapshot(self, selector: str | None = None) -> str:
        return self._backend.dom_snapshot(selector)

    def raw_html(self) -> str:
        return self._backend.raw_html()

    def current_url(self) -> str:
        return self._backend.current_url()


def run_browser_subagent(
    goal: str,
    backend: BrowserBackend,
    model_factory: Callable[[str], BaseChatModel],
    model_id: str = DEFAULT_MODEL_ID,
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
    debug_enter("run_browser_subagent", goal=goal, model_id=model_id)
    tools = [*build_browser_tools(backend), make_request_approval_tool()]

    # Lazy import to keep test imports light.
    from langchain.agents import create_agent

    llm = model_factory(model_id)
    agent = create_agent(llm, tools)
    calls_before = len(getattr(backend, "calls", []))

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

    all_calls = getattr(backend, "calls", [])
    actions = list(all_calls[calls_before:])
    logger.info(
        "sub-agent finished: %d messages, %d backend actions",
        len(messages),
        len(actions),
    )

    result_payload = {
        "messages": messages,
        "last_message_text": text,
        "actions": actions,
    }
    debug_exit(
        "run_browser_subagent",
        result={
            "last_message_text": text,
            "n_messages": len(messages),
            "n_actions": len(actions),
        },
    )
    return result_payload


def run_browser_discovery_subagent(
    goal: str,
    backend: BrowserBackend,
    model_factory: Callable[[str], BaseChatModel],
    model_id: str = DEFAULT_MODEL_ID,
    initial_url: str | None = None,
) -> dict[str, Any]:
    """Run a browser sub-agent in discovery mode.

    Unlike ``run_browser_subagent``, this mode never gets a fill or approval
    tool. It is used to learn how to reach the booking form and what the page
    looks like before user-specific reservation values are collected.
    """
    debug_enter(
        "run_browser_discovery_subagent",
        goal=goal,
        model_id=model_id,
        initial_url=initial_url,
    )
    discovery_backend = backend
    if initial_url:
        discovery_backend = _GroundedDiscoveryBackend(backend, initial_url=initial_url)
    tools = build_browser_tools(discovery_backend)

    from langchain.agents import create_agent

    llm = model_factory(model_id)
    agent = create_agent(llm, tools)
    calls_before = len(getattr(backend, "calls", []))

    with trace("sub_agent:browser_discovery", goal=goal[:80]):
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=discovery_subagent_prompt()),
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
        parts: list[str] = []
        for part in last_content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        text = "".join(parts)
    else:
        text = str(last_content)

    all_calls = getattr(backend, "calls", [])
    actions = list(all_calls[calls_before:])
    final_url = backend.current_url()
    final_dom = backend.raw_html()
    logger.info(
        "discovery sub-agent finished: %d messages, %d backend actions",
        len(messages),
        len(actions),
    )
    result_payload = {
        "messages": messages,
        "last_message_text": text,
        "actions": actions,
        "final_url": final_url,
        "final_dom": final_dom,
    }
    debug_exit(
        "run_browser_discovery_subagent",
        result={
            "last_message_text": text,
            "n_messages": len(messages),
            "n_actions": len(actions),
            "final_url": final_url,
            "final_dom": final_dom,
        },
    )
    return result_payload
