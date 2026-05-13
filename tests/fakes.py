"""Test doubles for chat models.

`FakeListChatModel` from langchain_core does not implement `bind_tools`, which
`langchain.agents.create_agent` requires. This file defines a minimal fake
that returns canned text responses without ever calling tools — enough to
exercise the orchestrator's guardrail + invocation path end-to-end.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class FakeAgentModel(BaseChatModel):
    """A chat model that always replies with `response` and accepts tool binding.

    Suitable for tests that exercise an agent without exercising actual tool
    selection. ``bind_tools`` returns self so the agent's tool-binding step
    succeeds.
    """

    response: str = "ok"

    @property
    def _llm_type(self) -> str:
        return "fake-agent"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(content=self.response)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeAgentModel:
        # No-op binding: the fake never decides to call a tool, it just replies.
        return self


class FakeToolCallingChatModel(BaseChatModel):
    """Chat model that emits a scripted sequence of tool calls / final text.

    Each element of ``responses`` is either:
      - a ``list[dict]`` of tool_call payloads (``{name, args, id}``) — emitted
        as an ``AIMessage`` with ``tool_calls`` set and empty content. The
        ReAct loop will execute the tools and call the model again.
      - a ``str`` — emitted as a final ``AIMessage`` text content, ending the
        loop.

    The list is consumed left-to-right via ``pop(0)``. When exhausted the
    model returns an empty terminal AIMessage so the loop terminates.

    Use this for tests that exercise the actual ReAct path of a sub-agent
    (where ``FakeAgentModel`` short-circuits before any tool would fire).
    """

    responses: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            msg = AIMessage(content="")
            return ChatResult(generations=[ChatGeneration(message=msg)])
        next_response = self.responses.pop(0)
        if isinstance(next_response, str):
            msg = AIMessage(content=next_response)
        else:
            # list of tool-call payloads
            msg = AIMessage(content="", tool_calls=list(next_response))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolCallingChatModel:
        return self
