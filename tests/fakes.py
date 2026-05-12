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
