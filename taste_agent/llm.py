"""Model layer: every chat model in the app is built here.

Two ways to reach a model:

- **Direct** (``"neuraldeep/<model>"``, ``"openai/<model>"``) — a plain ``ChatOpenAI``
  pointed at an OpenAI-compatible endpoint, same as in the course notebooks.
  Each provider gets one shared ``InMemoryRateLimiter`` so the agent, the judge
  and the browser sub-agents together never exceed the provider's RPM.
- **Gateway** (``"gateway"``) — a LiteLLM ``Router`` in front of several keys and
  providers: load-balances over all ``NEURAL_DEEP_API_KEY*`` keys and falls back
  to OpenAI when they fail. This is the "LLM gateway" box from the lecture,
  embedded in-process instead of run as a separate proxy.

Imports of provider SDKs are lazy so tests that inject a fake factory don't
need them.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from taste_agent.logging_ import get_logger

logger = get_logger(__name__)

GATEWAY_MODEL_ID = "gateway"

PROVIDERS: dict[str, dict[str, Any]] = {
    "neuraldeep": {
        "base_url": "https://api.neuraldeep.tech/v1",
        "api_key": "NEURAL_DEEP_API_KEY",
        # Free tier: 20 requests/min per key. Keep a margin.
        "requests_per_minute": 18,
        "temperature": 0.2,
    },
    "openai": {
        "base_url": None,
        "api_key": "OPENAI_API_KEY",
        "requests_per_minute": 60,
        # GPT-5 family are reasoning models and reject ``temperature``.
        "temperature": None,
    },
}

BACKUP_MODEL = "openai/gpt-5-nano"


def _split_model_id(model_id: str) -> tuple[str, str]:
    provider, _, model = model_id.partition("/")
    if provider not in PROVIDERS or not model:
        raise ValueError(f"Unknown model id {model_id!r}; expected '<provider>/<model>'")
    return provider, model


_RATE_LIMITERS: dict[str, InMemoryRateLimiter] = {}


def _rate_limiter(name: str, requests_per_minute: int) -> InMemoryRateLimiter:
    """One limiter per provider, shared by every model instance that talks to it."""
    if name not in _RATE_LIMITERS:
        _RATE_LIMITERS[name] = InMemoryRateLimiter(
            requests_per_second=requests_per_minute / 60,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    return _RATE_LIMITERS[name]


def neuraldeep_keys() -> list[str]:
    """All NeuralDeep keys found in env: ``NEURAL_DEEP_API_KEY``, ``..._2``, ``..._3``..."""
    keys = []
    base = PROVIDERS["neuraldeep"]["api_key"]
    for suffix in ["", *(f"_{i}" for i in range(2, 10))]:
        value = os.environ.get(base + suffix)
        if value:
            keys.append(value)
    return keys


def _direct_model(model_id: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    provider, model = _split_model_id(model_id)
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["api_key"])
    if not api_key:
        raise ValueError(f"{config['api_key']} is not set (Colab secrets or .env)")
    extra: dict[str, Any] = {}
    if config["temperature"] is not None:
        extra["temperature"] = config["temperature"]
    if config["base_url"]:
        extra["base_url"] = config["base_url"]
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        timeout=120,
        # Frequent retries after 429 can get a free-tier key blocked.
        max_retries=1,
        rate_limiter=_rate_limiter(provider, config["requests_per_minute"]),
        **extra,
    )


def build_router_model_list(neuraldeep_model: str) -> list[dict[str, Any]]:
    """LiteLLM ``model_list``: one ``main`` deployment per NeuralDeep key + OpenAI ``backup``."""
    nd = PROVIDERS["neuraldeep"]
    model_list: list[dict[str, Any]] = [
        {
            "model_name": "main",
            "litellm_params": {
                "model": f"openai/{neuraldeep_model}",
                "api_base": nd["base_url"],
                "api_key": key,
                "temperature": nd["temperature"],
                # Router skips a deployment that has used up its per-minute budget.
                "rpm": nd["requests_per_minute"],
            },
        }
        for key in neuraldeep_keys()
    ]
    openai_key = os.environ.get(PROVIDERS["openai"]["api_key"])
    if openai_key:
        model_list.append(
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": BACKUP_MODEL,
                    "api_key": openai_key,
                    "rpm": PROVIDERS["openai"]["requests_per_minute"],
                },
            }
        )
    return model_list


def _gateway_model(neuraldeep_model: str) -> BaseChatModel:
    from langchain_litellm import ChatLiteLLMRouter
    from litellm import Router

    model_list = build_router_model_list(neuraldeep_model)
    if not model_list:
        raise ValueError("gateway needs at least one NEURAL_DEEP_API_KEY* or OPENAI_API_KEY")
    names = {m["model_name"] for m in model_list}
    router = Router(
        model_list=model_list,
        fallbacks=[{"main": ["backup"]}] if names == {"main", "backup"} else [],
        num_retries=1,
        timeout=120,
    )
    n_main = sum(m["model_name"] == "main" for m in model_list)
    logger.info("gateway: %d NeuralDeep key(s), backup=%s", n_main, "backup" in names)
    rpm = PROVIDERS["neuraldeep"]["requests_per_minute"] * max(n_main, 1)
    llm = ChatLiteLLMRouter(router=router, rate_limiter=_rate_limiter("gateway", rpm))
    llm.set_default_model("main" if n_main else "backup")
    return llm


def get_model(model_id: str) -> BaseChatModel:
    """Build the chat model for ``model_id`` (``"<provider>/<model>"`` or ``"gateway"``)."""
    from taste_agent.config import NEURALDEEP_MODEL

    if model_id == GATEWAY_MODEL_ID:
        return _gateway_model(NEURALDEEP_MODEL)
    return _direct_model(model_id)
