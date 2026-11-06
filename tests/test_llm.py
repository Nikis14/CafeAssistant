"""Tests for the model layer: direct models, gateway model list, prompt version."""

import pytest

from taste_agent import llm
from taste_agent.prompts import prompt_version


@pytest.fixture
def _no_keys(monkeypatch):
    # litellm calls load_dotenv() on import; import it first so a local .env
    # can't re-populate the keys we delete below.
    import litellm  # noqa: F401

    for name in ["OPENAI_API_KEY", *(f"NEURAL_DEEP_API_KEY{s}" for s in ["", "_2", "_3"])]:
        monkeypatch.delenv(name, raising=False)


def test_unknown_provider_rejected(_no_keys):
    with pytest.raises(ValueError, match="Unknown model id"):
        llm.get_model("mistral/mistral-small-latest")


def test_missing_key_rejected(_no_keys):
    with pytest.raises(ValueError, match="NEURAL_DEEP_API_KEY"):
        llm.get_model("neuraldeep/qwen")


def test_direct_model_points_at_provider(_no_keys, monkeypatch):
    monkeypatch.setenv("NEURAL_DEEP_API_KEY", "nd-key")
    model = llm.get_model("neuraldeep/qwen")
    assert model.model_name == "qwen"
    assert "neuraldeep" in str(model.openai_api_base)
    assert model.rate_limiter is llm.get_model("neuraldeep/qwen").rate_limiter


def test_gpt5_gets_no_temperature(_no_keys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    assert llm.get_model("openai/gpt-5-nano").temperature is None


def test_router_model_list_one_deployment_per_key_plus_backup(_no_keys, monkeypatch):
    monkeypatch.setenv("NEURAL_DEEP_API_KEY", "k1")
    monkeypatch.setenv("NEURAL_DEEP_API_KEY_2", "k2")
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    model_list = llm.build_router_model_list("qwen")
    names = [m["model_name"] for m in model_list]
    assert names == ["main", "main", "backup"]
    assert [m["litellm_params"]["api_key"] for m in model_list] == ["k1", "k2", "oa"]
    assert model_list[0]["litellm_params"]["model"] == "openai/qwen"


def test_gateway_without_keys_rejected(_no_keys):
    with pytest.raises(ValueError, match="gateway needs"):
        llm.get_model(llm.GATEWAY_MODEL_ID)


def test_prompt_version_is_short_stable_hash():
    assert prompt_version() == prompt_version()
    assert len(prompt_version()) == 8
    assert prompt_version("orchestrator") != prompt_version("output_judge")
