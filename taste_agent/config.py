"""Runtime configuration: model registry, paths, env-var-derived constants."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelChoice:
    """A model option exposed in the Gradio dropdown."""

    label: str
    model_id: str
    env_var: str


NEURALDEEP_MODEL: str = "qwen3.6-35b-a3b-noreason"

MODEL_REGISTRY: list[ModelChoice] = [
    ModelChoice("Qwen 3.6 (NeuralDeep)", f"neuraldeep/{NEURALDEEP_MODEL}", "NEURAL_DEEP_API_KEY"),
    ModelChoice("Gateway: NeuralDeep keys + OpenAI backup", "gateway", "NEURAL_DEEP_API_KEY"),
    ModelChoice("GPT-5 nano (OpenAI)", "openai/gpt-5-nano", "OPENAI_API_KEY"),
]

DEFAULT_MODEL_ID: str = os.environ.get("TASTE_AGENT_MODEL_ID", f"neuraldeep/{NEURALDEEP_MODEL}")
ALLOW_RUNTIME_MOCKS: bool = os.environ.get("TASTE_AGENT_ALLOW_RUNTIME_MOCKS") == "1"

# Filesystem layout
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PACKAGE_ROOT: Path = PROJECT_ROOT / "taste_agent"
SKILLS_DIR: Path = PACKAGE_ROOT / "skills"

# Defaults used when the user hasn't specified otherwise
DEFAULT_TIMEZONE: str = "Europe/Belgrade"
