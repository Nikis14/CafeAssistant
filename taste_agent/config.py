"""Runtime configuration: model registry, paths, env-var-derived constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelChoice:
    """A model option exposed in the Gradio dropdown."""

    label: str
    litellm_id: str
    env_var: str


MODEL_REGISTRY: list[ModelChoice] = [
    ModelChoice("Mistral Small", "mistral/mistral-small-latest", "MISTRAL_API_KEY"),
    ModelChoice("GPT-5", "openai/gpt-5", "OPENAI_API_KEY"),
    ModelChoice("GPT-5 mini", "openai/gpt-5-mini", "OPENAI_API_KEY"),
    ModelChoice("GPT-5 nano", "openai/gpt-5-nano", "OPENAI_API_KEY"),
    ModelChoice("Gemini 2.5 Flash", "gemini/gemini-2.5-flash", "GOOGLE_API_KEY"),
    ModelChoice("Claude Sonnet 4.6", "anthropic/claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    ModelChoice("Claude Haiku 4.5", "anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY"),
]

DEFAULT_MODEL_ID: str = "mistral/mistral-small-latest"

# Filesystem layout
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PACKAGE_ROOT: Path = PROJECT_ROOT / "taste_agent"
SKILLS_DIR: Path = PACKAGE_ROOT / "skills"

# Defaults used when the user hasn't specified otherwise
DEFAULT_TIMEZONE: str = "Europe/Belgrade"
