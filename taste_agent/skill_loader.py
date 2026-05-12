"""Loader for SKILL.md folders (Anthropic Agent Skills convention).

Convention used in this project:

    skills/<skill_name>/
        SKILL.md           YAML frontmatter (name, description, ...) + body
        <skill_name>.py    exposes a ``run(...)`` function

The loader parses the frontmatter, imports the module, and wraps ``run`` as a
LangChain StructuredTool whose description includes the markdown body. The
agent then sees rich, progressive instructions for each skill without us
stuffing the system prompt.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import StructuredTool

from taste_agent.logging_ import get_logger

logger = get_logger(__name__)


def _parse_frontmatter(skill_md: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into (frontmatter_dict, body_markdown)."""
    if not skill_md.lstrip().startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter delimited by '---'")
    # split into ['', frontmatter, body]
    parts = skill_md.split("---", 2)
    if len(parts) < 3:
        raise ValueError("SKILL.md missing closing '---' after frontmatter")
    meta = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    if "name" not in meta or "description" not in meta:
        raise ValueError("SKILL.md frontmatter must include 'name' and 'description'")
    return meta, body


def _load_module(skill_dir: Path) -> Any:
    """Import the ``<skill_dir>/<skill_dir.name>.py`` module.

    Canonical case: when the skill lives under ``taste_agent/skills/<name>/``,
    we load via ``importlib.import_module``. This produces the *same instance*
    as a regular ``import taste_agent.skills.<name>.<name>``. Module-level
    state (e.g., ``set_default_backend``) is then shared, not forked.

    Adhoc case: tests construct skills in ``tmp_path``. Those aren't on the
    canonical import path, so we fall back to ``spec_from_file_location`` —
    a one-off module that lives only inside the test scope.
    """
    py_path = skill_dir / f"{skill_dir.name}.py"
    if not py_path.exists():
        raise FileNotFoundError(f"Expected skill module at {py_path}")

    canonical_name = f"taste_agent.skills.{skill_dir.name}.{skill_dir.name}"
    try:
        return importlib.import_module(canonical_name)
    except ModuleNotFoundError:
        # Skill isn't on the canonical import path (typical for tmp_path tests).
        # Load directly from the file. Use a unique synthetic module name to
        # avoid colliding with the canonical path in sys.modules.
        synthetic_name = f"taste_agent.skills._adhoc.{skill_dir.name}"
        spec = importlib.util.spec_from_file_location(synthetic_name, py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build module spec for {py_path}") from None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def load_skill(skill_dir: Path) -> StructuredTool:
    """Load one SKILL.md folder into a LangChain StructuredTool.

    Args:
        skill_dir: directory containing SKILL.md and <name>.py.

    Returns:
        A StructuredTool whose description embeds the SKILL.md body so the
        agent sees full usage instructions when it considers this skill.
    """
    md_path = skill_dir / "SKILL.md"
    if not md_path.exists():
        raise FileNotFoundError(f"No SKILL.md in {skill_dir}")
    meta, body = _parse_frontmatter(md_path.read_text(encoding="utf-8"))

    module = _load_module(skill_dir)
    if not hasattr(module, "run"):
        raise AttributeError(f"Skill '{meta['name']}' module must expose a `run(...)` function")

    description = meta["description"]
    if body:
        description = f"{description}\n\n--- Skill instructions ---\n{body}"

    return StructuredTool.from_function(
        func=module.run,
        name=meta["name"],
        description=description,
    )


def load_all_skills(skills_dir: Path) -> list[StructuredTool]:
    """Load every <skills_dir>/<name>/SKILL.md as a StructuredTool."""
    if not skills_dir.exists():
        logger.warning("skills directory does not exist: %s", skills_dir)
        return []
    skills: list[StructuredTool] = []
    for item in sorted(skills_dir.iterdir()):
        if item.is_dir() and (item / "SKILL.md").exists():
            logger.info("loading skill: %s", item.name)
            skills.append(load_skill(item))
    return skills
