"""Tests for the SKILL.md loader."""

from pathlib import Path

import pytest

from taste_agent.config import SKILLS_DIR
from taste_agent.skill_loader import (
    _parse_frontmatter,
    load_all_skills,
    load_skill,
)

# ── Frontmatter parser ───────────────────────────────────────────────────────


def test_parse_frontmatter_valid():
    md = "---\nname: test\ndescription: a test\n---\nBody here\n"
    meta, body = _parse_frontmatter(md)
    assert meta == {"name": "test", "description": "a test"}
    assert body == "Body here"


def test_parse_frontmatter_missing_opening_delim_raises():
    with pytest.raises(ValueError, match="frontmatter"):
        _parse_frontmatter("no frontmatter here")


def test_parse_frontmatter_missing_closing_delim_raises():
    with pytest.raises(ValueError, match="closing"):
        _parse_frontmatter("---\nname: x\ndescription: y\nbody never delimited")


def test_parse_frontmatter_missing_required_fields_raises():
    with pytest.raises(ValueError, match="name"):
        _parse_frontmatter("---\ndescription: only one\n---\nbody")


# ── Skill loading ────────────────────────────────────────────────────────────


def test_load_places_search_skill():
    skill_dir = SKILLS_DIR / "places_search"
    tool = load_skill(skill_dir)
    assert tool.name == "places_search"
    assert "restaurant" in tool.description.lower() or "cafe" in tool.description.lower()
    # The body of SKILL.md should be embedded in the tool description so the
    # agent sees the full instructions.
    assert "When to use" in tool.description


def test_load_all_skills_finds_places_search():
    tools = load_all_skills(SKILLS_DIR)
    names = {t.name for t in tools}
    assert "places_search" in names


def test_load_all_skills_empty_dir(tmp_path: Path):
    assert load_all_skills(tmp_path) == []


def test_load_skill_missing_module_raises(tmp_path: Path):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: broken\ndescription: no module\n---\nBody\n")
    with pytest.raises(FileNotFoundError):
        load_skill(skill_dir)


def test_load_skill_module_without_run_raises(tmp_path: Path):
    skill_dir = tmp_path / "noisy"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: noisy\ndescription: no run fn\n---\nBody\n")
    (skill_dir / "noisy.py").write_text("x = 1\n")
    with pytest.raises(AttributeError, match="run"):
        load_skill(skill_dir)


def test_load_skill_returns_invokable_tool(tmp_path: Path):
    skill_dir = tmp_path / "echo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: echo\ndescription: echoes input\n---\nEcho skill body\n"
    )
    (skill_dir / "echo.py").write_text("def run(text: str) -> str:\n    return f'echo:{text}'\n")
    tool = load_skill(skill_dir)
    assert tool.invoke({"text": "hi"}) == "echo:hi"


def test_loader_module_is_canonical_not_a_duplicate():
    """Regression: the loader used to import skills under a synthetic name
    (``..._loaded``), producing a second module instance with its own
    module-level state. Now it must return the same instance as a direct
    import — otherwise ``set_default_backend`` and similar functions silently
    fork state between the agent path and direct callers (orchestrator)."""
    import importlib

    from taste_agent.skill_loader import _load_module

    direct = importlib.import_module("taste_agent.skills.reserve_table.reserve_table")
    via_loader = _load_module(SKILLS_DIR / "reserve_table")
    assert via_loader is direct
