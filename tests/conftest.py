"""Shared pytest fixtures. Adds the project root to sys.path so tests can
import the ``taste_agent`` package regardless of pytest invocation cwd.
"""

from __future__ import annotations

import logging
import os
import sys
from io import StringIO
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Force episodic memory to use the deterministic fake embedding during tests.
# Production runs (app.py) leave this unset and pick up HuggingFace
# sentence-transformers instead. Set at module load so it precedes any
# ``EpisodicMemory`` construction triggered by test imports.
os.environ.setdefault("TASTE_AGENT_FAKE_EMBEDDING", "1")

# Skip the LLM judge in the output guardrail by default. Tests that want to
# exercise the judge path inject a deterministic fake JSON-returning model.
os.environ.setdefault("TASTE_AGENT_SKIP_OUTPUT_JUDGE", "1")

# Skip the reflection sub-agent by default. It costs an extra LLM call per
# turn and exercises tool-calling that needs a specialized fake; tests that
# want to verify reflection pass model_factory + skip_reflection=False.
os.environ.setdefault("TASTE_AGENT_SKIP_REFLECTION", "1")


@pytest.fixture
def capture_logs() -> StringIO:
    """Replace handlers on the ``taste_agent`` logger with a StringIO sink.

    Restores the original handlers + level + propagate after the test.
    """
    from taste_agent.logging_ import HierarchicalFormatter

    logger = logging.getLogger("taste_agent")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    for h in original_handlers:
        logger.removeHandler(h)

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(HierarchicalFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        for h in original_handlers:
            logger.addHandler(h)
        logger.setLevel(original_level)
        logger.propagate = original_propagate


@pytest.fixture(autouse=True)
def _reset_trace_depth():
    """Make sure the trace-depth contextvar starts at 0 for each test."""
    from taste_agent.logging_.hierarchical import _depth

    token = _depth.set(0)
    try:
        yield
    finally:
        _depth.reset(token)


@pytest.fixture(autouse=True)
def _reset_phase2_state():
    """Clear action-guardrail state, parser cache, and reserve_table backend.

    These are intentionally process-global for the demo. Tests must run from a
    clean slate or they leak state across each other.
    """
    from taste_agent.browser.parser_cache import clear_cache
    from taste_agent.browser.spec_cache import clear_spec_cache
    from taste_agent.guardrails import reset_action_state
    from taste_agent.skills.reserve_table import reserve_table as rt

    reset_action_state()
    clear_cache()
    clear_spec_cache()
    # Drop any backend set by a prior test
    rt._DEFAULT_BACKEND = None

    yield

    reset_action_state()
    clear_cache()
    clear_spec_cache()
    rt._DEFAULT_BACKEND = None


@pytest.fixture(autouse=True)
def _clear_env_pollution_from_dotenv():
    """``app.py`` calls ``load_dotenv()`` when imported. ``test_app.py``
    imports ``app``, which then leaks any keys from the developer's local
    ``.env`` into the rest of the test session — places_search tests that
    assume ``FOURSQUARE_API_KEY`` is unset (so the mock path runs) start
    hitting Foursquare with a fake key and failing.

    Pop the leaky keys before each test. Tests that need them set use
    ``monkeypatch.setenv`` themselves.
    """
    for var in ("FOURSQUARE_API_KEY", "TAVILY_API_KEY"):
        os.environ.pop(var, None)
    yield


@pytest.fixture(autouse=True)
def _reset_phase3_memory():
    """Reset semantic + episodic + procedural memory defaults across all sessions."""
    from taste_agent.memory import (
        reset_all_episodic_sessions,
        reset_all_procedural_sessions,
        reset_all_semantic_sessions,
    )

    reset_all_semantic_sessions()
    reset_all_episodic_sessions()
    reset_all_procedural_sessions()
    yield
    reset_all_semantic_sessions()
    reset_all_episodic_sessions()
    reset_all_procedural_sessions()


@pytest.fixture(autouse=True)
def _reset_orchestrator_graph():
    """Drop the compiled-graph singleton between tests.

    The graph closes over the node-function references at compile time, so a
    test that monkey-patches a node module-attribute would otherwise see
    nothing — the cached graph still holds the original. Cheap to rebuild;
    this fixture makes monkey-patching nodes "just work".
    """
    from taste_agent.orchestrator import reset_graph_cache

    reset_graph_cache()
    yield
    reset_graph_cache()
