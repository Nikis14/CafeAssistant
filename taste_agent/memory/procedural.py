"""Procedural memory: inferred behavioral patterns, derived periodically.

Distinct from semantic memory:

- **Semantic** = facts the user *stated* (dietary: vegetarian).
- **Procedural** = patterns we *inferred* from their behavior over time
  ("Prefers small intimate places — 5/6 visits rated ≥4").

Storage is SQLite (same pattern as ``semantic.py``) but the schema is simpler
— patterns are free text with confidence + evidence-count metadata. The
``derive`` job rewrites the table wholesale each time.

We also persist a tiny ``meta`` table holding ``last_derive_episode_count``
so the orchestrator can decide whether enough new episodes have accumulated
to warrant another derivation.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from taste_agent.logging_ import get_logger
from taste_agent.memory.schemas import InferredPattern

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    text            TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 1.0,
    evidence_count  INTEGER NOT NULL DEFAULT 1,
    derived_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_META_LAST_DERIVE_COUNT = "last_derive_episode_count"


class ProceduralMemory:
    """SQLite-backed store of inferred behavioral patterns.

    Single-source-of-truth: ``replace_all(patterns)`` is the only way to
    update the patterns table. Individual upserts would let stale patterns
    accumulate without a re-derivation; we want a clean snapshot per run.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Reads ────────────────────────────────────────────────────────────────

    def all(self) -> list[InferredPattern]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM patterns ORDER BY confidence DESC, evidence_count DESC"
            ).fetchall()
        return [self._row_to_pattern(r) for r in rows]

    def as_text(self) -> str:
        """Pre-rendered text block for prompt injection."""
        patterns = self.all()
        if not patterns:
            return ""
        return "\n".join(
            f"- {p.text} (confidence {p.confidence:.2f}, evidence {p.evidence_count})"
            for p in patterns
        )

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM patterns").fetchone()
        return int(row["n"])

    def last_derive_episode_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (_META_LAST_DERIVE_COUNT,)
            ).fetchone()
        return int(row["value"]) if row else 0

    # ── Writes ───────────────────────────────────────────────────────────────

    def replace_all(self, patterns: list[InferredPattern]) -> None:
        """Wholesale replace the patterns table.

        Why wholesale: derivation runs over the full episodic history and
        produces a fresh consistent snapshot. Merging would risk keeping
        stale patterns that the latest evidence no longer supports.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute("DELETE FROM patterns")
            for p in patterns:
                self._conn.execute(
                    """
                    INSERT INTO patterns(text, confidence, evidence_count, derived_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (p.text, p.confidence, p.evidence_count, p.derived_at or now),
                )
            self._conn.commit()
            logger.info("procedural patterns replaced: %d row(s)", len(patterns))

    def set_last_derive_episode_count(self, count: int) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO meta(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (_META_LAST_DERIVE_COUNT, str(count)),
            )
            self._conn.commit()

    def clear(self) -> None:
        """Wipe patterns + meta. Test-only."""
        with self._lock:
            self._conn.execute("DELETE FROM patterns")
            self._conn.execute("DELETE FROM meta")
            self._conn.commit()

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_pattern(row: sqlite3.Row) -> InferredPattern:
        return InferredPattern(
            text=row["text"],
            confidence=row["confidence"],
            evidence_count=row["evidence_count"],
            derived_at=datetime.fromisoformat(row["derived_at"]),
        )

    def __repr__(self) -> str:
        return f"ProceduralMemory(db_path={self._db_path!r})"


# ── Module-level default singletons, keyed by session id ────────────────────

_DEFAULT: dict[str, ProceduralMemory] = {}


def get_default() -> ProceduralMemory:
    """Return the default ``ProceduralMemory`` for the current session."""
    from taste_agent.memory._session import current_session_id

    sid = current_session_id()
    if sid not in _DEFAULT:
        _DEFAULT[sid] = ProceduralMemory(db_path=":memory:")
    return _DEFAULT[sid]


def set_default(memory: ProceduralMemory | None) -> None:
    """Set / clear the default for the current session. ``None`` clears."""
    from taste_agent.memory._session import current_session_id

    sid = current_session_id()
    if memory is None:
        _DEFAULT.pop(sid, None)
    else:
        _DEFAULT[sid] = memory


def reset_all_sessions() -> None:
    """Test-only: drop every per-session singleton."""
    _DEFAULT.clear()
