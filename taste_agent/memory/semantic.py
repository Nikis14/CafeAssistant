"""Semantic memory: SQLite-backed key-value store of durable user facts.

Why SQLite (not a vector store): semantic facts are categorical — the agent
needs an exact lookup ("does the user have a stated dietary restriction?"),
not a similarity search. SQLite is also trivial for students to inspect
during the seminar — they can open the .db file and see the rows.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from taste_agent.logging_ import get_logger, trace
from taste_agent.memory.schemas import SemanticFact

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    source      TEXT NOT NULL DEFAULT 'explicit',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)
"""


class SemanticMemory:
    """Key-value store of facts about the user, persisted to SQLite.

    Thread-safe via a single lock; the demo is single-user / single-thread,
    but Gradio handlers can interleave so the lock prevents a torn write.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # ── Reads ────────────────────────────────────────────────────────────────

    def read(self, key: str) -> SemanticFact | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM facts WHERE key = ?", (key,)).fetchone()
        return self._row_to_fact(row) if row else None

    def all(self) -> list[SemanticFact]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM facts ORDER BY updated_at DESC").fetchall()
        return [self._row_to_fact(r) for r in rows]

    def as_dict(self) -> dict[str, str]:
        """Shortcut: ``{key: value}`` for prompt injection."""
        return {f.key: f.value for f in self.all()}

    # ── Writes ───────────────────────────────────────────────────────────────

    def write(
        self,
        key: str,
        value: str,
        *,
        source: str = "explicit",
        confidence: float = 1.0,
    ) -> SemanticFact:
        """Upsert a fact. Returns the stored value."""
        now = datetime.now(timezone.utc).isoformat()
        with trace("memory:semantic:write", key=key), self._lock:
            existing = self._conn.execute(
                "SELECT created_at FROM facts WHERE key = ?", (key,)
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            self._conn.execute(
                """
                INSERT INTO facts(key, value, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (key, value, confidence, source, created_at, now),
            )
            self._conn.commit()
            logger.info("semantic fact written: %s=%s (source=%s)", key, value, source)
        return SemanticFact(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(now),
        )

    def delete(self, key: str) -> bool:
        """Remove a fact. Returns True if anything was deleted."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            self._conn.commit()
            removed = cur.rowcount > 0
        if removed:
            logger.info("semantic fact deleted: %s", key)
        return removed

    def clear(self) -> None:
        """Wipe all facts. Test-only."""
        with self._lock:
            self._conn.execute("DELETE FROM facts")
            self._conn.commit()

    # ── Conflict detection ───────────────────────────────────────────────────

    def detect_conflict(self, key: str, new_value: str) -> SemanticFact | None:
        """Return the existing fact if writing ``new_value`` would conflict.

        A conflict is: same key, different non-empty value, written by an
        explicit source (we don't override explicit user statements without
        signalling). Caller decides what to do with the conflict.
        """
        existing = self.read(key)
        if existing is None:
            return None
        if existing.value == new_value:
            return None
        if existing.source == "explicit":
            return existing
        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> SemanticFact:
        return SemanticFact(
            key=row["key"],
            value=row["value"],
            confidence=row["confidence"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def __repr__(self) -> str:
        return f"SemanticMemory(db_path={self._db_path!r})"


# ── Module-level default singletons, keyed by session id ────────────────────
# Phase 3 has one default session ("_default" from _session.py); Phase 4 will
# scope per real user/thread by setting the session-id ContextVar. The seam
# lives here so the call-site signatures don't change in Phase 4.

_DEFAULT: dict[str, SemanticMemory] = {}


def get_default() -> SemanticMemory:
    """Return the default ``SemanticMemory`` for the current session."""
    from taste_agent.memory._session import current_session_id

    sid = current_session_id()
    if sid not in _DEFAULT:
        _DEFAULT[sid] = SemanticMemory(db_path=":memory:")
    return _DEFAULT[sid]


def set_default(memory: SemanticMemory | None) -> None:
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
