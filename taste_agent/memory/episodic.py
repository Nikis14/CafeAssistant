"""Episodic memory: time-stamped user experiences, retrieved by similarity.

We store the human-readable ``summary`` field in Chroma for vector retrieval,
and the structured metadata (place, rating, date) alongside as Chroma
``metadatas``. Phase 3 ships an in-memory store; Phase 4 will persist to disk.

Default embedding selection:

- Production: HuggingFace ``sentence-transformers/all-MiniLM-L6-v2``. First
  call downloads ~80MB to the local HF cache, then runs fully offline.
- Tests: a deterministic fake so the suite is fast and offline. Activated by
  setting ``TASTE_AGENT_FAKE_EMBEDDING=1`` — ``tests/conftest.py`` sets this
  at collection time.

Callers can always override by passing ``embedding_function=`` to the
``EpisodicMemory`` constructor.
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import date as date_cls
from typing import Any

from taste_agent.logging_ import get_logger, trace
from taste_agent.memory.schemas import EpisodicEvent

logger = get_logger(__name__)

_FAKE_EMBEDDING_ENV = "TASTE_AGENT_FAKE_EMBEDDING"


def _default_embedding() -> Any:
    """Build the default embedding.

    Returns ``DeterministicFakeEmbedding`` when ``TASTE_AGENT_FAKE_EMBEDDING=1``
    (tests / offline demos), otherwise HuggingFace sentence-transformers.
    """
    if os.environ.get(_FAKE_EMBEDDING_ENV) == "1":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=64)
    # Lazy import: HuggingFace is heavy and many tests never need it.
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("loading HuggingFace embedding 'sentence-transformers/all-MiniLM-L6-v2'")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class EpisodicMemory:
    """Vector store of logged user experiences.

    Thin wrapper around Chroma. The point of the wrapper is to enforce the
    ``EpisodicEvent`` schema at the boundary so the rest of the codebase
    doesn't need to know about Chroma's metadata-flattening quirks.
    """

    def __init__(
        self,
        embedding_function: Any | None = None,
        collection_name: str = "episodic",
        persist_directory: str | None = None,
    ) -> None:
        from langchain_chroma import Chroma

        self._lock = threading.Lock()
        self._embedding = embedding_function or _default_embedding()
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embedding,
            persist_directory=persist_directory,
        )

    # ── Writes ───────────────────────────────────────────────────────────────

    def log(self, event: EpisodicEvent) -> str:
        """Persist an event. Returns the assigned document id."""
        if event.date is None:
            event = event.model_copy(update={"date": date_cls.today().isoformat()})
        doc_id = uuid.uuid4().hex
        text = f"{event.place_name}: {event.notes}"
        metadata = _flatten_metadata(event.model_dump())
        with trace("memory:episodic:log", place=event.place_name), self._lock:
            self._store.add_texts(texts=[text], metadatas=[metadata], ids=[doc_id])
            logger.info("episodic event logged: %s [%s]", event.place_name, doc_id)
        return doc_id

    # ── Reads ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[EpisodicEvent]:
        """Vector-similarity search. Returns up to ``k`` events ranked by relevance."""
        with trace("memory:episodic:search", query=query[:60], k=k), self._lock:
            docs = self._store.similarity_search(query=query, k=k)
        events = [_event_from_metadata(doc.metadata) for doc in docs]
        logger.info("episodic search: %r -> %d hits", query[:40], len(events))
        return events

    def count(self) -> int:
        with self._lock:
            return self._store._collection.count()

    def list_recent(self, k: int = 5) -> list[EpisodicEvent]:
        """Return the ``k`` most recently-logged events, ordered by date desc.

        Falls back to insertion order if dates are absent. This is the right
        accessor for "what did I do lately" — unlike ``search``, which orders
        by vector similarity to a query.
        """
        with self._lock:
            data = self._store._collection.get()
        metadatas = data.get("metadatas") or []
        events = [_event_from_metadata(m) for m in metadatas]
        # Sort by date desc; treat missing date as oldest.
        events.sort(key=lambda e: e.date or "", reverse=True)
        return events[:k]

    def clear(self) -> None:
        """Drop all stored events. Test-only."""
        with self._lock:
            ids = self._store._collection.get()["ids"]
            if ids:
                self._store.delete(ids=ids)


# ── Module-level default singletons, keyed by session id ────────────────────

_DEFAULT: dict[str, EpisodicMemory] = {}


def get_default() -> EpisodicMemory:
    """Return the default ``EpisodicMemory`` for the current session."""
    from taste_agent.memory._session import current_session_id

    sid = current_session_id()
    if sid not in _DEFAULT:
        _DEFAULT[sid] = EpisodicMemory()
    return _DEFAULT[sid]


def set_default(memory: EpisodicMemory | None) -> None:
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


# ── Internal ─────────────────────────────────────────────────────────────────


def _flatten_metadata(d: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str / int / float / bool. Drop None and
    convert anything else to str."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _event_from_metadata(meta: dict[str, Any]) -> EpisodicEvent:
    """Round-trip flattened metadata back into an ``EpisodicEvent``."""
    return EpisodicEvent(
        place_name=str(meta.get("place_name", "")),
        notes=str(meta.get("notes", "")),
        rating=int(meta["rating"]) if "rating" in meta and meta["rating"] is not None else None,
        date=meta.get("date"),
        address=meta.get("address"),
        cuisine=meta.get("cuisine"),
    )
