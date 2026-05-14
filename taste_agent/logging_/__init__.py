from __future__ import annotations

import json
from typing import Any

from taste_agent.logging_.hierarchical import (
    HierarchicalFormatter,
    configure_logging,
    get_logger,
    make_prefix,
    trace,
)


def debug_print(name: str) -> None:
    print(name, flush=True)


def _debug_summarize(value: Any, *, max_string: int = 240, max_items: int = 5) -> Any:
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return f"{value[:max_string]}...<{len(value)} chars>"
    if isinstance(value, dict):
        items = list(value.items())
        summarized = {
            str(k): _debug_summarize(v, max_string=max_string, max_items=max_items)
            for k, v in items[:max_items]
        }
        if len(items) > max_items:
            summarized["..."] = f"{len(items) - max_items} more keys"
        return summarized
    if isinstance(value, (list, tuple)):
        summarized = [
            _debug_summarize(v, max_string=max_string, max_items=max_items)
            for v in value[:max_items]
        ]
        if len(value) > max_items:
            summarized.append(f"... {len(value) - max_items} more items")
        return summarized
    return value


def _debug_emit(stage: str, name: str, payload: dict[str, Any] | None = None) -> None:
    print(f"\n=== DEBUG {stage}: {name} ===", flush=True)
    if payload:
        print(
            json.dumps(
                _debug_summarize(payload),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            flush=True,
        )
    print("=== DEBUG END ===", flush=True)


def debug_enter(name: str, **kwargs: Any) -> None:
    _debug_emit("ENTER", name, kwargs or None)


def debug_exit(name: str, result: Any = None, **kwargs: Any) -> None:
    payload: dict[str, Any] = {}
    if result is not None:
        payload["result"] = result
    payload.update(kwargs)
    _debug_emit("EXIT", name, payload or None)


__all__ = [
    "HierarchicalFormatter",
    "configure_logging",
    "debug_enter",
    "debug_exit",
    "debug_print",
    "get_logger",
    "make_prefix",
    "trace",
]
