"""Context-based gating for the reflection sub-agent.

Reflection should not run on every turn. This module inspects the latest user
turn with a short recent dialogue window for context, then decides whether the
turn contains durable memory signal worth sending to the reflection agent.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


_SHORT_TRANSACTIONAL_RE = re.compile(
    r"^\s*(yes|no|ok|okay|sure|book it|reserve it|that one|this one|the first one|the second one|tomorrow|tonight|for \d+|we are \d+)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(?:\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|tomorrow|tonight|today|next\s+\w+)\b",
    re.IGNORECASE,
)
_PARTY_SIZE_RE = re.compile(
    r"\b(?:for|party of|table for|we are)\s+\d{1,2}\b", re.IGNORECASE
)
_CONTACT_RE = re.compile(
    r"\b(?:my name is|call me|phone|number is|email is|reach me at)\b",
    re.IGNORECASE,
)
_BOOKING_RE = re.compile(
    r"\b(?:book|reserve|reservation|table|confirm|cancel)\b", re.IGNORECASE
)
_REFERENTIAL_RE = re.compile(
    r"\b(?:the first one|the second one|that one|this one|it|there|same one)\b",
    re.IGNORECASE,
)
_SEMANTIC_RE = re.compile(
    r"\b(?:i\s+(?:usually|generally|typically|normally|tend to)\b"
    r"|i\s+(?:prefer|like|love|hate|avoid|dislike)\b"
    r"|i\s+(?:am|i'm)\s+(?:vegetarian|vegan|allergic)\b"
    r"|i\s+need\b"
    r"|my\s+(?:budget|preference)\b"
    r"|for\s+work\s+i\s+need\b"
    r"|quiet\s+places\b|wifi\b|smoky\s+places\b)",
    re.IGNORECASE,
)
_EPISODIC_RE = re.compile(
    r"\b(?:last\s+(?:time|week|night|month)|yesterday|when\s+i\s+went|i\s+(?:went|visited|tried|had)\b|it\s+was\s+(?:too|very)\b|service\s+was\b|food\s+was\b|coffee\s+was\b)",
    re.IGNORECASE,
)
_MEMORY_CLARIFICATION_RE = re.compile(
    r"\b(?:should i remember|should i update your|do you generally prefer|should i note that|should i store)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryGateDecision:
    """Structured routing decision for the reflection layer."""

    should_reflect: bool
    allow_clarification: bool
    semantic_candidate: bool
    episodic_candidate: bool
    transactional_only: bool
    task_clarification_only: bool
    reason: str
    window_messages: list[BaseMessage]


def analyze_memory_relevance(
    history: list[BaseMessage],
    current_user_text: str,
    current_agent_text: str,
    *,
    max_window_messages: int = 6,
) -> MemoryGateDecision:
    """Classify whether this turn should trigger reflection.

    The current user turn is the anchor. A short prior dialogue window provides
    context for disambiguation, especially for short replies such as "yes" or
    "the first one".
    """
    window_messages = _build_window(
        history, current_user_text, current_agent_text, max_window_messages
    )
    last_assistant_text = _latest_assistant_text(history)
    user_text = current_user_text.strip()
    lowered = user_text.lower()

    memory_clarification_active = bool(
        last_assistant_text and _MEMORY_CLARIFICATION_RE.search(last_assistant_text)
    )
    transactional = _is_transactional_turn(user_text)
    semantic_candidate = _has_semantic_signal(user_text)
    episodic_candidate = _has_episodic_signal(user_text, history)

    if memory_clarification_active and _is_short_affirmation_or_correction(user_text):
        return MemoryGateDecision(
            should_reflect=True,
            allow_clarification=False,
            semantic_candidate=True,
            episodic_candidate=False,
            transactional_only=False,
            task_clarification_only=False,
            reason="reply_to_memory_clarification",
            window_messages=window_messages,
        )

    if semantic_candidate or episodic_candidate:
        return MemoryGateDecision(
            should_reflect=True,
            allow_clarification=True,
            semantic_candidate=semantic_candidate,
            episodic_candidate=episodic_candidate,
            transactional_only=False,
            task_clarification_only=False,
            reason="durable_memory_signal_in_current_turn",
            window_messages=window_messages,
        )

    if transactional and _REFERENTIAL_RE.search(lowered):
        return MemoryGateDecision(
            should_reflect=False,
            allow_clarification=False,
            semantic_candidate=False,
            episodic_candidate=False,
            transactional_only=True,
            task_clarification_only=True,
            reason="task_reference_without_memory_signal",
            window_messages=window_messages,
        )

    if transactional:
        return MemoryGateDecision(
            should_reflect=False,
            allow_clarification=False,
            semantic_candidate=False,
            episodic_candidate=False,
            transactional_only=True,
            task_clarification_only=False,
            reason="transactional_only_turn",
            window_messages=window_messages,
        )

    return MemoryGateDecision(
        should_reflect=False,
        allow_clarification=False,
        semantic_candidate=False,
        episodic_candidate=False,
        transactional_only=False,
        task_clarification_only=False,
        reason="no_durable_memory_signal",
        window_messages=window_messages,
    )


def render_window_for_reflection(messages: list[BaseMessage]) -> str:
    """Serialize a short dialogue window for the reflection prompt."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            continue
        text = _message_text(msg).strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _build_window(
    history: list[BaseMessage],
    current_user_text: str,
    current_agent_text: str,
    max_window_messages: int,
) -> list[BaseMessage]:
    relevant = [m for m in history if isinstance(m, HumanMessage | AIMessage)]
    relevant = relevant[-max_window_messages:]
    relevant.append(HumanMessage(content=current_user_text))
    relevant.append(AIMessage(content=current_agent_text))
    return relevant


def _latest_assistant_text(history: list[BaseMessage]) -> str:
    for msg in reversed(history):
        if isinstance(msg, AIMessage):
            return _message_text(msg).strip()
    return ""


def _is_transactional_turn(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return bool(
        _SHORT_TRANSACTIONAL_RE.match(normalized)
        or _TIME_RE.search(normalized)
        or _PARTY_SIZE_RE.search(normalized)
        or _CONTACT_RE.search(normalized)
        or _BOOKING_RE.search(normalized)
    )


def _has_semantic_signal(text: str) -> bool:
    return bool(_SEMANTIC_RE.search(text))


def _has_episodic_signal(text: str, history: list[BaseMessage]) -> bool:
    if _EPISODIC_RE.search(text):
        return True
    last_assistant = _latest_assistant_text(history).lower()
    return bool(
        text.lower().startswith(("it was", "service was", "food was", "coffee was"))
        and ("how was" in last_assistant or "did you like" in last_assistant)
    )


def _is_short_affirmation_or_correction(text: str) -> bool:
    cleaned = text.strip().lower().rstrip(".!?")
    if cleaned in {"yes", "yes please", "please do", "sure", "no", "nope"}:
        return True
    return len(cleaned.split()) <= 6 and bool(_SEMANTIC_RE.search(cleaned))


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)
