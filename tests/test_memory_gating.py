"""Tests for memory gating: reflection should run only on durable memory signal."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from taste_agent.memory.gating import analyze_memory_relevance, render_window_for_reflection


def test_transactional_booking_reply_is_skipped():
    history = [
        AIMessage(content="I recommend Kafeterija Magazin 1907 and Przionica."),
        AIMessage(content="Which one would you like me to book?"),
    ]

    decision = analyze_memory_relevance(
        history,
        "Please book the first one for tomorrow at 8.",
        "I need your name and phone number.",
    )

    assert decision.should_reflect is False
    assert decision.transactional_only is True
    assert decision.task_clarification_only is True
    assert decision.reason == "task_reference_without_memory_signal"


def test_mixed_booking_turn_with_preference_still_reflects():
    history = [AIMessage(content="I can book that cafe for you.")]

    decision = analyze_memory_relevance(
        history,
        "Book it for tomorrow at 8, and I generally prefer quiet places.",
        "I need your name and phone number.",
    )

    assert decision.should_reflect is True
    assert decision.semantic_candidate is True
    assert decision.allow_clarification is True
    assert decision.reason == "durable_memory_signal_in_current_turn"


def test_reply_to_memory_clarification_triggers_reflection():
    history = [
        AIMessage(content="Before I forget — a quick question: Should I remember that you generally prefer quiet cafes?"),
    ]

    decision = analyze_memory_relevance(
        history,
        "Yes please.",
        "Noted.",
    )

    assert decision.should_reflect is True
    assert decision.semantic_candidate is True
    assert decision.allow_clarification is False
    assert decision.reason == "reply_to_memory_clarification"


def test_episodic_report_detected():
    history = [AIMessage(content="How was your visit to Kafeterija?")]

    decision = analyze_memory_relevance(
        history,
        "It was too noisy, but the coffee was great.",
        "Thanks, that helps.",
    )

    assert decision.should_reflect is True
    assert decision.episodic_candidate is True


def test_render_window_formats_roles():
    messages = [
        HumanMessage(content="I usually prefer quiet places."),
        AIMessage(content="Noted."),
    ]

    text = render_window_for_reflection(messages)

    assert "User: I usually prefer quiet places." in text
    assert "Assistant: Noted." in text
