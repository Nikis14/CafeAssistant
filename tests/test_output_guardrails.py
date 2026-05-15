"""Tests for the output guardrail — PII redaction + LLM-judge."""

from __future__ import annotations

import pytest

from taste_agent.guardrails.output import (
    OutputGuardrailResult,
    _parse_judge_output,
    redact_output_pii,
    rewrite_internal_error_leaks,
    run_output_guardrails,
)
from tests.fakes import FakeAgentModel

# ── PII redaction (deterministic) ────────────────────────────────────────────


def test_redact_pii_strips_email():
    cleaned, n, concerns = redact_output_pii("Email me at chef@iva.rs for details")
    assert "[REDACTED-EMAIL]" in cleaned
    assert "chef@iva.rs" not in cleaned
    assert n == 1
    assert concerns == ["email addresses (1)"]


def test_redact_pii_strips_phone():
    cleaned, n, _concerns = redact_output_pii("Call +381 64 123 4567 to book")
    assert "[REDACTED-PHONE]" in cleaned
    assert n == 1


def test_redact_pii_does_not_strip_iso_date_as_phone():
    cleaned, n, concerns = redact_output_pii(
        "Prepared for 2026-05-15 at 12:00 for 2 people."
    )
    assert cleaned == "Prepared for 2026-05-15 at 12:00 for 2 people."
    assert n == 0
    assert concerns == []


def test_redact_pii_strips_card():
    cleaned, n, _concerns = redact_output_pii("Use card 4111 1111 1111 1111")
    assert "[REDACTED-CARD]" in cleaned
    assert n == 1


def test_redact_pii_strips_leaked_input_tokens():
    """If the agent parrots back a [EMAIL]/[PHONE] token from the input
    redaction, the output guardrail must strip that too."""
    cleaned, n, _concerns = redact_output_pii("Thanks, I'll email [EMAIL] later.")
    assert "[EMAIL]" not in cleaned
    assert "[REDACTED]" in cleaned
    assert n == 1


def test_redact_pii_multiple_kinds():
    cleaned, n, _concerns = redact_output_pii(
        "Email a@b.com or call +381 64 123 4567"
    )
    assert n == 2
    assert "[REDACTED-EMAIL]" in cleaned
    assert "[REDACTED-PHONE]" in cleaned


def test_redact_pii_clean_text_unchanged():
    text = "Try Kafeterija in Belgrade — best flat white."
    cleaned, n, concerns = redact_output_pii(text)
    assert cleaned == text
    assert n == 0
    assert concerns == []


def test_rewrite_internal_error_leaks_drops_provider_details():
    text = (
        "The live Places search chunk is returning an authorization error, so I can’t pull "
        "up live options right now. I can still help with web-based picks."
    )
    cleaned, rewritten, concerns = rewrite_internal_error_leaks(text)
    assert rewritten is True
    assert "authorization error" not in cleaned.lower()
    assert "live Places search" not in cleaned
    assert "web-based picks" in cleaned
    assert concerns


def test_rewrite_internal_error_leaks_falls_back_to_generic_message():
    text = "TAVILY_API_KEY is not set."
    cleaned, rewritten, concerns = rewrite_internal_error_leaks(text)
    assert rewritten is True
    assert "TAVILY_API_KEY" not in cleaned
    assert "search sources" in cleaned
    assert concerns


# ── _parse_judge_output ──────────────────────────────────────────────────────


def test_parse_judge_output_clean_json():
    raw = '{"factuality_ok": true, "factuality_concerns": [], "citation_ok": true, "citation_concerns": []}'
    parsed = _parse_judge_output(raw)
    assert parsed["factuality_ok"] is True
    assert parsed["factuality_concerns"] == []


def test_parse_judge_output_strips_markdown_fence():
    raw = '```json\n{"factuality_ok": false, "factuality_concerns": ["fabricated place"], "citation_ok": true, "citation_concerns": []}\n```'
    parsed = _parse_judge_output(raw)
    assert parsed["factuality_ok"] is False
    assert "fabricated place" in parsed["factuality_concerns"]


def test_parse_judge_output_handles_prose_preamble():
    raw = 'Here is my judgement:\n{"factuality_ok": true, "factuality_concerns": [], "citation_ok": true, "citation_concerns": []}\nThanks.'
    parsed = _parse_judge_output(raw)
    assert parsed["factuality_ok"] is True


def test_parse_judge_output_raises_on_no_json():
    with pytest.raises(ValueError, match="no JSON"):
        _parse_judge_output("the response is fine")


# ── run_output_guardrails ────────────────────────────────────────────────────


def test_skip_judge_runs_pii_only():
    """skip_judge=True is the cheap path. PII still gets stripped."""
    result = run_output_guardrails(
        "Email chef@iva.rs", skip_judge=True
    )
    assert result.judge_skipped is True
    assert "[REDACTED-EMAIL]" in result.response_text
    assert result.pii_leaked == 1
    assert result.factuality_ok is True  # default when judge skipped


def test_run_output_guardrails_rewrites_internal_error_leaks():
    result = run_output_guardrails(
        "The live Places search chunk is returning an authorization error. I can still help with web-based picks.",
        skip_judge=True,
    )
    assert result.internal_error_rewritten is True
    assert "authorization error" not in result.response_text.lower()
    assert "web-based picks" in result.response_text


def test_no_model_factory_implies_skip_judge():
    """If no factory is provided, the judge is skipped (graceful fallback)."""
    result = run_output_guardrails("Try Iva.", model_factory=None)
    assert result.judge_skipped is True


def _judge_factory(judge_json: str):
    """Build a factory whose model returns a fixed judge response."""

    def factory(_id: str):
        return FakeAgentModel(response=judge_json)

    return factory


def test_judge_passes_through_factuality_concerns():
    factory = _judge_factory(
        '{"factuality_ok": false, "factuality_concerns": ["place name not in context"], "citation_ok": true, "citation_concerns": []}'
    )
    result = run_output_guardrails(
        "Try Atlantis in Belgrade.",
        context_summary="(no places in context)",
        model_factory=factory,
        skip_judge=False,
    )
    assert result.judge_skipped is False
    assert result.factuality_ok is False
    assert "place name not in context" in result.factuality_concerns
    assert result.has_concerns is True


def test_judge_passes_through_citation_concerns():
    factory = _judge_factory(
        '{"factuality_ok": true, "factuality_concerns": [], "citation_ok": false, "citation_concerns": ["claimed opening hours not in tool output"]}'
    )
    result = run_output_guardrails(
        "Open until midnight.",
        model_factory=factory,
        skip_judge=False,
    )
    assert result.citation_ok is False
    assert len(result.citation_concerns) == 1


def test_judge_parse_failure_does_not_raise():
    """A malformed judge response must not crash the turn."""
    factory = _judge_factory("definitely not json")
    result = run_output_guardrails(
        "Try Iva.", model_factory=factory, skip_judge=False
    )
    assert result.judge_skipped is False
    assert result.judge_error is not None
    # PII redaction still happened (or, in this case, wasn't needed)
    assert result.response_text == "Try Iva."


def test_pii_runs_before_judge_so_judge_sees_clean_text():
    """The judge should evaluate the redacted text, not the leaked version."""
    captured: list[str] = []

    def factory(_id: str):
        class _CapturingFake(FakeAgentModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                captured.append(messages[0].content)
                return super()._generate(messages, stop, run_manager, **kwargs)

        return _CapturingFake(
            response='{"factuality_ok": true, "factuality_concerns": [], "citation_ok": true, "citation_concerns": []}'
        )

    run_output_guardrails(
        "Email chef@iva.rs to book",
        model_factory=factory,
        skip_judge=False,
    )
    assert any("[REDACTED-EMAIL]" in c for c in captured)
    assert all("chef@iva.rs" not in c for c in captured)


# ── Pydantic validation of judge payload ────────────────────────────────────


def test_judge_rejects_string_booleans():
    """Schema drift: model returns "false" instead of false. Must NOT silently
    coerce to True. The original Codex finding."""
    factory = _judge_factory(
        '{"factuality_ok": "false", "factuality_concerns": ["x"], "citation_ok": true, "citation_concerns": []}'
    )
    result = run_output_guardrails(
        "Try Iva.", model_factory=factory, skip_judge=False
    )
    # Validation failed → judge_error is set, defaults are kept (factuality_ok=True)
    assert result.judge_skipped is False
    assert result.judge_error is not None
    assert "parse-error" in result.judge_error


def test_judge_rejects_non_list_concerns():
    factory = _judge_factory(
        '{"factuality_ok": false, "factuality_concerns": "single string not list", "citation_ok": true, "citation_concerns": []}'
    )
    result = run_output_guardrails(
        "Try Iva.", model_factory=factory, skip_judge=False
    )
    assert result.judge_error is not None


def test_judge_accepts_well_formed_payload():
    factory = _judge_factory(
        '{"factuality_ok": false, "factuality_concerns": ["fabricated place"], "citation_ok": true, "citation_concerns": []}'
    )
    result = run_output_guardrails(
        "Try Atlantis.", model_factory=factory, skip_judge=False
    )
    assert result.judge_error is None
    assert result.factuality_ok is False
    assert "fabricated place" in result.factuality_concerns


def test_judge_does_not_rewrite_bad_draft_before_returning_to_user():
    factory = _judge_factory(
        '{"factuality_ok": false, "factuality_concerns": ["fabricated place"], "citation_ok": true, "citation_concerns": []}'
    )
    result = run_output_guardrails(
        "Try Atlantis in Belgrade.",
        model_factory=factory,
        skip_judge=False,
    )
    assert result.judge_rewritten is False
    assert result.judge_rewrite_reason is None
    assert result.response_text == "Try Atlantis in Belgrade."


def test_judge_does_not_rewrite_unsupported_citation_claims():
    factory = _judge_factory(
        '{"factuality_ok": true, "factuality_concerns": [], "citation_ok": false, "citation_concerns": ["claimed opening hours not in tool output"]}'
    )
    result = run_output_guardrails(
        "June Cafe is open until midnight.",
        model_factory=factory,
        skip_judge=False,
    )
    assert result.judge_rewritten is False
    assert result.judge_rewrite_reason is None
    assert result.response_text == "June Cafe is open until midnight."


# ── resolve_judge_model_id (env-driven) ──────────────────────────────────────


def test_resolve_judge_skip_env_returns_none(monkeypatch):
    from taste_agent.guardrails.output import resolve_judge_model_id

    monkeypatch.setenv("TASTE_AGENT_SKIP_OUTPUT_JUDGE", "1")
    assert resolve_judge_model_id() is None


def test_resolve_judge_override_env_takes_precedence(monkeypatch):
    from taste_agent.guardrails.output import resolve_judge_model_id

    monkeypatch.delenv("TASTE_AGENT_SKIP_OUTPUT_JUDGE", raising=False)
    monkeypatch.setenv("TASTE_AGENT_JUDGE_MODEL_ID", "openai/gpt-5-mini")
    assert resolve_judge_model_id() == "openai/gpt-5-mini"


def test_resolve_judge_skips_when_only_openai_key_present(monkeypatch):
    from taste_agent.guardrails.output import resolve_judge_model_id

    monkeypatch.delenv("TASTE_AGENT_SKIP_OUTPUT_JUDGE", raising=False)
    monkeypatch.delenv("TASTE_AGENT_JUDGE_MODEL_ID", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_judge_model_id() is None


def test_resolve_judge_skips_when_only_anthropic_key_present(monkeypatch):
    from taste_agent.guardrails.output import resolve_judge_model_id

    monkeypatch.delenv("TASTE_AGENT_SKIP_OUTPUT_JUDGE", raising=False)
    monkeypatch.delenv("TASTE_AGENT_JUDGE_MODEL_ID", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    assert resolve_judge_model_id() is None


def test_resolve_judge_skips_when_no_key_and_no_override(monkeypatch):
    """If neither judge provider key is present and there's no override, the
    judge should auto-skip rather than crash inside LiteLLM."""
    from taste_agent.guardrails.output import resolve_judge_model_id

    monkeypatch.delenv("TASTE_AGENT_SKIP_OUTPUT_JUDGE", raising=False)
    monkeypatch.delenv("TASTE_AGENT_JUDGE_MODEL_ID", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_judge_model_id() is None


def test_run_output_guardrails_auto_skips_when_env_unresolved(monkeypatch):
    """No env hints at all → run_output_guardrails should skip the judge."""
    monkeypatch.delenv("TASTE_AGENT_SKIP_OUTPUT_JUDGE", raising=False)
    monkeypatch.delenv("TASTE_AGENT_JUDGE_MODEL_ID", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    factory = _judge_factory("ignored — judge should not be invoked")
    result = run_output_guardrails("Try Iva.", model_factory=factory)
    assert result.judge_skipped is True


def test_summary_for_debug_includes_all_fields():
    result = OutputGuardrailResult(
        response_text="ok",
        pii_leaked=1,
        pii_concerns=["email"],
        factuality_ok=False,
        factuality_concerns=["bad"],
    )
    s = result.summary_for_debug()
    assert s["pii_leaked"] == 1
    assert s["factuality_ok"] is False
    assert "factuality_concerns" in s
