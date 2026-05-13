"""Guardrails — hand-rolled for the seminar.

Pedagogical choice: every component here is plain Python (regex + dict +
set membership) so students see the mechanism, not a black-box wrapper.

In production you'd swap in one of:

  - Microsoft Presidio (https://microsoft.github.io/presidio/)
      Replace the PII regexes in ``input.py``. Drop-in: an analyzer call
      returns recognised entities you can redact. Much higher accuracy than
      regex for names, locations, dates of birth, etc.

  - NVIDIA NeMo Guardrails (https://github.com/NVIDIA/NeMo-Guardrails)
      Whole-pipeline framework. Declarative rules in Colang for input,
      output, and dialogue flow. Replaces this folder if you want one
      framework end-to-end.

  - LLM Guard (https://llm-guard.com/)
      Security-focused. Has a trained ``PromptInjection`` scanner that
      replaces our injection-detection heuristics. Also offers anonymizer,
      ban-substrings, code-injection, secrets scanners.

  - Guardrails AI (https://www.guardrailsai.com/) + Guardrails Hub
      Validator library for OUTPUT shape: factuality, schema conformance,
      profanity, etc. Best fit for the Phase 4 output guardrail.

  - Llama Guard / Prompt Guard (Meta)
      Trained classifier models for unsafe content. Call them like any LLM;
      good for the output-toxicity check.

  - LangChain ``moderation`` chain
      Thin wrapper around OpenAI's moderation endpoint. Quickest output-side
      toxicity gate when you're already using OpenAI.

Keep ``action.py`` hand-rolled regardless — the deterministic confirm-gate
is the centerpiece teaching moment, and frameworks would obscure the
LLM-judge-for-ambiguity / set-membership-for-irreversibility contrast.
"""

from taste_agent.guardrails.action import (
    ApprovalRequest,
    approve,
    consume,
    gate_action,
    get,
    get_pending,
    is_approved,
    register_pending,
    reset_action_state,
)
from taste_agent.guardrails.input import GuardrailResult, run_input_guardrails
from taste_agent.guardrails.output import (
    OutputGuardrailResult,
    redact_output_pii,
    run_output_guardrails,
)

__all__ = [
    "ApprovalRequest",
    "GuardrailResult",
    "OutputGuardrailResult",
    "approve",
    "consume",
    "gate_action",
    "get",
    "get_pending",
    "is_approved",
    "redact_output_pii",
    "register_pending",
    "reset_action_state",
    "run_input_guardrails",
    "run_output_guardrails",
]
