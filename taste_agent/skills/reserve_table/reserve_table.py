"""reserve_table skill — drives a browser sub-agent to fill a reservation form.

The public ``run`` function is what the SKILL.md loader exposes to the
orchestrator agent. ``_run_impl`` is the testable inner function that accepts
a backend and model factory.

The skill never actually clicks the final submit. It stops at
``request_user_approval`` and returns a pending status. Finalization is the
orchestrator's job, gated by ``taste_agent.guardrails.action``.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

from taste_agent.browser.backend import BrowserBackend, MockBrowserBackend
from taste_agent.browser.parser_cache import (
    ActionTrace,
    format_trace,
    get_trace,
    has_trace,
    host_of,
    save_trace,
)
from taste_agent.browser.spec_cache import delete_spec, get_spec, save_spec
from taste_agent.browser.specs import BookingFieldSpec, BookingFlowSpec, BookingFlowStep
from taste_agent.browser.sub_agent import (
    _ValueBoundFillBackend,
    run_browser_discovery_subagent,
    run_browser_subagent,
)
from taste_agent.config import ALLOW_RUNTIME_MOCKS, DEFAULT_MODEL_ID
from taste_agent.guardrails.action import (
    consume,
    gate_action,
    get,
    register_pending,
)
from taste_agent.logging_ import debug_enter, debug_exit, get_logger, trace

logger = get_logger(__name__)

# Default selector for the final submit button. Real Playwright + Phase 4 will
# either rely on cached selectors or have the sub-agent report it explicitly.
_DEFAULT_SUBMIT_SELECTOR = "button.confirm-reservation"
_BACKEND_NOT_CONFIGURED_ERROR = "Browser automation is not configured for this environment."

# Module-level backend default. The orchestrator (or tests) can swap it via
# ``set_default_backend``. When unset, production code falls through to the
# BrowserBackendPool — see ``_checkout_backend_for_flow`` below.
_DEFAULT_BACKEND: BrowserBackend | None = None

# action_id → backend pinned to a multi-turn reservation flow. ``run()`` adds
# an entry when a flow reaches ``pending_approval``; ``finalize_reservation``
# and ``cancel_reservation`` consume it. The pinned backend is returned to the
# pool on consumption.
_FLOW_BACKENDS: dict[str, BrowserBackend] = {}


def _chat_model_kwargs(model_id: str) -> dict[str, Any]:
    """Return LiteLLM kwargs that are compatible with the target model."""
    normalized = model_id.lower()
    if normalized.startswith("openai/gpt-5"):
        return {}
    return {"temperature": 0.2}


def set_default_backend(backend: BrowserBackend) -> None:
    """Override the module-level backend used by ``run`` / ``finalize_reservation``.

    Primarily a test injection seam. When set, both ``run`` and
    ``finalize_reservation`` skip the BrowserBackendPool entirely and use this
    shared backend across all calls.
    """
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = backend


def _get_default_backend() -> BrowserBackend:
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        if not ALLOW_RUNTIME_MOCKS:
            raise RuntimeError(_BACKEND_NOT_CONFIGURED_ERROR)
        _DEFAULT_BACKEND = MockBrowserBackend()
    return _DEFAULT_BACKEND


def _checkout_backend_for_flow() -> tuple[BrowserBackend, bool]:
    """Acquire a backend for a new reservation flow.

    Returns ``(backend, is_pooled)``. ``is_pooled=True`` means the caller is
    responsible for pinning (multi-turn) or releasing (one-shot) via the pool;
    ``is_pooled=False`` means the backend is a shared singleton (test default
    or auto-created mock) and the caller does nothing on release.
    """
    if _DEFAULT_BACKEND is not None:
        return _DEFAULT_BACKEND, False
    from taste_agent.browser.pool import get_browser_pool

    pool = get_browser_pool()
    if pool is not None:
        return pool.checkout(), True
    return _get_default_backend(), False


def _checkin_pooled_backend(backend: BrowserBackend) -> None:
    """Return a pool-borrowed backend. No-op for non-pool backends."""
    from taste_agent.browser.pool import get_browser_pool

    pool = get_browser_pool()
    if pool is not None and backend is not _DEFAULT_BACKEND:
        pool.checkin(backend)


@contextlib.contextmanager
def _release_pinned_on_exit(pinned: BrowserBackend | None) -> Iterator[None]:
    """Release ``pinned`` back to the browser pool on context exit, if pinned."""
    try:
        yield
    finally:
        if pinned is not None:
            _checkin_pooled_backend(pinned)


def _default_model_factory(model_id: str) -> BaseChatModel:
    """Same lazy-LiteLLM factory pattern as the main orchestrator."""
    from langchain_litellm import ChatLiteLLM

    return ChatLiteLLM(model=model_id, **_chat_model_kwargs(model_id))


def _format_goal(
    *,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
) -> str:
    phone_part = f", phone {contact_phone}" if contact_phone else ""
    return (
        f"Reserve a table at {place_name}. The reservation page is at "
        f"{reservation_url}. Details: date {date}, time {time}, party size "
        f"{party_size}, name {contact_name}{phone_part}. Fill the form and "
        f"call request_user_approval when the form is ready to submit. Do not "
        f"click the final submit button."
    )


def _format_summary(
    *,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
) -> str:
    phone_part = f", phone {contact_phone}" if contact_phone else ""
    host = host_of(reservation_url)
    return (
        f"Reserve at {place_name} (via {host}): {date} {time}, party of {party_size}, "
        f"name {contact_name}{phone_part}"
    )


_PLACEHOLDER_CONTACT_NAMES = {"user", "guest", "customer", "test", "unknown"}
_RESERVATION_PATH_HINTS = (
    "reserve",
    "reservation",
    "reservations",
    "booking",
    "book",
    "table",
    "rezerw",
)


def _validate_booking_inputs(
    *,
    reservation_url: str,
    contact_name: str,
) -> str | None:
    """Reject obviously invented or unsafe booking inputs.

    This is intentionally lightweight: it catches clear placeholders and
    homepage URLs masquerading as reservation pages, while leaving the fuller
    discovery architecture for a later step.
    """
    parsed = urlparse(reservation_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "reservation_url must be a full grounded URL."

    normalized_name = contact_name.strip().lower()
    if normalized_name in _PLACEHOLDER_CONTACT_NAMES:
        return "contact_name looks like a placeholder; ask the user for their real name."

    path = (parsed.path or "").strip("/").lower()
    if not path or path == "index.html":
        return (
            "reservation_url looks like a homepage, not a booking page. "
            "Find a grounded reservation page before calling reserve_table."
        )
    if not any(hint in path for hint in _RESERVATION_PATH_HINTS):
        return (
            "reservation_url does not look like a reservation page. "
            "Find a grounded booking URL before calling reserve_table."
        )
    return None


def _validate_discovery_url(url: str) -> str | None:
    """Validate a grounded candidate page for booking-flow discovery.

    Discovery is intentionally broader than final reservation preparation: it
    may start from a menu page, an official site, or another candidate entry
    point and then click through to the booking flow.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "reservation_url must be a full grounded URL."
    return None


_REPLAYABLE_ACTIONS = {"navigate", "click", "fill", "wait_for", "dom_snapshot", "raw_html"}

_REQUIRED_FIELD_ORDER = ("date", "time", "party_size", "contact_name")
_OPTIONAL_FIELDS = {"contact_phone"}
_FIELD_REQUEST_LABELS = {
    "date": "date (YYYY-MM-DD)",
    "time": "time (HH:MM)",
    "party_size": "party size",
    "contact_name": "name for the reservation",
    "contact_email": "email address",
    "contact_phone": "contact phone (optional)",
    "country_dial_code": "country dial code",
    "terms_acceptance": "acceptance of the terms",
}


class _MissingFieldExtractionPayload(BaseModel):
    needs_user_input: bool
    missing_required_fields: list[str]
    required_field_prompts: list[str]
    message: str = ""


def _has_prepare_ready_required_fields(fields: list[BookingFieldSpec]) -> bool:
    names = {field.name for field in fields}
    return all(name in names for name in _REQUIRED_FIELD_ORDER)


def _missing_required_field_names(fields: list[BookingFieldSpec]) -> list[str]:
    names = {field.name for field in fields}
    return [name for name in _REQUIRED_FIELD_ORDER if name not in names]


def _infer_platform(host: str) -> str:
    lowered = host.lower()
    if "resos" in lowered:
        return "resos"
    if "opentable" in lowered:
        return "opentable"
    if "sevenrooms" in lowered:
        return "sevenrooms"
    return "unknown"


def _infer_field_from_selector(selector: str) -> tuple[str, str] | None:
    lowered = selector.lower()
    if "date" in lowered:
        return ("date", "date")
    if "time" in lowered:
        return ("time", "time")
    if "party" in lowered or "guest" in lowered or "people" in lowered:
        return ("party_size", "integer")
    if "email" in lowered or "e-mail" in lowered:
        return ("contact_email", "email")
    if "phone" in lowered or "tel" in lowered:
        return ("contact_phone", "phone")
    if "countrydialcode" in lowered or "dialcode" in lowered:
        return ("country_dial_code", "text")
    if "term" in lowered:
        return ("terms_acceptance", "checkbox")
    if "name" in lowered:
        return ("contact_name", "text")
    return None


def _infer_flow_spec(
    *,
    place_name: str,
    reservation_url: str,
    actions: ActionTrace,
) -> BookingFlowSpec:
    """Infer a first-cut flow spec from a successful pre-submit action trace.

    This is an incremental bridge: the current browser sub-agent still emits a
    raw trace, but we immediately normalize the learned structure into a spec
    so later steps can shift from trace-replay to spec-driven preparation.
    """
    host = host_of(reservation_url)
    first_fill_index = next(
        (idx for idx, (action_name, _) in enumerate(actions) if action_name == "fill"),
        len(actions),
    )
    steps_to_form = [
        BookingFlowStep(action=action_name, args=dict(args))
        for action_name, args in actions[:first_fill_index]
    ]

    fields_by_name: dict[str, BookingFieldSpec] = {}
    final_form_url = reservation_url

    for action_name, args in actions:
        if action_name == "navigate":
            url = str(args.get("url", "")).strip()
            if url:
                final_form_url = url
        if action_name != "fill":
            continue
        selector = str(args.get("selector", "")).strip()
        inferred = _infer_field_from_selector(selector)
        if inferred is None:
            continue
        field_name, field_type = inferred
        fields_by_name.setdefault(
            field_name,
            BookingFieldSpec(name=field_name, type=field_type, selector=selector),
        )

    required_fields = [
        fields_by_name[name] for name in _REQUIRED_FIELD_ORDER if name in fields_by_name
    ]
    optional_fields = [
        fields_by_name[name] for name in sorted(_OPTIONAL_FIELDS) if name in fields_by_name
    ]
    is_prepare_ready = _has_prepare_ready_required_fields(required_fields)
    has_any_fields = bool(required_fields or optional_fields)
    status = (
        "ok"
        if is_prepare_ready
        else "partial_booking_flow"
        if has_any_fields
        else "no_online_booking"
    )

    return BookingFlowSpec(
        status=status,
        place_name=place_name,
        source_host=host,
        platform=_infer_platform(host),
        entry_url=reservation_url,
        final_form_url=final_form_url,
        steps_to_form=steps_to_form,
        required_fields=required_fields,
        optional_fields=optional_fields,
        submit_selector=_DEFAULT_SUBMIT_SELECTOR if is_prepare_ready else None,
        confidence=0.8 if is_prepare_ready else 0.55 if has_any_fields else 0.3,
        notes=(
            "Inferred from a successful pre-submit browser action trace."
            if is_prepare_ready
            else "Partial booking flow inferred from browser action trace; some earlier required steps or fields are still missing."
            if has_any_fields
            else "No reliable booking flow could be inferred from the browser action trace."
        ),
    )


def _infer_fields_from_dom(dom: str) -> list[BookingFieldSpec]:
    fields_by_name: dict[str, BookingFieldSpec] = {}
    selector_patterns = [
        r"""(?:input|select|textarea)[^>]*name=['"]([^'"]+)['"]""",
        r"""(?:input|select|textarea)[^>]*id=['"]([^'"]+)['"]""",
    ]
    for pattern in selector_patterns:
        for match in re.finditer(pattern, dom, flags=re.IGNORECASE):
            attr_value = match.group(1)
            selector = f"""input[name='{attr_value}']"""
            inferred = _infer_field_from_selector(attr_value)
            if inferred is None:
                inferred = _infer_field_from_selector(selector)
            if inferred is None:
                continue
            field_name, field_type = inferred
            fields_by_name.setdefault(
                field_name,
                BookingFieldSpec(name=field_name, type=field_type, selector=selector),
            )
    return [fields_by_name[name] for name in _REQUIRED_FIELD_ORDER if name in fields_by_name] + [
        fields_by_name[name] for name in sorted(_OPTIONAL_FIELDS) if name in fields_by_name
    ]


def _spec_from_discovery(
    *,
    place_name: str,
    reservation_url: str,
    actions: ActionTrace,
    final_url: str,
    final_dom: str,
) -> BookingFlowSpec:
    required_and_optional = _infer_fields_from_dom(final_dom)
    required_fields = [
        field for field in required_and_optional if field.name in _REQUIRED_FIELD_ORDER
    ]
    optional_fields = [field for field in required_and_optional if field.name in _OPTIONAL_FIELDS]
    is_prepare_ready = _has_prepare_ready_required_fields(required_fields)
    has_any_fields = bool(required_fields or optional_fields)
    return BookingFlowSpec(
        status="ok"
        if is_prepare_ready
        else "partial_booking_flow"
        if has_any_fields
        else "no_online_booking",
        place_name=place_name,
        source_host=host_of(final_url or reservation_url),
        platform=_infer_platform(host_of(final_url or reservation_url)),
        entry_url=reservation_url,
        final_form_url=final_url or reservation_url,
        steps_to_form=[BookingFlowStep(action=name, args=dict(args)) for name, args in actions],
        required_fields=required_fields,
        optional_fields=optional_fields,
        submit_selector=_DEFAULT_SUBMIT_SELECTOR if is_prepare_ready else None,
        confidence=0.7 if is_prepare_ready else 0.45 if has_any_fields else 0.3,
        notes=(
            "Inferred from browser discovery before user value collection."
            if is_prepare_ready
            else "Partial booking flow detected during discovery; some earlier required steps or fields were not confidently recovered."
            if has_any_fields
            else "No reliable online booking form was discovered."
        ),
    )


def _booking_values(
    *,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
) -> dict[str, str]:
    return {
        "date": date,
        "time": time,
        "party_size": str(party_size),
        "contact_name": contact_name,
        "contact_phone": contact_phone,
    }


def _known_user_fill_values(*raw_values: object) -> set[str]:
    """Return the exact non-empty values we are allowed to type into the form.

    This is intentionally value-based, not schema-based: the browser agent may
    discover arbitrary fields on arbitrary sites, but it must never invent
    user data that was not explicitly supplied or deterministically normalized
    from supplied values.
    """
    allowed: set[str] = set()
    for value in raw_values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        allowed.add(text)
    return allowed


def _field_request_label(field_name: str) -> str:
    return _FIELD_REQUEST_LABELS.get(field_name, field_name.replace("_", " "))


def _parse_missing_field_payload(text: str) -> _MissingFieldExtractionPayload:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in missing-field extractor output")
    return _MissingFieldExtractionPayload.model_validate(json.loads(text[start : end + 1]))


def _fallback_missing_field_extraction(last_message_text: str) -> dict[str, Any] | None:
    lowered = last_message_text.lower()
    if not any(token in lowered for token in ("need", "required", "provide", "missing", "proceed")):
        return None

    candidates: list[tuple[str, tuple[str, ...]]] = [
        ("contact_email", ("email", "e-mail")),
        ("contact_phone", ("phone", "telephone", "mobile")),
        ("terms_acceptance", ("terms", "accept the terms", "acceptance")),
        ("country_dial_code", ("country code", "dial code")),
        ("contact_name", ("name",)),
    ]
    missing_required_fields: list[str] = []
    for field_name, needles in candidates:
        if any(needle in lowered for needle in needles):
            missing_required_fields.append(field_name)

    if not missing_required_fields:
        return None

    required_field_prompts = [_field_request_label(name) for name in missing_required_fields]
    return {
        "status": "needs_user_input",
        "source": "agentic_missing_info",
        "missing_required_fields": missing_required_fields,
        "required_field_prompts": required_field_prompts,
        "message": last_message_text.strip(),
        "next_step": "Ask the user for: " + ", ".join(required_field_prompts) + ".",
    }


def _extract_missing_fields_from_tool_messages(messages: list[Any]) -> dict[str, Any] | None:
    missing_required_fields: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue
        lowered = content.lower()
        if "could not fill " not in lowered:
            continue
        if (
            "blank values are not allowed" not in lowered
            and "blank value" not in lowered
            and "user did not provide that booking detail" not in lowered
        ):
            continue
        match = re.search(r"could not fill\s+(.+?):", content, flags=re.IGNORECASE)
        if not match:
            continue
        selector = match.group(1).strip()
        inferred = _infer_field_from_selector(selector)
        if inferred is None:
            continue
        field_name, _field_type = inferred
        if field_name not in missing_required_fields:
            missing_required_fields.append(field_name)

    if not missing_required_fields:
        return None

    required_field_prompts = [_field_request_label(name) for name in missing_required_fields]
    return {
        "status": "needs_user_input",
        "source": "agentic_missing_info",
        "missing_required_fields": missing_required_fields,
        "required_field_prompts": required_field_prompts,
        "message": "The form still needs additional user-provided details before it is ready for approval.",
        "next_step": "Ask the user for: " + ", ".join(required_field_prompts) + ".",
    }


def _extract_missing_user_input(
    *,
    last_message_text: str,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
    model_factory: Callable[[str], BaseChatModel],
    model_id: str,
) -> dict[str, Any] | None:
    text = last_message_text.strip()
    if not text:
        return None
    lowered = text.lower()
    if not any(token in lowered for token in ("need", "required", "provide", "missing", "proceed")):
        return None

    prompt = (
        "You are extracting whether a browser booking agent has explicitly asked "
        "for additional user-provided information before it can request final "
        "approval.\n"
        "Return strict JSON with shape:\n"
        '{"needs_user_input": true|false, "missing_required_fields": ["snake_case_field"], '
        '"required_field_prompts": ["user-facing prompt"], "message": "..."}\n'
        "Only return needs_user_input=true if the message explicitly says the "
        "agent still needs the user to provide or confirm some missing detail.\n"
        "Known booking details already supplied by the user:\n"
        f"- place_name: {place_name}\n"
        f"- reservation_url: {reservation_url}\n"
        f"- date: {date}\n"
        f"- time: {time}\n"
        f"- party_size: {party_size}\n"
        f"- contact_name: {contact_name}\n"
        f"- contact_phone: {contact_phone or '(not provided)'}\n\n"
        f"Sub-agent final message:\n{text}"
    )
    try:
        llm = model_factory(model_id)
        raw = llm.invoke([HumanMessage(content=prompt)])
        content = raw.content if isinstance(raw.content, str) else str(raw.content)
        payload = _parse_missing_field_payload(content)
        if not payload.needs_user_input:
            return None
        prompts = payload.required_field_prompts or [
            _field_request_label(name) for name in payload.missing_required_fields
        ]
        return {
            "status": "needs_user_input",
            "source": "agentic_missing_info",
            "missing_required_fields": payload.missing_required_fields,
            "required_field_prompts": prompts,
            "message": payload.message or text,
            "next_step": "Ask the user for: " + ", ".join(prompts) + ".",
        }
    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        logger.warning("missing-field extractor parse/validate failed: %s", e)
    except Exception as e:  # pragma: no cover
        logger.warning("missing-field extractor failed: %s", e)

    return _fallback_missing_field_extraction(text)


def _post_submit_looks_successful(html: str) -> bool:
    lowered = html.lower()
    success_markers = (
        "thank you",
        "reservation confirmed",
        "booking confirmed",
        "booking request received",
        "reservation request received",
        "your reservation is confirmed",
        "your booking is confirmed",
        "confirmation",
        "we have received your booking request",
    )
    return any(marker in lowered for marker in success_markers)


def _post_submit_still_looks_like_form(html: str, submit_selector: str) -> bool:
    lowered = html.lower()
    submit_hint = submit_selector.lower() if submit_selector else ""
    form_markers = (
        "input name=",
        "textarea",
        'type="submit"',
        "type='submit'",
        "terms",
        "aria-invalid",
        "muiinputbase-input",
        "booking request",
        "your contact information",
    )
    return any(marker in lowered for marker in form_markers) or (
        submit_hint and submit_hint in lowered
    )


def _form_fingerprint(html: str) -> tuple[str, ...]:
    selectors: set[str] = set()
    patterns = [
        r"""<(?:input|select|textarea)[^>]*name=['"]([^'"]+)['"]""",
        r"""<(?:input|select|textarea)[^>]*id=['"]([^'"]+)['"]""",
        r"""<button[^>]*type=['"]submit['"][^>]*>(.*?)</button>""",
        r"""<button[^>]*aria-label=['"]([^'"]+)['"]""",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            value = " ".join(match.group(1).split()).strip().lower()
            if value:
                selectors.add(value)
    return tuple(sorted(selectors))


def _post_submit_still_looks_like_same_form(pre_html: str, post_html: str) -> bool:
    before = set(_form_fingerprint(pre_html))
    after = set(_form_fingerprint(post_html))
    if not before or not after:
        return False
    overlap = before & after
    return len(overlap) >= min(3, len(before), len(after))


def _build_discovery_payload(
    *,
    flow_spec: BookingFlowSpec,
    source: str,
    message: str = "",
) -> dict[str, Any]:
    required_names = [field.name for field in flow_spec.required_fields]
    optional_names = [field.name for field in flow_spec.optional_fields]
    missing_required_names = _missing_required_field_names(flow_spec.required_fields)
    required_prompts = [_field_request_label(name) for name in required_names]
    optional_prompts = [_field_request_label(name) for name in optional_names]
    missing_required_prompts = [_field_request_label(name) for name in missing_required_names]

    next_step = ""
    if flow_spec.status == "ok" and required_prompts:
        next_step = "Ask the user only for any missing required details from this list: "
        next_step += ", ".join(required_prompts)
        if optional_prompts:
            next_step += ". Optionally ask for " + ", ".join(optional_prompts) + "."
        else:
            next_step += "."
    elif flow_spec.status == "partial_booking_flow":
        next_step = (
            "A partial online booking flow was detected, but the recovered form understanding is incomplete. "
            "Tell the agent that discovery likely reached a later step without reliably recovering earlier required steps."
        )
        if required_prompts:
            next_step += " Recovered fields: " + ", ".join(required_prompts) + "."
        if missing_required_prompts:
            next_step += (
                " Missing earlier required fields/steps: "
                + ", ".join(missing_required_prompts)
                + "."
            )
    elif flow_spec.status != "ok":
        next_step = (
            "No reliable online booking form was discovered. Offer to keep looking "
            "or suggest another place with a booking page."
        )

    payload: dict[str, Any] = {
        "status": flow_spec.status,
        "source": source,
        "flow_spec": flow_spec.model_dump(),
        "required_fields": required_names,
        "optional_fields": optional_names,
        "missing_required_fields": missing_required_names,
        "required_field_prompts": required_prompts,
        "optional_field_prompts": optional_prompts,
        "requirements_summary": ", ".join(required_prompts) if required_prompts else "",
        "next_step": next_step,
    }
    if message:
        payload["message"] = message
    return payload


def _rediscovery_goal_after_submit_failure(
    *,
    place_name: str,
    reservation_url: str,
    failure_reason: str,
) -> str:
    return (
        f"Re-evaluate the online reservation flow for {place_name}. You are on "
        f"{reservation_url}. A submit attempt or late-stage interaction did not "
        f"complete successfully. Failure feedback: {failure_reason}. Inspect the "
        "current rendered page, identify any validation errors, missing required "
        "fields, checkboxes, or earlier steps that still block submission, and "
        "determine what additional information is needed before retrying. Do not "
        "fill or submit anything."
    )


def _discovery_goal(*, place_name: str, reservation_url: str) -> str:
    return (
        f"Discover the online reservation flow for {place_name}. Start from "
        f"{reservation_url}. Find the booking form and determine which fields "
        "are required before any user-specific values are entered. Do not fill "
        "or submit anything."
    )


def _prepare_from_spec(
    *,
    flow_spec: BookingFlowSpec,
    backend: BrowserBackend,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
) -> dict[str, Any]:
    """Instantiate a discovered booking flow with current user values.

    This is the first real split between discovery and preparation. The spec
    tells us how to reach the form and which selectors to fill; the current
    user-provided values are injected here, not copied from an old trace.
    """
    with trace(
        "skill:reserve_table:prepare_from_spec",
        n_steps=len(flow_spec.steps_to_form),
        n_required=len(flow_spec.required_fields),
    ):
        for step in flow_spec.steps_to_form:
            action_name = step.action
            args = step.args
            if action_name not in _REPLAYABLE_ACTIONS:
                logger.error(
                    "unknown cached spec step %r for %s — refusing to prepare",
                    action_name,
                    reservation_url,
                )
                return {
                    "status": "failed",
                    "error": f"cached flow spec contains unknown action {action_name!r}; "
                    "cache may be stale — rediscover the booking flow",
                }
            if action_name == "navigate":
                backend.navigate(str(args.get("url", "")))
            elif action_name == "click":
                backend.click(str(args.get("selector", "")))
            elif action_name == "wait_for":
                timeout_ms = args.get("timeout_ms", 5000)
                backend.wait_for(
                    str(args.get("selector", "")),
                    timeout_ms=int(timeout_ms) if isinstance(timeout_ms, int) else 5000,
                )
            elif action_name == "dom_snapshot":
                backend.dom_snapshot(args.get("selector"))  # type: ignore[arg-type]
            elif action_name == "raw_html":
                backend.raw_html()
            elif action_name == "fill":
                backend.fill(str(args.get("selector", "")), str(args.get("value", "")))

        values = _booking_values(
            date=date,
            time=time,
            party_size=party_size,
            contact_name=contact_name,
            contact_phone=contact_phone,
        )
        for field in flow_spec.required_fields:
            backend.fill(field.selector, values[field.name])
        for field in flow_spec.optional_fields:
            value = values.get(field.name, "")
            if value:
                backend.fill(field.selector, value)

        summary = _format_summary(
            place_name=place_name,
            reservation_url=reservation_url,
            date=date,
            time=time,
            party_size=party_size,
            contact_name=contact_name,
            contact_phone=contact_phone,
        )
        action_id = register_pending(
            tool_name="confirm_reservation",
            summary=summary,
            args={
                "source": "spec",
                "submit_selector": flow_spec.submit_selector or _DEFAULT_SUBMIT_SELECTOR,
                "place_name": place_name,
                "reservation_url": reservation_url,
            },
        )
        return {
            "status": "pending_approval",
            "action_id": action_id,
            "summary": summary,
            "source": "spec",
            "flow_spec": flow_spec.model_dump(),
        }


def _discover_impl(
    *,
    place_name: str,
    reservation_url: str,
    backend: BrowserBackend,
    model_factory: Callable[[str], BaseChatModel],
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    cached_spec = get_spec(reservation_url)
    if cached_spec is not None and cached_spec.status == "ok":
        return _build_discovery_payload(flow_spec=cached_spec, source="cached_spec")
    if cached_spec is not None:
        delete_spec(reservation_url)

    result = run_browser_discovery_subagent(
        goal=_discovery_goal(place_name=place_name, reservation_url=reservation_url),
        backend=backend,
        model_factory=model_factory,
        model_id=model_id,
        initial_url=reservation_url,
    )
    flow_spec = _spec_from_discovery(
        place_name=place_name,
        reservation_url=reservation_url,
        actions=result.get("actions", []),
        final_url=str(result.get("final_url", reservation_url)),
        final_dom=str(result.get("final_dom", "")),
    )
    if flow_spec.status == "ok":
        save_spec(reservation_url, flow_spec)
    return _build_discovery_payload(
        flow_spec=flow_spec,
        source="discovery",
        message=result.get("last_message_text", ""),
    )


def _replay_cached(
    *,
    cached_trace: ActionTrace,
    backend: BrowserBackend,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
) -> dict[str, Any]:
    """Replay a cached action sequence.

    Phase 2 replays verbatim — the summary returned to the user reflects the
    *new* arguments so they can detect mismatches and cancel. Phase 4 will
    parameterize cached traces so a single trace works for any date/party.

    Refuses to replay if the cache contains an unknown action — forces
    re-discovery via the sub-agent path rather than risk a partial submit.
    """
    with trace("skill:reserve_table:replay_cached", n_actions=len(cached_trace)):
        # Pre-flight: any unknown action means the cache is stale / corrupt.
        # Better to fall back to a fresh sub-agent run than risk a partial form.
        for action_name, _ in cached_trace:
            if action_name not in _REPLAYABLE_ACTIONS:
                logger.error(
                    "unknown cached action %r in trace for %s — refusing to replay",
                    action_name,
                    reservation_url,
                )
                return {
                    "status": "failed",
                    "error": f"cached trace contains unknown action {action_name!r}; "
                    "cache may be stale — clear and retry",
                }

        for action_name, args in cached_trace:
            if action_name == "navigate":
                backend.navigate(str(args.get("url", "")))
            elif action_name == "click":
                backend.click(str(args.get("selector", "")))
            elif action_name == "fill":
                backend.fill(str(args.get("selector", "")), str(args.get("value", "")))
            elif action_name == "wait_for":
                timeout_ms = args.get("timeout_ms", 5000)
                backend.wait_for(
                    str(args.get("selector", "")),
                    timeout_ms=int(timeout_ms) if isinstance(timeout_ms, int) else 5000,
                )
            elif action_name == "dom_snapshot":
                backend.dom_snapshot(args.get("selector"))  # type: ignore[arg-type]
            elif action_name == "raw_html":
                backend.raw_html()

        summary = _format_summary(
            place_name=place_name,
            reservation_url=reservation_url,
            date=date,
            time=time,
            party_size=party_size,
            contact_name=contact_name,
            contact_phone=contact_phone,
        )
        action_id = register_pending(
            tool_name="confirm_reservation",
            summary=summary,
            args={
                "source": "cached",
                "submit_selector": _DEFAULT_SUBMIT_SELECTOR,
                "place_name": place_name,
                "reservation_url": reservation_url,
            },
        )
        cached_spec = get_spec(reservation_url)
        return {
            "status": "pending_approval",
            "action_id": action_id,
            "summary": summary,
            "source": "cached",
            "flow_spec": cached_spec.model_dump() if cached_spec else None,
        }


def _run_impl(
    *,
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str,
    backend: BrowserBackend,
    model_factory: Callable[[str], BaseChatModel],
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    with trace(
        "skill:reserve_table",
        place=place_name,
        url=reservation_url,
        cached=has_trace(reservation_url),
    ):
        # Defense in depth: forbid the submit selector at the backend level
        # *before* any LLM-driven activity touches the browser. If the
        # sub-agent goes off-prompt, the backend itself refuses the click.
        # The forbid is lifted only by ``finalize_reservation`` (after the
        # deterministic gate passes) or ``cancel_reservation``.
        backend.forbidden_selectors.add(_DEFAULT_SUBMIT_SELECTOR)

        cached_spec = get_spec(reservation_url)
        if (
            cached_spec is not None
            and cached_spec.status == "ok"
            and _has_prepare_ready_required_fields(cached_spec.required_fields)
        ):
            return _prepare_from_spec(
                flow_spec=cached_spec,
                backend=backend,
                place_name=place_name,
                reservation_url=reservation_url,
                date=date,
                time=time,
                party_size=party_size,
                contact_name=contact_name,
                contact_phone=contact_phone,
            )
        if cached_spec is not None and not _has_prepare_ready_required_fields(
            cached_spec.required_fields
        ):
            delete_spec(reservation_url)

        cached = get_trace(reservation_url)
        if cached is not None:
            return _replay_cached(
                cached_trace=cached,
                backend=backend,
                place_name=place_name,
                reservation_url=reservation_url,
                date=date,
                time=time,
                party_size=party_size,
                contact_name=contact_name,
                contact_phone=contact_phone,
            )

        goal = _format_goal(
            place_name=place_name,
            reservation_url=reservation_url,
            date=date,
            time=time,
            party_size=party_size,
            contact_name=contact_name,
            contact_phone=contact_phone,
        )
        guarded_backend = _ValueBoundFillBackend(
            backend,
            allowed_fill_values=_known_user_fill_values(
                date,
                time,
                party_size,
                contact_name,
                contact_phone,
            ),
        )

        try:
            result = run_browser_subagent(
                goal=goal,
                backend=guarded_backend,
                model_factory=model_factory,
                model_id=model_id,
            )
        except Exception:
            # Sub-agent crashed — drop the forbid so the backend is reusable.
            backend.forbidden_selectors.discard(_DEFAULT_SUBMIT_SELECTOR)
            raise

        # We don't pin to a specific id here; the sub-agent is the only thing
        # that registers a pending action in this code path.
        pending_after = None
        from taste_agent.guardrails.action import get_pending

        pending_after = get_pending()
        if pending_after is None:
            logger.warning("sub-agent finished without registering approval")
            backend.forbidden_selectors.discard(_DEFAULT_SUBMIT_SELECTOR)
            extracted = _extract_missing_user_input(
                last_message_text=str(result.get("last_message_text", "")),
                place_name=place_name,
                reservation_url=reservation_url,
                date=date,
                time=time,
                party_size=party_size,
                contact_name=contact_name,
                contact_phone=contact_phone,
                model_factory=model_factory,
                model_id=model_id,
            )
            if extracted is not None:
                extracted["actions"] = result.get("actions", [])
                return extracted
            return {
                "status": "failed",
                "error": "sub-agent finished without registering approval",
                "actions": result.get("actions", []),
            }

        actions = result.get("actions", [])
        extracted_from_tools = _extract_missing_fields_from_tool_messages(
            list(result.get("messages", []))
        )
        if extracted_from_tools is not None:
            consume(pending_after.action_id)
            backend.forbidden_selectors.discard(_DEFAULT_SUBMIT_SELECTOR)
            extracted_from_tools["actions"] = actions
            return extracted_from_tools
        logger.info(
            "browser recipe discovered for host=%s\n%s",
            host_of(reservation_url),
            format_trace(actions),
        )
        save_trace(reservation_url, actions)
        flow_spec = _infer_flow_spec(
            place_name=place_name,
            reservation_url=reservation_url,
            actions=actions,
        )
        if flow_spec.status == "ok":
            save_spec(reservation_url, flow_spec)

        return {
            "status": "pending_approval",
            "action_id": pending_after.action_id,
            "summary": pending_after.summary,
            "source": "agentic",
            "n_actions": len(actions),
            "flow_spec": flow_spec.model_dump(),
        }


def run(
    place_name: str,
    reservation_url: str,
    date: str,
    time: str,
    party_size: int,
    contact_name: str,
    contact_phone: str = "",
) -> dict[str, Any]:
    """Drive a browser sub-agent to fill a reservation form. STOPS before submit.

    Returns:
        Dict with ``status`` ("pending_approval" or "failed"). When pending,
        also includes ``action_id`` and ``summary`` for the user-approval flow.
    """
    debug_enter(
        "reserve_table.run",
        place_name=place_name,
        reservation_url=reservation_url,
        date=date,
        time=time,
        party_size=party_size,
        contact_name=contact_name,
        contact_phone=contact_phone,
    )
    validation_error = _validate_booking_inputs(
        reservation_url=reservation_url,
        contact_name=contact_name,
    )
    if validation_error is not None:
        logger.warning("reserve_table rejected ungrounded inputs: %s", validation_error)
        result = {"status": "failed", "error": validation_error, "source": "validation"}
        debug_exit("reserve_table.run", result=result)
        return result

    try:
        backend, is_pooled = _checkout_backend_for_flow()
    except RuntimeError as e:
        result = {"status": "failed", "error": str(e), "source": "configuration"}
        debug_exit("reserve_table.run", result=result)
        return result

    try:
        result = _run_impl(
            place_name=place_name,
            reservation_url=reservation_url,
            date=date,
            time=time,
            party_size=party_size,
            contact_name=contact_name,
            contact_phone=contact_phone,
            backend=backend,
            model_factory=_default_model_factory,
            model_id=DEFAULT_MODEL_ID,
        )
    except Exception:
        if is_pooled:
            _checkin_pooled_backend(backend)
        raise

    # Pin the backend to this action_id only when it reached approval; the
    # next turn's finalize_reservation/cancel_reservation will release it.
    # Failure paths return the backend immediately so a pool slot doesn't leak.
    if is_pooled:
        if result.get("status") == "pending_approval":
            action_id = str(result.get("action_id", ""))
            if action_id:
                _FLOW_BACKENDS[action_id] = backend
            else:
                _checkin_pooled_backend(backend)
        else:
            _checkin_pooled_backend(backend)

    debug_exit("reserve_table.run", result=result)
    return result


def discover_booking_flow(
    place_name: str,
    reservation_url: str,
) -> dict[str, Any]:
    """Explore a grounded candidate page and cache a reusable booking flow spec."""
    debug_enter(
        "reserve_table.discover_booking_flow",
        place_name=place_name,
        reservation_url=reservation_url,
    )
    validation_error = _validate_discovery_url(reservation_url)
    if validation_error is not None:
        logger.warning("discover_booking_flow rejected ungrounded URL: %s", validation_error)
        result = {"status": "failed", "error": validation_error, "source": "validation"}
        debug_exit("reserve_table.discover_booking_flow", result=result)
        return result

    try:
        backend, is_pooled = _checkout_backend_for_flow()
    except RuntimeError as e:
        result = {"status": "failed", "error": str(e), "source": "configuration"}
        debug_exit("reserve_table.discover_booking_flow", result=result)
        return result

    try:
        result = _discover_impl(
            place_name=place_name,
            reservation_url=reservation_url,
            backend=backend,
            model_factory=_default_model_factory,
            model_id=DEFAULT_MODEL_ID,
        )
    finally:
        if is_pooled:
            _checkin_pooled_backend(backend)
    debug_exit("reserve_table.discover_booking_flow", result=result)
    return result


def finalize_reservation(
    action_id: str,
    backend: BrowserBackend | None = None,
    submit_selector: str = _DEFAULT_SUBMIT_SELECTOR,
) -> dict[str, Any]:
    """Click the final submit button. Gated by the action guardrail.

    Raises ``PermissionError`` if ``action_id`` has not been user-approved.
    Looks up the approval by ``action_id`` (not by "most recent pending") so a
    concurrent registration cannot trick us into displaying the wrong summary.
    Consumes the pending action on success.
    """
    debug_enter(
        "finalize_reservation",
        action_id=action_id,
        submit_selector=submit_selector,
    )
    # Prefer an explicit backend (test override), then the pinned backend from
    # ``run()``, then the legacy default. The pinned slot is consumed
    # unconditionally; ``_release_pinned_on_exit`` guarantees the pool slot is
    # freed regardless of finalization outcome.
    pinned: BrowserBackend | None = _FLOW_BACKENDS.pop(action_id, None) if backend is None else None
    with trace("finalize_reservation", action_id=action_id), _release_pinned_on_exit(pinned):
        gate_action(action_id, tool_name="confirm_reservation")
        bk = backend or pinned or _get_default_backend()
        # Look up by id, not "latest pending" — concurrent registrations are
        # possible and the gate validated *this* id, so use *this* summary.
        approval = get(action_id)
        summary = approval.summary if approval else "(unknown)"
        effective_submit_selector = submit_selector
        if approval and approval.args.get("submit_selector"):
            effective_submit_selector = str(approval.args["submit_selector"])
        place_name = str(approval.args.get("place_name", "")) if approval else ""
        recovery_url = str(approval.args.get("reservation_url", "")) if approval else ""
        if not effective_submit_selector:
            result = {
                "status": "failed",
                "action_id": action_id,
                "summary": summary,
                "error": "final submit selector was not captured during approval",
            }
            debug_exit("finalize_reservation", result=result)
            return result
        # Lift the forbid for this specific selector now that the gate passed,
        # then perform the irreversible action.
        bk.forbidden_selectors.discard(effective_submit_selector)
        try:
            pre_submit_html = bk.raw_html()
        except Exception:
            pre_submit_html = ""
        try:
            bk.click(effective_submit_selector)
        except Exception as e:
            failure_reason = str(e)
        else:
            failure_reason = ""
            try:
                post_submit_html = bk.raw_html()
            except Exception:
                post_submit_html = ""
            if not _post_submit_looks_successful(
                post_submit_html
            ) and _post_submit_still_looks_like_form(post_submit_html, effective_submit_selector):
                failure_reason = (
                    "submit click did not produce a recognizable confirmation state; "
                    "the page still looks like an unresolved booking form"
                )
            elif not _post_submit_looks_successful(
                post_submit_html
            ) and _post_submit_still_looks_like_same_form(pre_submit_html, post_submit_html):
                failure_reason = (
                    "submit click did not materially change the visible booking form; "
                    "the same required fields still appear after submission"
                )

        if failure_reason:
            current_url = bk.current_url()
            discovery_result = run_browser_discovery_subagent(
                goal=_rediscovery_goal_after_submit_failure(
                    place_name=place_name or "this venue",
                    reservation_url=recovery_url or current_url,
                    failure_reason=failure_reason,
                ),
                backend=bk,
                model_factory=_default_model_factory,
                model_id=DEFAULT_MODEL_ID,
                initial_url=current_url,
            )
            flow_spec = _spec_from_discovery(
                place_name=place_name or "this venue",
                reservation_url=recovery_url or current_url,
                actions=discovery_result.get("actions", []),
                final_url=str(discovery_result.get("final_url", current_url)),
                final_dom=str(discovery_result.get("final_dom", "")),
            )
            result = _build_discovery_payload(
                flow_spec=flow_spec,
                source="submit_recovery",
                message=discovery_result.get("last_message_text", ""),
            )
            consume(action_id)
            recovery_payload = {
                "status": "needs_rediscovery",
                "action_id": action_id,
                "summary": summary,
                "error": failure_reason,
                "recovery": result,
            }
            debug_exit("finalize_reservation", result=recovery_payload)
            return recovery_payload

        consume(action_id)
        logger.info("reservation finalized: %s", summary)
        result = {"status": "confirmed", "action_id": action_id, "summary": summary}
        debug_exit("finalize_reservation", result=result)
        return result


def cancel_reservation(
    action_id: str,
    backend: BrowserBackend | None = None,
    submit_selector: str = _DEFAULT_SUBMIT_SELECTOR,
) -> dict[str, Any]:
    """Discard a pending reservation without submitting.

    Lifts the backend's submit-forbid so the backend is reusable for a future
    reservation attempt.
    """
    debug_enter(
        "cancel_reservation",
        action_id=action_id,
        submit_selector=submit_selector,
    )
    pinned: BrowserBackend | None = _FLOW_BACKENDS.pop(action_id, None) if backend is None else None
    with trace("cancel_reservation", action_id=action_id), _release_pinned_on_exit(pinned):
        bk = backend or pinned or _get_default_backend()
        bk.forbidden_selectors.discard(submit_selector)
        consume(action_id)
        logger.info("reservation cancelled: %s", action_id)
        result = {"status": "cancelled", "action_id": action_id}
        debug_exit("cancel_reservation", result=result)
        return result
