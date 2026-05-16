"""Structured booking-flow schemas.

These models capture what the browser discovery/preparation flow has learned
about a reservation site. They are more reusable and inspectable than a raw
action trace and are the foundation for splitting:

1. booking-flow discovery
2. booking-flow instantiation with user-provided values
"""

from pydantic import BaseModel, Field


class BookingFlowStep(BaseModel):
    """One deterministic browser step in a discovered booking flow."""

    action: str = Field(..., description="Browser action name, e.g. navigate/click.")
    args: dict[str, object] = Field(
        default_factory=dict,
        description="Arguments passed to the backend action.",
    )


class BookingFieldSpec(BaseModel):
    """One input field the booking flow expects."""

    name: str = Field(..., description="Canonical field name, e.g. date or party_size.")
    type: str = Field(..., description="Field type, e.g. date/time/integer/text/phone.")
    selector: str = Field(..., description="Selector used to fill the field.")


class BookingFlowSpec(BaseModel):
    """A discovered booking flow, normalized for reuse across runs."""

    status: str = Field(
        ...,
        description="Discovery status, e.g. ok, partial_booking_flow, or no_online_booking.",
    )
    place_name: str = Field(..., description="Human-readable place name.")
    source_host: str = Field(..., description="Host this flow was learned from.")
    platform: str = Field(default="unknown", description="Booking platform or host family.")
    entry_url: str = Field(..., description="URL used to enter the booking flow.")
    final_form_url: str | None = Field(
        default=None,
        description="Best known final form URL, if discovery reached one.",
    )
    steps_to_form: list[BookingFlowStep] = Field(
        default_factory=list,
        description="Prefix steps needed before the first field-fill happens.",
    )
    required_fields: list[BookingFieldSpec] = Field(
        default_factory=list,
        description="Required inputs inferred from the discovered form.",
    )
    optional_fields: list[BookingFieldSpec] = Field(
        default_factory=list,
        description="Optional inputs inferred from the discovered form.",
    )
    submit_selector: str | None = Field(
        default=None,
        description="Selector for the irreversible final submit button, if known.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str = Field(default="", description="Free-form notes about the flow.")
