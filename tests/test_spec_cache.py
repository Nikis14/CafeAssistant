"""Tests for the in-memory booking flow spec cache."""

from taste_agent.browser.spec_cache import get_spec, has_spec, save_spec
from taste_agent.browser.specs import BookingFieldSpec, BookingFlowSpec, BookingFlowStep


def test_save_and_get_spec_round_trip():
    spec = BookingFlowSpec(
        status="ok",
        place_name="June Cafe",
        source_host="june-cafe.resos.com",
        platform="resos",
        entry_url="https://june-cafe.resos.com/booking",
        final_form_url="https://june-cafe.resos.com/booking",
        steps_to_form=[
            BookingFlowStep(
                action="navigate",
                args={"url": "https://june-cafe.resos.com/booking"},
            )
        ],
        required_fields=[
            BookingFieldSpec(name="date", type="date", selector="input[name='date']")
        ],
    )
    save_spec("https://june-cafe.resos.com/booking", spec)
    cached = get_spec("https://june-cafe.resos.com/anything")
    assert cached is not None
    assert cached.place_name == "June Cafe"
    assert cached.required_fields[0].name == "date"


def test_has_spec_true_after_save():
    spec = BookingFlowSpec(
        status="ok",
        place_name="X",
        source_host="x.example",
        entry_url="https://x.example/book",
        final_form_url="https://x.example/book",
    )
    save_spec("https://x.example/book", spec)
    assert has_spec("https://x.example/elsewhere") is True
