from __future__ import annotations

from taste_agent.browser.backend import MockBrowserBackend
from taste_agent.browser.spec_cache import get_spec
from taste_agent.skills.reserve_table.reserve_table import _discover_impl


def _factory(_id: str):
    from tests.fakes import FakeAgentModel

    return FakeAgentModel(response="done")


def test_discover_impl_infers_spec_from_final_dom(monkeypatch):
    backend = MockBrowserBackend()
    url = "https://june-cafe.resos.com/booking"

    def _fake_discovery(**_kwargs):
        backend.navigate(url)
        backend.click("a.reserve")
        backend.expect_dom(
            """
            <form>
              <input name="date" />
              <input name="time" />
              <input name="party_size" />
              <input name="name" />
              <input name="phone" />
            </form>
            """
        )
        return {
            "messages": [],
            "last_message_text": "found booking form",
            "actions": list(backend.calls),
            "final_url": url,
            "final_dom": backend.dom_snapshot("body"),
        }

    monkeypatch.setattr(
        "taste_agent.skills.reserve_table.reserve_table.run_browser_discovery_subagent",
        _fake_discovery,
    )

    result = _discover_impl(
        place_name="June Cafe",
        reservation_url=url,
        backend=backend,
        model_factory=_factory,
    )

    assert result["status"] == "ok"
    assert result["source"] == "discovery"
    assert result["required_fields"] == [
        "date",
        "time",
        "party_size",
        "contact_name",
    ]
    assert result["optional_fields"] == ["contact_phone"]
    assert result["requirements_summary"] == (
        "date (YYYY-MM-DD), time (HH:MM), party size, name for the reservation"
    )
    assert "Ask the user only for any missing required details" in result["next_step"]
    spec = get_spec(url)
    assert spec is not None
    assert [field.name for field in spec.required_fields] == [
        "date",
        "time",
        "party_size",
        "contact_name",
    ]
    assert [field.name for field in spec.optional_fields] == ["contact_phone"]
