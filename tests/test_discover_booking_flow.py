from __future__ import annotations

import pytest

from taste_agent.browser.backend import MockBrowserBackend
from taste_agent.browser.spec_cache import get_spec, has_spec, save_spec
from taste_agent.browser.specs import BookingFlowSpec, BookingFlowStep
from taste_agent.browser.sub_agent import run_browser_discovery_subagent
from taste_agent.skills.reserve_table.reserve_table import _discover_impl, discover_booking_flow


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


def test_discover_booking_flow_accepts_non_booking_candidate_url(monkeypatch):
    observed: dict[str, str] = {}

    def fake_impl(**kwargs):
        observed["url"] = kwargs["reservation_url"]
        return {"status": "ok", "source": "discovery", "flow_spec": {}}

    monkeypatch.setattr(
        "taste_agent.skills.reserve_table.reserve_table._discover_impl",
        fake_impl,
    )

    result = discover_booking_flow(
        place_name="June Cafe",
        reservation_url="https://june-cafe.menu-world.com/",
    )

    assert result["status"] == "ok"
    assert observed["url"] == "https://june-cafe.menu-world.com/"


def test_discover_impl_ignores_negative_cached_spec(monkeypatch):
    backend = MockBrowserBackend()
    url = "https://june-cafe.menu-world.com/"
    save_spec(
        url,
        BookingFlowSpec(
            status="no_online_booking",
            place_name="June Cafe",
            source_host="june-cafe.menu-world.com",
            platform="unknown",
            entry_url=url,
            final_form_url=url,
            steps_to_form=[BookingFlowStep(action="navigate", args={"url": url})],
        ),
    )
    seen = {"calls": 0}

    def _fake_discovery(**_kwargs):
        seen["calls"] += 1
        return {
            "messages": [],
            "last_message_text": "No booking entrypoint found on this inspected page.",
            "actions": [("navigate", {"url": url})],
            "final_url": url,
            "final_dom": "<body>No reservations</body>",
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

    assert seen["calls"] == 1
    assert result["status"] == "no_online_booking"
    assert result["source"] == "discovery"
    assert has_spec(url) is False


def test_discover_impl_does_not_cache_partial_required_fields(monkeypatch):
    backend = MockBrowserBackend()
    url = "https://june-cafe.resos.com/booking"

    def _fake_discovery(**_kwargs):
        return {
            "messages": [],
            "last_message_text": "found partial form",
            "actions": [("navigate", {"url": url}), ("raw_html", {})],
            "final_url": url,
            "final_dom": "<form><input name='name' /><input name='phone' /></form>",
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

    assert result["status"] == "partial_booking_flow"
    assert result["required_fields"] == ["contact_name"]
    assert result["optional_fields"] == ["contact_phone"]
    assert result["missing_required_fields"] == ["date", "time", "party_size"]
    assert "Partial" in result["next_step"] or "partial" in result["next_step"]
    assert has_spec(url) is False


def test_discovery_subagent_blocks_cross_host_navigation_without_crashing(monkeypatch):
    from langchain_core.messages import AIMessage

    backend = MockBrowserBackend()

    class _FakeAgent:
        def __init__(self, tools):
            self._tools = {tool.name: tool for tool in tools}

        def invoke(self, _payload):
            self._tools["browser_navigate"].invoke(
                {"url": "https://june-cafe.menu-world.com/"}
            )
            self._tools["browser_page_context"].invoke({})
            self._tools["browser_navigate"].invoke(
                {"url": "https://www.junecafebkk.com/"}
            )
            return {"messages": [AIMessage(content="done")]}

    monkeypatch.setattr(
        "langchain.agents.create_agent",
        lambda _llm, tools, *_a, **_kw: _FakeAgent(tools),
    )

    result = run_browser_discovery_subagent(
        goal="Inspect June Cafe from the grounded page.",
        backend=backend,
        model_factory=_factory,
        initial_url="https://june-cafe.menu-world.com/",
    )
    assert result["last_message_text"] == "done"
    assert result["actions"] == [
        ("navigate", {"url": "https://june-cafe.menu-world.com/"}),
        ("raw_html", {}),
    ]
