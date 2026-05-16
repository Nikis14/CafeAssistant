"""Tests for the in-memory parser cache."""

from taste_agent.browser.parser_cache import (
    get_trace,
    has_trace,
    host_of,
    save_trace,
)


def test_host_of_extracts_hostname():
    assert host_of("https://www.thefork.com/restaurant/iva-12345") == "www.thefork.com"


def test_host_of_handles_subdomain_and_port():
    assert host_of("http://api.example.com:8080/path") == "api.example.com:8080"


def test_save_and_get_trace_round_trip():
    trace = [
        ("navigate", {"url": "https://x.example/r"}),
        ("fill", {"selector": "input#name", "value": "Ana"}),
        ("click", {"selector": "button.next"}),
    ]
    save_trace("https://x.example/reserve", trace)
    assert get_trace("https://x.example/anything") == trace


def test_has_trace_true_after_save():
    save_trace("https://y.example/r", [("click", {"selector": "x"})])
    assert has_trace("https://y.example/somewhere") is True


def test_has_trace_false_for_unknown_host():
    assert has_trace("https://unknown.example/r") is False


def test_save_overrides_previous():
    save_trace("https://z.example/r", [("a", {})])
    save_trace("https://z.example/r2", [("b", {})])
    cached = get_trace("https://z.example/anything")
    assert cached == [("b", {})]
