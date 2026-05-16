"""Tests for the hierarchical logger."""

from taste_agent.logging_ import get_logger, make_prefix, trace
from taste_agent.logging_.hierarchical import current_depth


def test_make_prefix_at_depth_0():
    assert make_prefix(0) == ""


def test_make_prefix_at_depth_1():
    assert make_prefix(1) == "|- "


def test_make_prefix_at_depth_2():
    assert make_prefix(2) == "   |- "


def test_make_prefix_at_depth_3():
    assert make_prefix(3) == "      |- "


def test_top_level_log_has_no_tree_prefix(capture_logs):
    get_logger().info("hello world")
    out = capture_logs.getvalue()
    assert "hello world" in out
    # The portion before the message should not contain a branch character
    before_msg = out.split("hello world")[0]
    assert "|-" not in before_msg


def test_trace_logs_entry_at_current_depth(capture_logs):
    with trace("parent"):
        pass
    out = capture_logs.getvalue()
    assert "parent" in out
    # The trace entry itself is logged at depth 0 — no prefix
    assert "|- parent" not in out


def test_trace_indents_child_logs(capture_logs):
    with trace("parent"):
        get_logger().info("child")
    out = capture_logs.getvalue()
    assert "parent" in out
    assert "|- child" in out


def test_nested_traces_indent_progressively(capture_logs):
    with trace("level1"), trace("level2"):
        get_logger().info("deepest")
    out = capture_logs.getvalue()
    assert "|- level2" in out
    assert "   |- deepest" in out


def test_trace_restores_depth_on_exit():
    assert current_depth() == 0
    with trace("ephemeral"):
        assert current_depth() == 1
    assert current_depth() == 0


def test_trace_restores_depth_on_exception():
    assert current_depth() == 0
    try:
        with trace("boom"):
            raise RuntimeError("intentional")
    except RuntimeError:
        pass
    assert current_depth() == 0


def test_trace_includes_extras_in_message(capture_logs):
    with trace("with_extras", model="claude", n=3):
        pass
    out = capture_logs.getvalue()
    assert "with_extras" in out
    assert "model=claude" in out
    assert "n=3" in out
