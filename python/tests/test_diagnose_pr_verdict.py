"""Tests for shadow.diagnose_pr.risk — verdict logic + dangerous-tool
detection. These pin the spec §3.5 verdict mapping."""

from __future__ import annotations

from shadow.diagnose_pr.risk import classify_verdict, is_dangerous_violation


def test_zero_affected_is_ship() -> None:
    assert (
        classify_verdict(affected=0, total=10, has_dangerous_violation=False, has_severe_axis=False)
        == "ship"
    )


def test_zero_affected_with_dangerous_violation_still_stop() -> None:
    """A dangerous policy violation alone is enough for STOP, even
    if the 9-axis diff hasn't (yet) classified any trace as
    affected. This catches the case where the violation is in the
    candidate but the differ found no other axes moving."""
    assert (
        classify_verdict(affected=0, total=10, has_dangerous_violation=True, has_severe_axis=False)
        == "stop"
    )


def test_affected_traces_with_severe_axis_is_stop() -> None:
    assert (
        classify_verdict(affected=5, total=10, has_dangerous_violation=False, has_severe_axis=True)
        == "stop"
    )


def test_affected_traces_no_severe_no_dangerous_is_hold() -> None:
    assert (
        classify_verdict(affected=5, total=10, has_dangerous_violation=False, has_severe_axis=False)
        == "hold"
    )


def test_dangerous_violation_short_circuits_to_stop() -> None:
    assert (
        classify_verdict(affected=5, total=10, has_dangerous_violation=True, has_severe_axis=False)
        == "stop"
    )


def test_dangerous_keyword_in_tool_name() -> None:
    """v1 keyword fallback: refund / issue_refund / wire_transfer
    are all flagged as dangerous when the rule is severity error."""
    rule = {"params": {"tool": "issue_refund"}, "severity": "error", "tags": []}
    assert is_dangerous_violation(rule) is True


def test_explicit_tags_dangerous_marks_dangerous() -> None:
    rule = {"params": {"tool": "harmless_tool"}, "severity": "error", "tags": ["dangerous"]}
    assert is_dangerous_violation(rule) is True


def test_low_severity_violation_not_dangerous_even_with_dangerous_tool() -> None:
    rule = {"params": {"tool": "issue_refund"}, "severity": "info", "tags": []}
    assert is_dangerous_violation(rule) is False


def test_must_call_before_uses_then_field_for_keyword_match() -> None:
    """In must_call_before rules, the dangerous tool is `then`, not
    `tool`. We have to look at both."""
    rule = {
        "kind": "must_call_before",
        "params": {"first": "confirm", "then": "issue_refund"},
        "severity": "error",
        "tags": [],
    }
    assert is_dangerous_violation(rule) is True
