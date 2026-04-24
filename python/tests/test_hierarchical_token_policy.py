"""Tests for the token-level + policy-level layers of hierarchical diff."""

from __future__ import annotations

from typing import Any

import pytest

from shadow.errors import ShadowConfigError
from shadow.hierarchical import (
    check_policy,
    load_policy,
    policy_diff,
    render_policy_diff,
    render_token_diff,
    token_diff,
)

# ---- fixtures -------------------------------------------------------------


def _response(
    input_tokens: int = 100,
    output_tokens: int = 50,
    thinking_tokens: int = 0,
    content: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    return {
        "kind": "chat_response",
        "payload": {
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
            },
            "content": content or [{"type": "text", "text": "ok"}],
            "stop_reason": stop_reason,
        },
    }


def _tool_use(name: str, args: dict | None = None) -> dict:
    return {
        "type": "tool_use",
        "id": f"call_{name}",
        "name": name,
        "input": args or {},
    }


# ---- token diff -----------------------------------------------------------


def test_token_diff_identical_returns_zero_shift() -> None:
    base = [_response(100, 50) for _ in range(5)]
    cand = [_response(100, 50) for _ in range(5)]
    out = token_diff(base, cand)
    assert out.pair_count == 5
    assert out.normalised_shift["input_tokens"] == 0.0
    assert out.normalised_shift["output_tokens"] == 0.0
    # All per-pair deltas are zero.
    for pair in out.worst_pairs:
        assert sum(abs(v) for v in pair.delta.values()) == 0


def test_token_diff_catches_output_blow_up() -> None:
    base = [_response(100, 50) for _ in range(5)]
    cand = [_response(100, 200) for _ in range(5)]  # 4x output tokens
    out = token_diff(base, cand)
    assert out.normalised_shift["output_tokens"] == pytest.approx(3.0)
    assert out.worst_pairs[0].delta["output_tokens"] == 150


def test_token_diff_worst_pairs_ordered() -> None:
    base = [_response(100, 50) for _ in range(3)]
    cand = [
        _response(100, 50),  # no change
        _response(500, 50),  # +400 input
        _response(100, 200),  # +150 output
    ]
    out = token_diff(base, cand)
    # Pair #1 (+400) should come before pair #2 (+150).
    assert out.worst_pairs[0].pair_index == 1
    assert out.worst_pairs[1].pair_index == 2


def test_token_diff_handles_empty_traces() -> None:
    out = token_diff([], [])
    assert out.pair_count == 0
    assert all(v == 0.0 for v in out.normalised_shift.values())


def test_token_diff_unequal_lengths_uses_min() -> None:
    base = [_response() for _ in range(3)]
    cand = [_response() for _ in range(5)]
    out = token_diff(base, cand)
    assert out.pair_count == 3


def test_token_diff_handles_baseline_zero_median() -> None:
    base = [_response(thinking_tokens=0) for _ in range(3)]
    cand = [_response(thinking_tokens=1000) for _ in range(3)]
    out = token_diff(base, cand)
    assert out.normalised_shift["thinking_tokens"] == float("inf")


def test_render_token_diff_has_expected_fields() -> None:
    base = [_response(100, 50) for _ in range(5)]
    cand = [_response(150, 50) for _ in range(5)]
    out = token_diff(base, cand)
    rendered = render_token_diff(out)
    assert "Token-level diff" in rendered
    assert "input_tokens" in rendered
    assert "+50.00%" in rendered


def test_token_diff_percentiles_correct() -> None:
    base = [_response(output_tokens=v) for v in [10, 20, 30, 40, 50]]
    cand = [_response(output_tokens=v) for v in [10, 20, 30, 40, 50]]
    out = token_diff(base, cand)
    assert out.baseline["output_tokens"].median == pytest.approx(30)
    # p95 across [10,20,30,40,50] linear-interp = 48
    assert out.baseline["output_tokens"].p95 == pytest.approx(48)
    assert out.baseline["output_tokens"].maximum == 50
    assert out.baseline["output_tokens"].total == 150


# ---- policy diff ----------------------------------------------------------


def test_load_policy_accepts_wrapped_and_bare() -> None:
    wrapped = {"rules": [{"id": "r1", "kind": "no_call", "params": {"tool": "bad"}}]}
    rules = load_policy(wrapped)
    assert rules[0].id == "r1"
    assert rules[0].kind == "no_call"
    bare = [{"id": "r1", "kind": "no_call", "params": {"tool": "bad"}}]
    rules2 = load_policy(bare)
    assert rules2[0].id == "r1"


def test_load_policy_rejects_unknown_kind() -> None:
    with pytest.raises(ShadowConfigError):
        load_policy([{"id": "r", "kind": "make-coffee", "params": {}}])


def test_load_policy_rejects_wrong_shape() -> None:
    with pytest.raises(ShadowConfigError):
        load_policy(42)


def test_must_call_before_passes_when_ordered_correctly() -> None:
    rules = load_policy(
        [
            {
                "id": "backup-before-migrate",
                "kind": "must_call_before",
                "params": {"first": "backup_database", "then": "run_migration"},
            }
        ]
    )
    records = [
        _response(content=[_tool_use("backup_database")], stop_reason="tool_use"),
        _response(content=[_tool_use("run_migration")], stop_reason="tool_use"),
    ]
    assert check_policy(records, rules) == []


def test_must_call_before_flags_wrong_order() -> None:
    rules = load_policy(
        [
            {
                "id": "backup-before-migrate",
                "kind": "must_call_before",
                "params": {"first": "backup_database", "then": "run_migration"},
            }
        ]
    )
    records = [
        _response(content=[_tool_use("run_migration")], stop_reason="tool_use"),
        _response(content=[_tool_use("backup_database")], stop_reason="tool_use"),
    ]
    violations = check_policy(records, rules)
    assert len(violations) == 1
    assert violations[0].rule_id == "backup-before-migrate"
    assert violations[0].pair_index == 0  # the run_migration turn


def test_no_call_flags_each_invocation() -> None:
    rules = load_policy([{"id": "no-rm-rf", "kind": "no_call", "params": {"tool": "rm_rf"}}])
    records = [
        _response(content=[_tool_use("rm_rf")], stop_reason="tool_use"),
        _response(content=[_tool_use("rm_rf")], stop_reason="tool_use"),
    ]
    violations = check_policy(records, rules)
    assert [v.pair_index for v in violations] == [0, 1]


def test_must_call_once_catches_duplicates() -> None:
    rules = load_policy(
        [{"id": "one-login", "kind": "must_call_once", "params": {"tool": "login"}}]
    )
    records = [
        _response(content=[_tool_use("login")], stop_reason="tool_use"),
        _response(content=[_tool_use("login")], stop_reason="tool_use"),
    ]
    violations = check_policy(records, rules)
    assert len(violations) == 1
    assert "2 times" in violations[0].detail


def test_max_turns_limit() -> None:
    rules = load_policy([{"id": "short", "kind": "max_turns", "params": {"limit": 2}}])
    records = [_response() for _ in range(3)]
    violations = check_policy(records, rules)
    assert len(violations) == 1


def test_required_stop_reason() -> None:
    rules = load_policy(
        [
            {
                "id": "finish-clean",
                "kind": "required_stop_reason",
                "params": {"allowed": ["end_turn"]},
            }
        ]
    )
    # final stop_reason = content_filter → violation
    bad = [_response(stop_reason="end_turn"), _response(stop_reason="content_filter")]
    assert check_policy(bad, rules)
    # final = end_turn → pass
    good = [_response(stop_reason="end_turn"), _response(stop_reason="end_turn")]
    assert check_policy(good, rules) == []


def test_max_total_tokens() -> None:
    rules = load_policy([{"id": "budget", "kind": "max_total_tokens", "params": {"limit": 500}}])
    records = [
        _response(input_tokens=200, output_tokens=200),
        _response(input_tokens=200, output_tokens=50),
    ]
    violations = check_policy(records, rules)
    assert len(violations) == 1
    assert "650" in violations[0].detail


def test_must_include_text_and_forbidden_text() -> None:
    must_rules = load_policy(
        [{"id": "cite", "kind": "must_include_text", "params": {"text": "SOURCE:"}}]
    )
    forbid_rules = load_policy(
        [{"id": "no-pii", "kind": "forbidden_text", "params": {"text": "SSN:"}}]
    )
    records_bad = [
        _response(content=[{"type": "text", "text": "answer without source"}]),
    ]
    records_good = [
        _response(content=[{"type": "text", "text": "answer with citation. SOURCE: wiki"}]),
    ]
    records_pii = [
        _response(content=[{"type": "text", "text": "leaked SSN: 123-45-6789"}]),
    ]
    assert check_policy(records_bad, must_rules)
    assert check_policy(records_good, must_rules) == []
    assert check_policy(records_pii, forbid_rules)


def test_policy_diff_classifies_regressions_and_fixes() -> None:
    rules = load_policy([{"id": "no-rm", "kind": "no_call", "params": {"tool": "rm_rf"}}])
    # baseline: clean. candidate: calls rm_rf → regression.
    base = [_response()]
    cand = [_response(content=[_tool_use("rm_rf")], stop_reason="tool_use")]
    diff = policy_diff(base, cand, rules)
    assert len(diff.regressions) == 1
    assert len(diff.fixes) == 0
    # Flip it: baseline has violation, candidate clean → fix.
    diff2 = policy_diff(cand, base, rules)
    assert len(diff2.fixes) == 1
    assert len(diff2.regressions) == 0


def test_render_policy_diff_has_labels() -> None:
    rules = load_policy([{"id": "no-rm", "kind": "no_call", "params": {"tool": "rm_rf"}}])
    base = [_response()]
    cand = [_response(content=[_tool_use("rm_rf")], stop_reason="tool_use")]
    diff = policy_diff(base, cand, rules)
    rendered = render_policy_diff(diff)
    assert "regressions" in rendered
    assert "no-rm" in rendered
