"""Tests for shadow.diagnose_pr.policy — thin adapter to
shadow.hierarchical for diagnose-pr's per-pair use."""

from __future__ import annotations

from pathlib import Path

_SAMPLE_POLICY = """
apiVersion: shadow.dev/v1alpha1
rules:
  - id: confirm-before-refund
    kind: must_call_before
    params:
      first: confirm_refund_amount
      then: issue_refund
    severity: error
"""


def _make_pair_with_violation(tmp_path: Path) -> tuple[list[dict], list[dict]]:
    """Build a baseline that confirms-before-refund and a candidate
    that doesn't. Used to assert the policy adapter detects the
    regression."""
    from shadow import _core
    from shadow.sdk import Session

    base_path = tmp_path / "baseline.agentlog"
    cand_path = tmp_path / "candidate.agentlog"

    # Baseline: confirm then refund (compliant)
    with Session(output_path=base_path, tags={}) as s:
        s.record_chat(
            request={
                "model": "x",
                "messages": [{"role": "user", "content": "refund"}],
                "params": {},
            },
            response={
                "model": "x",
                "content": [
                    {"type": "tool_use", "name": "confirm_refund_amount", "input": {}, "id": "1"}
                ],
                "stop_reason": "tool_use",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
        s.record_chat(
            request={
                "model": "x",
                "messages": [{"role": "user", "content": "ok"}],
                "params": {},
            },
            response={
                "model": "x",
                "content": [{"type": "tool_use", "name": "issue_refund", "input": {}, "id": "2"}],
                "stop_reason": "tool_use",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    # Candidate: refund directly (violation)
    with Session(output_path=cand_path, tags={"v": "cand"}) as s:
        s.record_chat(
            request={
                "model": "x",
                "messages": [{"role": "user", "content": "refund"}],
                "params": {},
            },
            response={
                "model": "x",
                "content": [{"type": "tool_use", "name": "issue_refund", "input": {}, "id": "1"}],
                "stop_reason": "tool_use",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    return _core.parse_agentlog(base_path.read_bytes()), _core.parse_agentlog(
        cand_path.read_bytes()
    )


def test_evaluate_policy_returns_zero_when_no_policy_path(tmp_path: Path) -> None:
    from shadow.diagnose_pr.policy import evaluate_policy

    base, cand = _make_pair_with_violation(tmp_path)
    res = evaluate_policy(None, base, cand)
    assert res.new_violations == 0
    assert res.worst_rule is None
    assert res.regressions == []


def test_evaluate_policy_detects_regression_from_yaml_file(tmp_path: Path) -> None:
    from shadow.diagnose_pr.policy import evaluate_policy

    p = tmp_path / "policy.yaml"
    p.write_text(_SAMPLE_POLICY)
    base, cand = _make_pair_with_violation(tmp_path)
    res = evaluate_policy(p, base, cand)
    assert res.new_violations >= 1
    assert res.worst_rule == "confirm-before-refund"
    assert any(v["rule_id"] == "confirm-before-refund" for v in res.regressions)


def test_evaluate_policy_compliant_candidate_has_no_regressions(tmp_path: Path) -> None:
    from shadow.diagnose_pr.policy import evaluate_policy

    p = tmp_path / "policy.yaml"
    p.write_text(_SAMPLE_POLICY)
    # Baseline already compliant, use it as both sides.
    base, _ = _make_pair_with_violation(tmp_path)
    res = evaluate_policy(p, base, base)
    assert res.new_violations == 0
    assert res.worst_rule is None
