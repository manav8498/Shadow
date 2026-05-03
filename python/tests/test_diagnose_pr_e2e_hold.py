"""End-to-end test of the canonical refund-confirmation scenario
from the design spec §3.7.

Construct:
  * baseline trace: agent confirms refund amount before issuing
  * candidate trace: agent issues refund directly
  * policy: must_call_before(confirm_refund_amount, issue_refund),
    severity error

Expected:
  * verdict in {hold, stop}  (Week 3 distinguishes them via causal CI)
  * worst_policy_rule == "confirm-before-refund"
  * affected_traces > 0
  * PR comment names the rule
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from shadow.cli.app import app

_POLICY = """
apiVersion: shadow.dev/v1alpha1
rules:
  - id: confirm-before-refund
    kind: must_call_before
    params:
      first: confirm_refund_amount
      then: issue_refund
    severity: error
"""

_BASELINE_CFG = """
model: claude-opus-4-7
params:
  temperature: 0.0
prompt:
  system: "You are a refund agent. Always confirm the refund amount before issuing."
"""

_CANDIDATE_CFG = """
model: claude-opus-4-7
params:
  temperature: 0.0
prompt:
  system: "You are a refund agent. Process refund requests."
"""


def _build_compliant_baseline(path: Path) -> None:
    """Confirm-then-refund: compliant with the policy."""
    from shadow.sdk import Session

    with Session(output_path=path, tags={"side": "baseline"}) as s:
        s.record_chat(
            request={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "Refund order #123."}],
                "params": {},
            },
            response={
                "model": "claude-opus-4-7",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "confirm_refund_amount",
                        "input": {"order_id": "123"},
                        "id": "1",
                    }
                ],
                "stop_reason": "tool_use",
                "latency_ms": 50,
                "usage": {"input_tokens": 20, "output_tokens": 10, "thinking_tokens": 0},
            },
        )
        s.record_chat(
            request={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "Confirmed."}],
                "params": {},
            },
            response={
                "model": "claude-opus-4-7",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "issue_refund",
                        "input": {"order_id": "123"},
                        "id": "2",
                    }
                ],
                "stop_reason": "tool_use",
                "latency_ms": 60,
                "usage": {"input_tokens": 25, "output_tokens": 12, "thinking_tokens": 0},
            },
        )


def _build_violating_candidate(path: Path) -> None:
    """Refund without confirmation: violates the policy."""
    from shadow.sdk import Session

    with Session(output_path=path, tags={"side": "candidate"}) as s:
        s.record_chat(
            request={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "Refund order #123."}],
                "params": {},
            },
            response={
                "model": "claude-opus-4-7",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "issue_refund",
                        "input": {"order_id": "123"},
                        "id": "1",
                    }
                ],
                "stop_reason": "tool_use",
                "latency_ms": 50,
                "usage": {"input_tokens": 20, "output_tokens": 10, "thinking_tokens": 0},
            },
        )


def test_refund_confirmation_scenario_yields_hold_or_stop(tmp_path: Path) -> None:
    runner = CliRunner()
    base_dir = tmp_path / "baseline_traces"
    cand_dir = tmp_path / "candidate_traces"
    base_dir.mkdir()
    cand_dir.mkdir()
    _build_compliant_baseline(base_dir / "scenario1.agentlog")
    _build_violating_candidate(cand_dir / "scenario1.agentlog")

    base_cfg = tmp_path / "baseline.yaml"
    cand_cfg = tmp_path / "candidate.yaml"
    pol = tmp_path / "policy.yaml"
    base_cfg.write_text(_BASELINE_CFG)
    cand_cfg.write_text(_CANDIDATE_CFG)
    pol.write_text(_POLICY)

    out_json = tmp_path / "report.json"
    out_md = tmp_path / "comment.md"
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(base_dir),
            "--candidate-traces",
            str(cand_dir),
            "--baseline-config",
            str(base_cfg),
            "--candidate-config",
            str(cand_cfg),
            "--policy",
            str(pol),
            "--out",
            str(out_json),
            "--pr-comment",
            str(out_md),
        ],
    )
    assert result.exit_code == 0, f"stdout:\n{result.stdout}"
    parsed = json.loads(out_json.read_text())
    assert parsed["verdict"] in {"hold", "stop"}, parsed["verdict"]
    assert parsed["affected_traces"] > 0
    assert parsed["worst_policy_rule"] == "confirm-before-refund"
    assert parsed["new_policy_violations"] >= 1

    md = out_md.read_text()
    # The verdict header must be visible.
    assert parsed["verdict"].upper() in md
    # The violated rule must be cited.
    assert "confirm-before-refund" in md
