"""End-to-end test of the canonical refund-agent wedge demo with
causal cause attribution.

Two scenarios:

  1. Recorded backend (default): with one delta extracted (only
     prompt changed in this scenario), simple_attribution names
     the prompt change as the dominant cause with confidence=1.0.
     Suggested-fix text references the prompt.

  2. Mock backend (--backend mock): the synthetic causal_from_replay
     pipeline runs and also names the prompt delta as dominant,
     with bootstrap CI excluding zero (which surfaces in the
     report as ci_low/ci_high non-null and confidence=1.0).
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

_CANDIDATE_CFG_PROMPT_ONLY = """
model: claude-opus-4-7
params:
  temperature: 0.0
prompt:
  system: "You are a refund agent. Process refund requests."
"""


def _build_compliant_baseline(path: Path) -> None:
    from shadow.sdk import Session

    with Session(output_path=path, tags={"side": "baseline"}) as s:
        s.record_chat(
            request={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "Refund #123."}],
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
    from shadow.sdk import Session

    with Session(output_path=path, tags={"side": "candidate"}) as s:
        s.record_chat(
            request={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "Refund #123."}],
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


def _stage(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
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
    cand_cfg.write_text(_CANDIDATE_CFG_PROMPT_ONLY)
    pol.write_text(_POLICY)
    return base_dir, cand_dir, base_cfg, cand_cfg, pol


def test_refund_recorded_backend_names_prompt_as_dominant_cause(tmp_path: Path) -> None:
    """Default --backend recorded: one delta extracted (prompt), so
    simple_attribution promotes it to confidence=1.0. The PR comment
    must name the dominant cause and include the fix hint."""
    runner = CliRunner()
    base_dir, cand_dir, base_cfg, cand_cfg, pol = _stage(tmp_path)
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
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(out_json.read_text())
    assert parsed["dominant_cause"] is not None
    assert parsed["dominant_cause"]["delta_id"] == "prompt.system"
    assert parsed["dominant_cause"]["confidence"] == 1.0
    assert parsed["suggested_fix"] is not None
    md = out_md.read_text()
    assert "prompt.system" in md
    assert "Suggested fix" in md or "suggested fix" in md.lower()


def test_refund_mock_backend_yields_ci_excluding_zero(tmp_path: Path) -> None:
    """--backend mock: synthetic per-delta intervention. With one
    real delta, causal_from_replay must produce a CI that excludes
    zero (confidence=1.0) and an E-value > 1."""
    runner = CliRunner()
    base_dir, cand_dir, base_cfg, cand_cfg, pol = _stage(tmp_path)
    out_json = tmp_path / "report.json"
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
            "--backend",
            "mock",
            "--n-bootstrap",
            "300",
        ],
    )
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(out_json.read_text())
    dom = parsed["dominant_cause"]
    assert dom is not None
    assert dom["delta_id"] == "prompt.system"
    assert dom["ci_low"] is not None and dom["ci_high"] is not None
    assert dom["ci_low"] > 0.0  # CI excludes zero — strong evidence
    assert dom["e_value"] is not None and dom["e_value"] > 1.0
    assert dom["confidence"] == 1.0
