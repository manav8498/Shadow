"""Tests for `shadow gate-pr` — thin CI wrapper around diagnose-pr.

Exit codes:
  0 = ship   (no behavior regression)
  1 = probe / hold (held; investigate before merge)
  2 = stop   (critical violation)
  3 = internal/tooling error (treat as failure in CI)
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from typer.testing import CliRunner

from shadow.cli.app import app


def _record(s, user: str, tool: str | None = None) -> None:
    if tool:
        content = [{"type": "tool_use", "name": tool, "input": {}, "id": "1"}]
        stop = "tool_use"
    else:
        content = [{"type": "text", "text": "ok"}]
        stop = "end_turn"
    s.record_chat(
        request={"model": "x", "messages": [{"role": "user", "content": user}], "params": {}},
        response={
            "model": "x",
            "content": content,
            "stop_reason": stop,
            "latency_ms": 10,
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        },
    )


def _stage_traces(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    base.mkdir()
    cand.mkdir()
    from shadow.sdk import Session

    with Session(output_path=base / "t.agentlog", tags={"side": "b"}) as s:
        _record(s, "refund #1", tool="confirm_refund_amount")
        _record(s, "ok", tool="issue_refund")
    with Session(output_path=cand / "t.agentlog", tags={"side": "c"}) as s:
        _record(s, "refund #1", tool="issue_refund")
    return base, cand


def _stage_configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = tmp_path / "baseline.yaml"
    cand = tmp_path / "candidate.yaml"
    pol = tmp_path / "policy.yaml"
    base.write_text(
        "model: x\nparams: {temperature: 0.0}\nprompt:\n  system: 'Always confirm refunds.'\n"
    )
    cand.write_text("model: x\nparams: {temperature: 0.0}\nprompt:\n  system: 'Process refunds.'\n")
    pol.write_text(
        "apiVersion: shadow.dev/v1alpha1\n"
        "rules:\n"
        "  - id: confirm-before-refund\n"
        "    kind: must_call_before\n"
        "    params: {first: confirm_refund_amount, then: issue_refund}\n"
        "    severity: error\n"
    )
    return base, cand, pol


def test_gate_pr_help_shows_exit_code_table() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["gate-pr", "--help"])
    assert result.exit_code == 0
    txt = result.stdout.lower()
    assert "ship" in txt or "stop" in txt


def test_gate_pr_stop_verdict_returns_exit_2(tmp_path: Path) -> None:
    """Refund scenario with policy violation -> verdict=stop -> exit 2."""
    runner = CliRunner()
    base_dir, cand_dir = _stage_traces(tmp_path)
    base_cfg, cand_cfg, pol = _stage_configs(tmp_path)
    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "gate-pr",
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
        ],
    )
    assert result.exit_code == 2, result.stdout
    parsed = json.loads(out_json.read_text())
    assert parsed["verdict"] == "stop"


def test_gate_pr_failure_summary_is_pytest_shaped(tmp_path: Path) -> None:
    """On non-ship verdicts, the 1-screen summary must carry the
    five key pieces a developer needs to act:
      1. Verdict + exit + blast-radius
      2. Dominant cause id (file:line preferred when available)
      3. Numeric evidence (axis + ATE + CI + E-value)
      4. Policy violation if any
      5. The verify-fix command

    Drift between the engine and the renderer here is the bug — if
    a CauseEstimate field disappears, this test fires before the PR
    comment regression hits a real user.
    """
    runner = CliRunner()
    base_dir, cand_dir = _stage_traces(tmp_path)
    base_cfg, cand_cfg, pol = _stage_configs(tmp_path)
    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "gate-pr",
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
            "--backend",
            "mock",
            "--out",
            str(out_json),
        ],
    )
    assert result.exit_code == 2, result.output
    # Combine streams — click 8.1 vs 8.3 differ on stderr separation.
    captured = result.output
    with contextlib.suppress(ValueError, AttributeError):
        captured = (captured or "") + (result.stderr or "")
    low = captured.lower()
    # Five-line summary structure.
    assert "stop" in low or "exit 2" in low
    assert "cause" in low or "main cause" in low or "candidate" in low
    assert "ate=" in low or "trajectory" in low
    assert "policy" in low
    assert "verify-fix" in low or "verify:" in low
    # Verbose flag NOT set — full JSON report should NOT appear inline.
    # (it's still written to --out)
    assert '"schema_version"' not in captured


def test_gate_pr_verbose_flag_dumps_full_report(tmp_path: Path) -> None:
    """--verbose surfaces the JSON report after the summary so power
    users have everything in one place without re-running."""
    runner = CliRunner()
    base_dir, cand_dir = _stage_traces(tmp_path)
    base_cfg, cand_cfg, pol = _stage_configs(tmp_path)
    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "gate-pr",
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
            "--backend",
            "mock",
            "--out",
            str(out_json),
            "--verbose",
        ],
    )
    assert result.exit_code == 2
    captured = result.output
    with contextlib.suppress(ValueError, AttributeError):
        captured = (captured or "") + (result.stderr or "")
    # Summary still present.
    assert "stop" in captured.lower() or "exit 2" in captured.lower()
    # Plus the full JSON dump.
    assert '"schema_version"' in captured


def test_gate_pr_ship_verdict_returns_exit_0(tmp_path: Path) -> None:
    """Identical baseline + identical 'candidate' -> no per-trace
    diff -> verdict=ship -> exit 0. Uses --traces only (no
    --candidate-traces) so the per-pair diff path is skipped."""
    runner = CliRunner()
    base_dir, _ = _stage_traces(tmp_path)
    base_cfg, cand_cfg, _ = _stage_configs(tmp_path)
    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "gate-pr",
            "--traces",
            str(base_dir),
            "--baseline-config",
            str(base_cfg),
            "--candidate-config",
            str(cand_cfg),
            "--out",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(out_json.read_text())
    assert parsed["verdict"] == "ship"
