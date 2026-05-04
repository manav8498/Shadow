"""Snapshot test for the refund-causal-diagnosis demo's PR comment.

Pins the rendered markdown so voice drift in the renderer flags
in CI before it ships. To regenerate after a deliberate voice
change, run `examples/refund-causal-diagnosis/demo.sh` and copy
its `comment.md` into `examples/refund-causal-diagnosis/expected/`.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from shadow.cli.app import app

_DEMO_ROOT = Path(__file__).resolve().parent.parent.parent / "examples" / "refund-causal-diagnosis"


def test_refund_demo_pr_comment_matches_snapshot(tmp_path: Path) -> None:
    """Run the demo via CLI exactly as `demo.sh` does, then compare
    the rendered comment against the committed `expected/comment.md`."""
    out = tmp_path / "report.json"
    md = tmp_path / "comment.md"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(_DEMO_ROOT / "baseline_traces"),
            "--candidate-traces",
            str(_DEMO_ROOT / "candidate_traces"),
            "--baseline-config",
            str(_DEMO_ROOT / "baseline.yaml"),
            "--candidate-config",
            str(_DEMO_ROOT / "candidate.yaml"),
            "--policy",
            str(_DEMO_ROOT / "policy.yaml"),
            "--changed-files",
            "prompts/candidate.md",
            "--out",
            str(out),
            "--pr-comment",
            str(md),
            "--backend",
            "mock",
            "--n-bootstrap",
            "500",
        ],
    )
    assert result.exit_code == 0, result.stdout

    expected = (_DEMO_ROOT / "expected" / "comment.md").read_text()
    actual = md.read_text()
    assert actual == expected, (
        "PR comment drifted from the committed snapshot. "
        "If this voice change is intentional, regenerate via "
        "`./examples/refund-causal-diagnosis/demo.sh` and copy "
        "comment.md into expected/."
    )


def test_refund_demo_report_shape_is_stable(tmp_path: Path) -> None:
    """Pin the structural fields of the demo's report.json so a
    schema bump can't accidentally drop a field. Verifies the v0.1
    contract end-to-end on real fixtures."""
    out = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(_DEMO_ROOT / "baseline_traces"),
            "--candidate-traces",
            str(_DEMO_ROOT / "candidate_traces"),
            "--baseline-config",
            str(_DEMO_ROOT / "baseline.yaml"),
            "--candidate-config",
            str(_DEMO_ROOT / "candidate.yaml"),
            "--policy",
            str(_DEMO_ROOT / "policy.yaml"),
            "--out",
            str(out),
            "--backend",
            "mock",
        ],
    )
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(out.read_text())

    assert parsed["schema_version"] == "diagnose-pr/v0.1"
    assert parsed["verdict"] == "stop"
    assert parsed["total_traces"] == 3
    assert parsed["affected_traces"] == 3
    assert parsed["blast_radius"] == 1.0
    dom = parsed["dominant_cause"]
    assert dom is not None
    assert dom["delta_id"] == "prompt.system"
    assert dom["confidence"] == 1.0
    assert parsed["worst_policy_rule"] == "confirm-before-refund"
    assert parsed["new_policy_violations"] >= 1
    assert "synthetic_mock" in parsed["flags"]
