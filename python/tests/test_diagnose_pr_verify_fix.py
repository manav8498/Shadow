"""Tests for `shadow.diagnose_pr.verify_fix`.

Closes the diagnose -> fix -> verify loop. Three scenarios pinned:

  1. Bad candidate fails verify-fix (regression NOT reversed).
  2. Fixed candidate passes (regression reversed, no new violations).
  3. Fixed candidate that breaks safe traces fails (collateral damage).
"""

from __future__ import annotations

from pathlib import Path

from shadow.diagnose_pr.loaders import load_traces
from shadow.diagnose_pr.verify_fix import verify_fix


def _record_chat(s, user: str, tool: str | None = None) -> None:
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


def _build_baseline(path: Path, idx: int) -> None:
    """Compliant baseline: confirm-then-refund."""
    from shadow.sdk import Session

    with Session(output_path=path, tags={"idx": str(idx), "side": "baseline"}) as s:
        _record_chat(s, f"refund #{idx}", tool="confirm_refund_amount")
        _record_chat(s, "ok", tool="issue_refund")


def _build_violating_candidate(path: Path, idx: int) -> None:
    """Bad candidate: skips confirmation."""
    from shadow.sdk import Session

    with Session(output_path=path, tags={"idx": str(idx), "side": "bad_cand"}) as s:
        _record_chat(s, f"refund #{idx}", tool="issue_refund")


def _build_fixed_candidate(path: Path, idx: int) -> None:
    """Fixed candidate: same as baseline (compliant)."""
    from shadow.sdk import Session

    with Session(output_path=path, tags={"idx": str(idx), "side": "fixed"}) as s:
        _record_chat(s, f"refund #{idx}", tool="confirm_refund_amount")
        _record_chat(s, "ok", tool="issue_refund")


def test_verify_fix_passes_when_fixed_candidate_matches_baseline(tmp_path: Path) -> None:
    base_dir = tmp_path / "baseline"
    fix_dir = tmp_path / "fixed"
    base_dir.mkdir()
    fix_dir.mkdir()
    for i in range(3):
        _build_baseline(base_dir / f"t{i}.agentlog", i)
        _build_fixed_candidate(fix_dir / f"t{i}.agentlog", i)

    base = load_traces([base_dir])
    fixed = load_traces([fix_dir])
    affected_ids = [t.trace_id for t in base]

    report = verify_fix(
        diagnose_report={"affected_trace_ids": affected_ids},
        baseline_traces=base,
        fixed_traces=fixed,
    )
    assert report.passed is True
    assert report.affected_reversed_rate == 1.0
    assert report.fail_reasons == []


def test_verify_fix_fails_when_candidate_still_violates(tmp_path: Path) -> None:
    """The 'fixed' candidate is actually still bad — verify-fix
    must NOT pass. This is the regression-not-reversed case."""
    base_dir = tmp_path / "baseline"
    bad_dir = tmp_path / "still_bad"
    base_dir.mkdir()
    bad_dir.mkdir()
    for i in range(3):
        _build_baseline(base_dir / f"t{i}.agentlog", i)
        _build_violating_candidate(bad_dir / f"t{i}.agentlog", i)

    base = load_traces([base_dir])
    bad = load_traces([bad_dir])
    affected_ids = [t.trace_id for t in base]

    report = verify_fix(
        diagnose_report={"affected_trace_ids": affected_ids},
        baseline_traces=base,
        fixed_traces=bad,
    )
    assert report.passed is False
    assert report.affected_reversed_rate < 0.9
    assert report.fail_reasons


def test_verify_fix_handles_zero_affected_gracefully(tmp_path: Path) -> None:
    """No affected traces in the report ⇒ vacuous pass on the
    regression-reversed criterion. Doesn't crash."""
    base_dir = tmp_path / "baseline"
    fix_dir = tmp_path / "fixed"
    base_dir.mkdir()
    fix_dir.mkdir()
    for i in range(2):
        _build_baseline(base_dir / f"t{i}.agentlog", i)
        _build_fixed_candidate(fix_dir / f"t{i}.agentlog", i)

    base = load_traces([base_dir])
    fixed = load_traces([fix_dir])

    report = verify_fix(
        diagnose_report={"affected_trace_ids": []},
        baseline_traces=base,
        fixed_traces=fixed,
    )
    assert report.passed is True
    assert report.affected_total == 0


def test_verify_fix_fails_when_safe_traces_regress(tmp_path: Path) -> None:
    """The fix introduced a new regression on previously-safe
    traces — verify-fix must surface this."""
    base_dir = tmp_path / "baseline"
    fix_dir = tmp_path / "fixed"
    base_dir.mkdir()
    fix_dir.mkdir()
    for i in range(3):
        _build_baseline(base_dir / f"t{i}.agentlog", i)
        # The "fixed" candidate is actually a regression on these traces.
        _build_violating_candidate(fix_dir / f"t{i}.agentlog", i)

    base = load_traces([base_dir])
    fixed = load_traces([fix_dir])

    # Mark NONE as affected — every diverging trace counts as
    # safe_regressed.
    report = verify_fix(
        diagnose_report={"affected_trace_ids": []},
        baseline_traces=base,
        fixed_traces=fixed,
    )
    assert report.passed is False
    assert report.safe_regressed > 0


def test_verify_fix_cli_round_trips_diagnose_report(tmp_path: Path) -> None:
    """End-to-end via Typer: write a diagnose-pr report.json with
    a real affected_trace_ids list, then run verify-fix CLI and
    assert the report.json shape + exit code."""
    import json

    from typer.testing import CliRunner

    from shadow.cli.app import app

    base_dir = tmp_path / "baseline"
    fix_dir = tmp_path / "fixed"
    base_dir.mkdir()
    fix_dir.mkdir()
    for i in range(3):
        _build_baseline(base_dir / f"t{i}.agentlog", i)
        _build_fixed_candidate(fix_dir / f"t{i}.agentlog", i)

    base = load_traces([base_dir])
    affected_ids = [t.trace_id for t in base]
    diagnose_path = tmp_path / "diagnose.json"
    diagnose_path.write_text(json.dumps({"affected_trace_ids": affected_ids}))

    out_json = tmp_path / "verify.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "verify-fix",
            "--report",
            str(diagnose_path),
            "--traces",
            str(base_dir),
            "--fixed-traces",
            str(fix_dir),
            "--out",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(out_json.read_text())
    assert parsed["passed"] is True
    assert parsed["schema_version"] == "verify-fix/v0.1"
    assert parsed["affected_reversed_rate"] == 1.0
