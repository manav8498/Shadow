"""Smoke tests for the `shadow diagnose-pr` CLI command.

These exercise the full path: argv -> load configs -> load traces ->
extract deltas -> build report -> write JSON -> write markdown ->
exit code. They use the bundled quickstart fixtures so the test
runs offline."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shadow.cli.app import app


def _quickstart_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy the bundled quickstart fixtures into a writable tmp dir
    and return (baseline_cfg, candidate_cfg, traces_dir)."""
    import shadow.quickstart_data as _qs_data

    root = resources.files(_qs_data)
    baseline_yaml = tmp_path / "baseline.yaml"
    candidate_yaml = tmp_path / "candidate.yaml"
    baseline_yaml.write_bytes(root.joinpath("config_a.yaml").read_bytes())
    candidate_yaml.write_bytes(root.joinpath("config_b.yaml").read_bytes())
    traces = tmp_path / "traces"
    traces.mkdir()
    fixtures = root / "fixtures"
    for name in ("baseline.agentlog", "candidate.agentlog"):
        (traces / name).write_bytes(fixtures.joinpath(name).read_bytes())
    return baseline_yaml, candidate_yaml, traces


def test_diagnose_pr_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["diagnose-pr", "--help"])
    assert result.exit_code == 0
    assert "diagnose-pr" in result.stdout.lower() or "Diagnose" in result.stdout


def test_diagnose_pr_writes_json_and_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    base_cfg, cand_cfg, traces = _quickstart_files(tmp_path)
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "comment.md"
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(traces),
            "--baseline-config",
            str(base_cfg),
            "--candidate-config",
            str(cand_cfg),
            "--out",
            str(out_json),
            "--pr-comment",
            str(out_md),
        ],
    )
    if result.exit_code != 0:
        pytest.fail(f"exit={result.exit_code}\nstdout:\n{result.stdout}")
    assert out_json.is_file()
    assert out_md.is_file()
    parsed = json.loads(out_json.read_text())
    assert parsed["schema_version"] == "diagnose-pr/v0.1"
    assert parsed["total_traces"] >= 1
    assert "Shadow verdict" in out_md.read_text()


def test_diagnose_pr_skeleton_skips_fail_on_when_ship(tmp_path: Path) -> None:
    """In the v0.1 skeleton with no real "affected" classification,
    the demo fixtures yield ship and `--fail-on probe` should still
    return exit 0 (because verdict < probe)."""
    runner = CliRunner()
    base_cfg, cand_cfg, traces = _quickstart_files(tmp_path)
    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(traces),
            "--baseline-config",
            str(base_cfg),
            "--candidate-config",
            str(cand_cfg),
            "--out",
            str(out_json),
            "--fail-on",
            "probe",
        ],
    )
    parsed = json.loads(out_json.read_text())
    assert parsed["verdict"] == "ship"
    assert result.exit_code == 0


def test_diagnose_pr_missing_config_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    out_json = tmp_path / "report.json"
    # Need a real traces dir (even if empty) so the loader doesn't trip first
    traces = tmp_path / "traces"
    traces.mkdir()
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(traces),
            "--baseline-config",
            str(tmp_path / "nope.yaml"),
            "--candidate-config",
            str(tmp_path / "also_nope.yaml"),
            "--out",
            str(out_json),
        ],
    )
    assert result.exit_code != 0
