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


def test_diagnose_pr_mining_actually_samples_when_corpus_exceeds_max_traces(
    tmp_path: Path,
) -> None:
    """Regression: mining must filter `loaded` by `trace_id`
    (matching `MinedCase.baseline_source`), NOT by chat-record id.

    Earlier code mistakenly compared `loaded[0].records[0].id`
    (metadata id, == trace_id) against `case.request_record.id`
    (chat_request id, a different content address). The match set was
    always empty, the fallback `or loaded` kicked in, and a 1000-trace
    corpus silently bypassed mining. This test asserts that
    `total_traces` after mining is < the original corpus size, not ==.
    """
    runner = CliRunner()
    base_cfg, cand_cfg, _ = _quickstart_files(tmp_path)
    # Use a fresh traces dir (the _quickstart_files helper already
    # populated tmp_path/traces with two fixtures; we want a corpus
    # we control end-to-end so the mining math is predictable).
    traces = tmp_path / "big_traces"
    traces.mkdir()
    from shadow.sdk import Session

    n_in = 30
    # Vary `tags` per trace so each trace gets a unique metadata
    # content hash — without this, byte-identical metadata records
    # collide to one trace_id and mining's `baseline_source` filter
    # collapses the corpus to one survivor (the v3.0.5 envelope fix
    # that landed in shadow-core/diff isn't yet plumbed through
    # shadow.mine; tracked as a follow-up).
    for i in range(n_in):
        # Vary stop_reason + latency too so mining sees multiple
        # clusters and produces multiple representative cases.
        stop = ["end_turn", "tool_use", "max_tokens", "error"][i % 4]
        latency = [50, 200, 800, 5000][i % 4]
        with Session(output_path=traces / f"t{i:03d}.agentlog", tags={"idx": str(i)}) as s:
            s.record_chat(
                request={
                    "model": "x",
                    "messages": [{"role": "user", "content": f"q{i}"}],
                    "params": {},
                },
                response={
                    "model": "x",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": stop,
                    "latency_ms": latency,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            )

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
            "--max-traces",
            "5",  # well below n_in
        ],
    )
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(out_json.read_text())
    assert parsed["total_traces"] < n_in, (
        f"mining did not sample down: total_traces={parsed['total_traces']}, "
        f"expected < {n_in}. The fallback may have masked an empty filter."
    )


def test_diagnose_pr_invalid_fail_on_value_is_rejected(tmp_path: Path) -> None:
    """Regression: --fail-on must reject unknown values rather than
    silently treating them as 'none'. Earlier code used `.get(fail_on,
    -1)` which masked typos."""
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
            "garbage",
        ],
    )
    assert result.exit_code != 0
    # CliRunner merges stderr into stdout by default; if --fail-on
    # validation actually fired, the message will be in stdout.
    assert "fail-on" in result.stdout.lower() or result.exit_code == 1


def test_diagnose_pr_candidate_traces_with_no_filename_overlap_fails_loud(
    tmp_path: Path,
) -> None:
    """Regression: when --candidate-traces has zero filename overlap
    with --traces, every pair silently became 'no candidate, mark
    unaffected' and the verdict was SHIP — same class of silent
    failure as the mining-fallback bug. Now fails loud with exit 1."""
    runner = CliRunner()
    base_cfg, cand_cfg, traces = _quickstart_files(tmp_path)
    mismatch = tmp_path / "mismatch_candidate"
    mismatch.mkdir()
    # Copy a fixture into the candidate dir under a name that doesn't
    # exist in `traces` — guaranteed to produce zero overlap.
    fixtures_dir = traces  # already has baseline.agentlog + candidate.agentlog
    src = fixtures_dir / "baseline.agentlog"
    (mismatch / "totally_different_name.agentlog").write_bytes(src.read_bytes())

    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(traces),
            "--candidate-traces",
            str(mismatch),
            "--baseline-config",
            str(base_cfg),
            "--candidate-config",
            str(cand_cfg),
            "--out",
            str(out_json),
        ],
    )
    assert result.exit_code != 0
    # _fail() writes to stderr. click 8.1 mixed stderr into stdout by
    # default; click 8.3+ separates them. Combine both safely.
    captured = result.stdout
    try:
        captured = (captured or "") + (result.stderr or "")
    except (ValueError, AttributeError):
        pass
    assert "filename" in captured.lower()
