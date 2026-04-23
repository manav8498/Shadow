"""Tests for shadow.cli. Exercise every subcommand end-to-end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from shadow import _core
from shadow.cli.app import app
from shadow.sdk import Session

runner = CliRunner()


def _make_trace(path: Path, latency_ms: int, text: str) -> None:
    with Session(output_path=path, tags={"env": "test"}) as s:
        s.record_chat(
            request={"model": "claude-opus-4-7", "messages": [], "params": {}},
            response={
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "latency_ms": latency_ms,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )


def test_init_creates_shadow_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".shadow" / "traces").is_dir()
    assert (tmp_path / ".shadow" / "config.toml").is_file()


def test_diff_with_judge_sanity_populates_axis_8(tmp_path: Path, monkeypatch: Any) -> None:
    """`diff --judge sanity` replaces the empty axis-8 row with a real one."""
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    _make_trace(baseline, latency_ms=100, text="Paris is the capital of France.")
    _make_trace(candidate, latency_ms=100, text="Paris.")

    # Patch SanityJudge to return a deterministic verdict without a real LLM.
    async def fake_score(
        self: Any, b: dict[str, Any], c: dict[str, Any], ctx: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {"verdict": "equal", "confidence": 0.9, "reason": "same answer", "score": 1.0}

    from shadow.judge import sanity as sanity_mod

    monkeypatch.setattr(sanity_mod.SanityJudge, "score_pair", fake_score)

    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "diff",
            str(baseline),
            str(candidate),
            "--judge",
            "sanity",
            "--judge-backend",
            "mock",
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_json.read_text())
    judge_row = next(r for r in data["rows"] if r["axis"] == "judge")
    assert judge_row["n"] == 1
    assert judge_row["candidate_median"] == 1.0
    assert judge_row["severity"] == "none"


def test_diff_produces_a_nine_axis_report(tmp_path: Path) -> None:
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    _make_trace(baseline, latency_ms=100, text="hello")
    _make_trace(candidate, latency_ms=250, text="hello")
    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "diff",
            str(baseline),
            str(candidate),
            "--seed",
            "1",
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_json.is_file()
    data = json.loads(out_json.read_text())
    assert len(data["rows"]) == 9
    latency_row = next(r for r in data["rows"] if r["axis"] == "latency")
    assert latency_row["delta"] > 0


def test_report_renders_markdown_and_github_pr(tmp_path: Path) -> None:
    fake_report = {
        "rows": [
            {
                "axis": axis,
                "baseline_median": 1.0,
                "candidate_median": 1.1,
                "delta": 0.1,
                "ci95_low": 0.05,
                "ci95_high": 0.15,
                "severity": "minor",
                "n": 3,
            }
            for axis in [
                "semantic",
                "trajectory",
                "safety",
                "verbosity",
                "latency",
                "cost",
                "reasoning",
                "judge",
                "conformance",
            ]
        ],
        "baseline_trace_id": "sha256:" + "a" * 64,
        "candidate_trace_id": "sha256:" + "b" * 64,
        "pair_count": 3,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(fake_report))

    md = runner.invoke(app, ["report", str(path), "--format", "markdown"])
    assert md.exit_code == 0, md.output
    assert "| axis |" in md.stdout
    assert "worst severity" in md.stdout.lower()

    gh = runner.invoke(app, ["report", str(path), "--format", "github-pr"])
    assert gh.exit_code == 0, gh.output
    assert "<details>" in gh.stdout


def test_replay_uses_mock_backend_on_matching_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "b.agentlog"
    _make_trace(baseline, latency_ms=100, text="hello")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: claude-opus-4-7\nparams:\n  temperature: 0.2\n")
    out_path = tmp_path / "candidate.agentlog"
    result = runner.invoke(
        app,
        [
            "replay",
            str(config_path),
            "--baseline",
            str(baseline),
            "--backend",
            "mock",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    records = _core.parse_agentlog(out_path.read_bytes())
    kinds = [r["kind"] for r in records]
    assert kinds[0] == "metadata"
    assert kinds[-1] == "replay_summary"


def test_bisect_runs_end_to_end_on_differing_configs(tmp_path: Path) -> None:
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("model: m\nparams:\n  temperature: 0.2\n")
    cfg_b.write_text("model: m\nparams:\n  temperature: 0.7\n")
    traces = tmp_path / "traces.agentlog"
    _make_trace(traces, latency_ms=100, text="x")
    result = runner.invoke(
        app,
        ["bisect", str(cfg_a), str(cfg_b), "--traces", str(traces)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["deltas"][0]["path"] == "params.temperature"
    assert "attributions" in data


def test_version_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "shadow 0.1.0" in result.stdout
