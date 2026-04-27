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
        ["bisect", str(cfg_a), str(cfg_b), "--traces", str(traces), "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["deltas"][0]["path"] == "params.temperature"
    assert "attributions" in data


def test_bisect_default_format_is_hedged_terminal_text(tmp_path: Path) -> None:
    """Default `shadow bisect` output must lead with the correlational
    caveat instead of dumping JSON or claiming proven causation."""
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
    assert "estimated, correlational" in result.stdout
    assert "shadow replay" in result.stdout
    # Bare check-mark is gone in favor of explicit qualifiers.
    assert " ✓" not in result.stdout


def test_version_prints_version() -> None:
    from shadow import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"shadow {__version__}" in result.stdout


# ---- shadow diff --fail-on (merge-gate) ---------------------------------


def test_diff_fail_on_never_exits_zero_even_on_severe(tmp_path: Path) -> None:
    """The default --fail-on never must always exit 0 so existing
    callers don't suddenly start failing CI."""
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    _make_trace(baseline, latency_ms=10, text="hi")
    _make_trace(candidate, latency_ms=10000, text="hi")  # severe latency regression
    result = runner.invoke(app, ["diff", str(baseline), str(candidate)])
    assert result.exit_code == 0, result.output


def test_diff_fail_on_severe_exits_one_when_severe_axis_present(tmp_path: Path) -> None:
    """A severe latency regression must trigger the gate at --fail-on severe."""
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    # Need n>=2 so the bootstrap CI is meaningful.
    for _ in range(3):
        _make_trace(baseline, latency_ms=10, text="hi")
        _make_trace(candidate, latency_ms=10000, text="hi")
    result = runner.invoke(
        app,
        ["diff", str(baseline), str(candidate), "--fail-on", "severe"],
    )
    assert result.exit_code == 1, result.output


def test_diff_fail_on_severe_exits_zero_when_traces_match(tmp_path: Path) -> None:
    """Identical traces must never trip the gate."""
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    _make_trace(baseline, latency_ms=100, text="hello")
    _make_trace(candidate, latency_ms=100, text="hello")
    result = runner.invoke(
        app,
        ["diff", str(baseline), str(candidate), "--fail-on", "severe"],
    )
    assert result.exit_code == 0, result.output


def test_diff_fail_on_invalid_value_exits_two(tmp_path: Path) -> None:
    """Misuse of --fail-on must surface as exit 2 with a clear message,
    not a silent pass."""
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    _make_trace(baseline, latency_ms=100, text="hello")
    _make_trace(candidate, latency_ms=100, text="hello")
    result = runner.invoke(
        app,
        ["diff", str(baseline), str(candidate), "--fail-on", "catastrophic"],
    )
    assert result.exit_code == 2
    assert "fail-on" in result.output


def test_diff_output_json_contains_policy_diff(tmp_path: Path) -> None:
    """Regression: --policy results were terminal-only. CI scripts and PR-comment
    renderers couldn't see which rules fired or where. Now policy_diff
    rides along in the report JSON."""
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    _make_trace(baseline, latency_ms=100, text="hello FY2023")  # forbidden text
    _make_trace(candidate, latency_ms=100, text="hello FY2026")  # clean
    policy = tmp_path / "p.yaml"
    policy.write_text(
        "version: '1'\n"
        "rules:\n"
        "  - id: no-stale\n"
        "    kind: forbidden_text\n"
        "    severity: error\n"
        "    params:\n"
        "      text: 'FY2023'\n"
    )
    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "diff",
            str(baseline),
            str(candidate),
            "--policy",
            str(policy),
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_json.read_text())
    assert "policy_diff" in data, "policy_diff must appear in the saved report JSON"
    pd = data["policy_diff"]
    assert pd["baseline_violations"], "baseline had FY2023, expected violation"
    assert not pd["candidate_violations"], "candidate had FY2026, expected clean"
    assert len(pd["fixes"]) == 1
    assert pd["fixes"][0]["rule_id"] == "no-stale"


def test_pricing_table_skips_underscore_prefixed_metadata_keys() -> None:
    """Regression: in-tree pricing.json carries _comment / _updated keys
    as documentation. _parse_pricing_table used to raise
    'pricing entry for _comment must be ...' against any such file."""
    from shadow.cli.app import _parse_pricing_table

    raw = {
        "_comment": "this is a documentation note",
        "_updated": "2026-04",
        "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
        "claude-haiku": [0.0000008, 0.000004],
    }
    out = _parse_pricing_table(raw)
    assert "_comment" not in out
    assert "_updated" not in out
    assert "gpt-4o-mini" in out
    assert "claude-haiku" in out


def test_load_pricing_file_round_trips_repo_pricing(tmp_path: Path) -> None:
    """Regression: the in-tree pricing.json carries _comment and
    _updated metadata. The MCP server's shadow_diff handler used to
    raw-json-load the same file and crash inside the Rust core
    because _comment is a string, not a ModelPricing. Now both CLI
    and MCP go through load_pricing_file."""
    import json as _json

    from shadow.cli.app import load_pricing_file

    raw = {
        "_comment": "doc note",
        "_updated": "2026-04",
        "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
    }
    src = tmp_path / "pricing.json"
    src.write_text(_json.dumps(raw))
    out = load_pricing_file(src)
    assert "_comment" not in out
    assert "_updated" not in out
    assert "gpt-4o-mini" in out
