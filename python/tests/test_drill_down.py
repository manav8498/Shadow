"""Tests for the drill-down field end-to-end through the Rust bridge + renderers.

Confirms the drill-down data produced by the Rust core round-trips
through PyO3 into a shape the Python renderers can consume, and that
the markdown / terminal / github-pr renderers surface it when present
and stay silent when absent.
"""

from __future__ import annotations

import io
from typing import Any

from rich.console import Console

from shadow.report import render_github_pr, render_markdown, render_terminal


def _stub_report(drill_down: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a minimal valid DiffReport dict for renderer tests."""
    return {
        "rows": [
            {
                "axis": axis,
                "baseline_median": 0.0,
                "candidate_median": 0.0,
                "delta": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "severity": "none",
                "n": 1,
                "flags": [],
            }
            for axis in (
                "semantic",
                "trajectory",
                "safety",
                "verbosity",
                "latency",
                "cost",
                "reasoning",
                "judge",
                "conformance",
            )
        ],
        "baseline_trace_id": "sha256:aa",
        "candidate_trace_id": "sha256:bb",
        "pair_count": 2,
        "first_divergence": None,
        "divergences": [],
        "recommendations": [],
        "drill_down": drill_down or [],
    }


def _stub_row(pair_index: int, dominant: str, score: float) -> dict[str, Any]:
    return {
        "pair_index": pair_index,
        "baseline_turn": pair_index,
        "candidate_turn": pair_index,
        "regression_score": score,
        "dominant_axis": dominant,
        "axis_scores": [
            {
                "axis": dominant,
                "baseline_value": 0.0,
                "candidate_value": 1.0,
                "delta": 1.0,
                "normalized_delta": 2.0,
            },
            {
                "axis": "verbosity",
                "baseline_value": 100.0,
                "candidate_value": 250.0,
                "delta": 150.0,
                "normalized_delta": 1.5,
            },
        ],
    }


# ---- markdown ------------------------------------------------------------


def test_markdown_renders_top_regressive_pairs_section() -> None:
    report = _stub_report([_stub_row(0, "trajectory", 4.2), _stub_row(3, "verbosity", 3.1)])
    md = render_markdown(report)
    assert "### Top regressive pairs" in md
    assert "pair `#0`" in md
    assert "pair `#3`" in md
    # v3.0+ uses plain-English axis labels ("tool calls" not "trajectory").
    assert "tool calls" in md
    assert "score `4.20`" in md


def test_markdown_collapses_pairs_beyond_top_3() -> None:
    report = _stub_report([_stub_row(i, "latency", 5.0 - i * 0.1) for i in range(6)])
    md = render_markdown(report)
    # First 3 inline, next 3 collapsed.
    assert "<details><summary>+ 3 more regressive pair(s)</summary>" in md


def test_markdown_omits_section_when_drill_down_empty() -> None:
    md = render_markdown(_stub_report([]))
    assert "Top regressive pairs" not in md


# ---- terminal ------------------------------------------------------------


def test_terminal_renders_drill_down() -> None:
    report = _stub_report([_stub_row(2, "semantic", 3.5), _stub_row(4, "verbosity", 2.1)])
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=False, width=120)
    render_terminal(report, console=con)
    out = buf.getvalue()
    assert "top regressive pairs" in out
    assert "pair #2" in out
    assert "pair #4" in out
    # v3.0+ uses plain-English axis labels ("response meaning" not "semantic").
    assert "response meaning" in out


def test_terminal_silent_when_drill_down_empty() -> None:
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=False, width=120)
    render_terminal(_stub_report([]), console=con)
    out = buf.getvalue()
    assert "top regressive pairs" not in out


# ---- github_pr (inherits from markdown) ----------------------------------


def test_github_pr_includes_drill_down_when_present() -> None:
    """v3.0+: github-pr renderer puts top regressive pairs into a
    dedicated `<details>` fold with a plain-English summary; the
    'Top regressive pairs' header is no longer in the body verbatim."""
    report = _stub_report([_stub_row(1, "conformance", 2.9)])
    body = render_github_pr(report)
    assert "top regressive turn pairs" in body
    assert "pair #1" in body


# ---- end-to-end through the Rust differ ---------------------------------


def test_drill_down_populated_on_real_devops_fixture() -> None:
    """Full integration: Rust differ → PyO3 → Python dict → render.

    Uses committed real-world fixtures so this test catches regressions
    in the Rust extractors, the serde bridge, OR the renderer layer.
    """
    from pathlib import Path

    from shadow import _core

    repo_root = Path(__file__).resolve().parents[2]
    baseline = _core.parse_agentlog(
        (repo_root / "examples/devops-agent/fixtures/baseline.agentlog").read_bytes()
    )
    candidate = _core.parse_agentlog(
        (repo_root / "examples/devops-agent/fixtures/candidate.agentlog").read_bytes()
    )
    report = _core.compute_diff_report(baseline, candidate, None, 42)

    drill = report.get("drill_down", [])
    assert len(drill) >= 1, "drill_down should surface ≥ 1 regressive pair"
    assert all(row["regression_score"] >= 0 for row in drill)
    # Scores must be sorted descending.
    scores = [row["regression_score"] for row in drill]
    assert scores == sorted(scores, reverse=True)
    # Every row carries per-axis scores for 8 axes (Judge excluded).
    assert all(len(row["axis_scores"]) == 8 for row in drill)
    # Dominant axis must be one of the axes with the highest normalised
    # delta in that pair.
    for row in drill:
        max_norm = max(s["normalized_delta"] for s in row["axis_scores"])
        dominant = next(s for s in row["axis_scores"] if s["axis"] == row["dominant_axis"])
        assert dominant["normalized_delta"] == max_norm
    # On the devops-agent scenario, trajectory or verbosity should
    # dominate most pairs (real agent dropped safety tool calls).
    dominants = {row["dominant_axis"] for row in drill}
    assert dominants & {
        "trajectory",
        "verbosity",
    }, f"expected trajectory/verbosity in dominants, got {dominants}"
