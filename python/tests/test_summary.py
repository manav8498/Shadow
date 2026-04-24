"""Tests for the deterministic `summarise_report` function.

The summary is user-facing — a PR reviewer reads it before the axis
table — so regressions here regress the whole UX. Every test locks
down one specific rendering invariant.
"""

from __future__ import annotations

from typing import Any

from shadow.report.summary import summarise_report


def _row(
    axis: str,
    *,
    severity: str = "none",
    delta: float = 0.0,
    baseline: float = 0.0,
    candidate: float = 0.0,
) -> dict[str, Any]:
    return {
        "axis": axis,
        "severity": severity,
        "delta": delta,
        "baseline_median": baseline,
        "candidate_median": candidate,
        "ci95_low": 0.0,
        "ci95_high": 0.0,
        "n": 10,
        "flags": [],
    }


def _nine_zero_rows() -> list[dict[str, Any]]:
    return [
        _row(axis)
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
    ]


def _report(
    rows: list[dict[str, Any]] | None = None,
    *,
    pair_count: int = 10,
    first_divergence: dict[str, Any] | None = None,
    drill_down: list[dict[str, Any]] | None = None,
    recommendations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "pair_count": pair_count,
        "rows": rows or _nine_zero_rows(),
        "first_divergence": first_divergence,
        "divergences": [],
        "drill_down": drill_down or [],
        "recommendations": recommendations or [],
    }


# ---- low-n guidance ------------------------------------------------------


def test_low_n_pair_count_1_leads_with_caveat() -> None:
    s = summarise_report(_report(pair_count=1))
    first_line = s.splitlines()[0]
    assert "Only 1 paired response" in first_line
    assert "directional, not definitive" in first_line


def test_low_n_pair_count_4_still_warns() -> None:
    s = summarise_report(_report(pair_count=4))
    assert "Only 4 paired response" in s.splitlines()[0]


def test_n_equal_5_no_caveat() -> None:
    s = summarise_report(_report(pair_count=5))
    # At n=5 we stop warning.
    assert not s.startswith("Only")


def test_zero_pair_count_says_empty() -> None:
    s = summarise_report(_report(pair_count=0))
    assert s.startswith("No paired responses")


# ---- all-quiet case ------------------------------------------------------


def test_all_noise_floor_says_no_regression() -> None:
    s = summarise_report(_report(pair_count=10))
    assert s == "All axes within noise floor — no regression detected."


# ---- structural axes lead the summary -----------------------------------


def test_structural_axes_lead_scalar_axes() -> None:
    rows = _nine_zero_rows()
    # Make verbosity severe and trajectory severe; trajectory must appear first.
    for r in rows:
        if r["axis"] == "verbosity":
            r["severity"] = "severe"
            r["delta"] = 500.0
        if r["axis"] == "trajectory":
            r["severity"] = "severe"
            r["delta"] = 0.8
    s = summarise_report(_report(rows)).lower()
    # `tool-call trajectory` must appear before `response length`.
    assert s.index("tool-call trajectory") < s.index("response length")


def test_minor_severity_axes_omitted() -> None:
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "latency":
            r["severity"] = "minor"  # below threshold
            r["delta"] = 30
        if r["axis"] == "verbosity":
            r["severity"] = "severe"
            r["delta"] = 400
    s = summarise_report(_report(rows)).lower()
    assert "response length" in s  # verbosity stayed
    assert "latency" not in s  # minor omitted


# ---- delta units --------------------------------------------------------


def test_verbosity_delta_formatted_as_tokens() -> None:
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "verbosity":
            r["severity"] = "severe"
            r["delta"] = -226
    s = summarise_report(_report(rows))
    assert "-226 tokens" in s


def test_latency_delta_formatted_as_ms() -> None:
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "latency":
            r["severity"] = "severe"
            r["delta"] = 314
    s = summarise_report(_report(rows))
    assert "+314 ms" in s


def test_cost_delta_formatted_as_usd() -> None:
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "cost":
            r["severity"] = "severe"
            r["delta"] = 0.0123
    s = summarise_report(_report(rows))
    assert "$+0.0123" in s


# ---- first divergence ---------------------------------------------------


def test_first_divergence_line_present_when_fd_populated() -> None:
    fd = {
        "baseline_turn": 2,
        "candidate_turn": 3,
        "kind": "structural_drift",
        "explanation": "added tool call fetch(url)",
        "confidence": 1.0,
    }
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "trajectory":
            r["severity"] = "severe"
            r["delta"] = 1.0
    s = summarise_report(_report(rows, first_divergence=fd))
    assert "First divergence: turn #2/#3 (structural drift) — added tool call fetch(url)" in s


def test_no_first_divergence_omits_fd_line() -> None:
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "semantic":
            r["severity"] = "severe"
            r["delta"] = -0.8
    s = summarise_report(_report(rows, first_divergence=None))
    assert "First divergence" not in s


# ---- drill-down top pair ------------------------------------------------


def test_worst_pair_mentioned_when_regression_score_high() -> None:
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "verbosity":
            r["severity"] = "severe"
            r["delta"] = 500
    drill = [
        {"pair_index": 1, "regression_score": 9.32, "dominant_axis": "verbosity"},
        {"pair_index": 2, "regression_score": 3.1, "dominant_axis": "latency"},
    ]
    s = summarise_report(_report(rows, drill_down=drill))
    assert "Worst pair: turn #1" in s
    assert "score 9.3" in s


def test_worst_pair_suppressed_when_regression_below_1() -> None:
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "verbosity":
            r["severity"] = "severe"
            r["delta"] = 500
    drill = [{"pair_index": 1, "regression_score": 0.3, "dominant_axis": "verbosity"}]
    s = summarise_report(_report(rows, drill_down=drill))
    assert "Worst pair" not in s


# ---- recommendations ----------------------------------------------------


def test_error_recommendation_leads_the_fix_line() -> None:
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "semantic":
            r["severity"] = "severe"
            r["delta"] = -0.8
    recs = [
        {"severity": "warning", "message": "Review tone at turn 2."},
        {"severity": "error", "message": "Restore JSON contract at turn 1."},
        {"severity": "info", "message": "Verify cost delta."},
    ]
    s = summarise_report(_report(rows, recommendations=recs))
    # Error should lead, despite appearing second in input.
    assert "First fix: Restore JSON contract at turn 1" in s


def test_no_error_falls_back_to_first_warning() -> None:
    rows = _nine_zero_rows()
    for r in rows:
        if r["axis"] == "semantic":
            r["severity"] = "severe"
            r["delta"] = -0.8
    recs = [{"severity": "warning", "message": "Review tone at turn 2."}]
    s = summarise_report(_report(rows, recommendations=recs))
    assert "Suggested check: Review tone at turn 2" in s


# ---- word-budget discipline ---------------------------------------------


def test_summary_fits_in_reasonable_byte_budget() -> None:
    """A realistic summary should fit in ~800 bytes. Anything longer
    is a sign the templates are growing and should be pruned."""
    rows = [
        _row("semantic", severity="severe", delta=-0.9),
        _row("trajectory", severity="severe", delta=1.0),
        _row("safety", severity="severe", delta=0.33),
        _row("verbosity", severity="severe", delta=-226),
        _row("latency", severity="severe", delta=-1110),
        _row("cost", severity="none"),
        _row("reasoning", severity="none"),
        _row("judge", severity="severe", delta=-0.5),
        _row("conformance", severity="severe", delta=-1.0),
    ]
    fd = {
        "baseline_turn": 0,
        "candidate_turn": 0,
        "kind": "structural_drift",
        "explanation": (
            "tool set changed: removed backup_database(database,label), "
            "check_replication_lag(database), added run_migration(db,migration_id)"
        ),
        "confidence": 1.0,
    }
    drill = [{"pair_index": 1, "regression_score": 9.32, "dominant_axis": "verbosity"}]
    recs = [{"severity": "error", "message": "Restore 6-tool safety sequence"}]
    s = summarise_report(_report(rows, first_divergence=fd, drill_down=drill, recommendations=recs))
    assert len(s) < 800, f"summary too long: {len(s)} bytes"
