"""Tests for shadow.diagnose_pr.diffing — wraps the 9-axis Rust
differ for per-trace use inside diagnose-pr."""

from __future__ import annotations

from importlib import resources


def _quickstart_pair() -> tuple[list[dict], list[dict]]:
    import shadow.quickstart_data as q
    from shadow import _core

    root = resources.files(q) / "fixtures"
    base = _core.parse_agentlog(root.joinpath("baseline.agentlog").read_bytes())
    cand = _core.parse_agentlog(root.joinpath("candidate.agentlog").read_bytes())
    return base, cand


def test_diff_pair_returns_dict_with_rows_and_severity_per_axis() -> None:
    from shadow.diagnose_pr.diffing import diff_pair

    base, cand = _quickstart_pair()
    report = diff_pair(base, cand)
    assert "rows" in report
    assert isinstance(report["rows"], list)
    severities = {row["severity"] for row in report["rows"]}
    assert severities <= {"none", "minor", "moderate", "severe"}


def test_is_affected_true_when_any_axis_severity_moderate_or_above() -> None:
    from shadow.diagnose_pr.diffing import is_affected

    rep = {"rows": [{"axis": "trajectory", "severity": "moderate"}]}
    assert is_affected(rep) is True
    rep = {"rows": [{"axis": "trajectory", "severity": "severe"}]}
    assert is_affected(rep) is True


def test_is_affected_false_when_all_axes_minor_or_none() -> None:
    from shadow.diagnose_pr.diffing import is_affected

    rep = {
        "rows": [
            {"axis": "trajectory", "severity": "none"},
            {"axis": "verbosity", "severity": "minor"},
        ]
    }
    assert is_affected(rep) is False


def test_is_affected_true_when_first_divergence_present_and_nontrivial() -> None:
    """Even a 'minor' on every axis is affected if the report includes
    a first_divergence — that's a flag from the differ that something
    structurally interesting happened."""
    from shadow.diagnose_pr.diffing import is_affected

    rep = {
        "rows": [{"axis": "trajectory", "severity": "minor"}],
        "first_divergence": {"pair_index": 1, "axis": "trajectory", "detail": "x"},
    }
    assert is_affected(rep) is True


def test_worst_axis_returns_axis_with_highest_severity() -> None:
    from shadow.diagnose_pr.diffing import worst_axis_for

    rep = {
        "rows": [
            {"axis": "verbosity", "severity": "minor"},
            {"axis": "trajectory", "severity": "severe"},
            {"axis": "safety", "severity": "moderate"},
        ]
    }
    assert worst_axis_for(rep) == "trajectory"


def test_worst_axis_returns_none_when_all_axes_are_none() -> None:
    from shadow.diagnose_pr.diffing import worst_axis_for

    rep = {"rows": [{"axis": "trajectory", "severity": "none"}]}
    assert worst_axis_for(rep) is None


def test_diff_pair_on_real_demo_fixtures_returns_nine_axes() -> None:
    from shadow.diagnose_pr.diffing import diff_pair

    base, cand = _quickstart_pair()
    rep = diff_pair(base, cand)
    axes = {row["axis"] for row in rep["rows"]}
    expected = {
        "semantic",
        "trajectory",
        "safety",
        "verbosity",
        "latency",
        "cost",
        "reasoning",
        "judge",
        "conformance",
    }
    # Judge may be missing if no judge configured; allow that.
    assert (axes & expected) >= (expected - {"judge"})
