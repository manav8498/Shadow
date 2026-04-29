"""Tests for `shadow call` — decision logic, rendering, CLI behaviour.

The decision module is pure, so every rule has a focused unit test built
on a hand-crafted mini-report. The renderer is tested by capturing Rich
output to a string buffer. The CLI tests run end-to-end via Typer's
`CliRunner` against the bundled quickstart fixtures.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from rich.console import Console
from typer.testing import CliRunner

import shadow.quickstart_data as _qs_data
from shadow import _core
from shadow.call import Confidence, Tier, compute_call, render_call
from shadow.call.decide import (
    AxisLine,
    _classify_confidence,
    _decide_tier,
    _extract_driver,
    _short_id,
    _summarise_worst_axes,
)
from shadow.cli.app import app
from shadow.sdk import Session

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    axis: str,
    *,
    severity: str = "none",
    delta: float = 0.0,
    ci_low: float = 0.0,
    ci_high: float = 0.0,
    n: int = 0,
) -> dict[str, Any]:
    """Build a minimal axis-row dict the decide module accepts."""
    return {
        "axis": axis,
        "severity": severity,
        "delta": delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "n": n,
    }


def _report(
    *,
    pairs: int,
    rows: list[dict[str, Any]],
    divergences: list[dict[str, Any]] | None = None,
    anchor: str = "sha256:abcd1234deadbeef",
    candidate: str = "sha256:1111222233334444",
) -> dict[str, Any]:
    """Build a minimal DiffReport dict fixture."""
    return {
        "pair_count": pairs,
        "rows": rows,
        "divergences": divergences or [],
        "baseline_trace_id": anchor,
        "candidate_trace_id": candidate,
    }


def _bundled() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the bundled quickstart fixtures as parsed agentlog records."""
    root = resources.files(_qs_data) / "fixtures"
    b = _core.parse_agentlog(root.joinpath("baseline.agentlog").read_bytes())
    c = _core.parse_agentlog(root.joinpath("candidate.agentlog").read_bytes())
    return b, c


def _record_with_stop_reason(
    path: Path,
    *,
    stop_reason: str,
    latency_ms: int = 100,
    pairs: int = 1,
) -> None:
    """Record N pairs into one `.agentlog`. Session overwrites on close,
    so writing multiple pairs requires a single context-manager scope."""
    with Session(output_path=path, tags={"env": "test"}) as s:
        for _ in range(pairs):
            s.record_chat(
                request={"model": "claude-opus-4-7", "messages": [], "params": {}},
                response={
                    "model": "claude-opus-4-7",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": stop_reason,
                    "latency_ms": latency_ms,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            )


# ---------------------------------------------------------------------------
# Tier rules — pure logic, hand-crafted mini-reports
# ---------------------------------------------------------------------------


def test_tier_probe_when_pair_count_below_floor() -> None:
    tier, reasons = _decide_tier(pair_count=3, rows=[_row("trajectory", severity="severe")])
    assert tier is Tier.PROBE
    assert any("record at least" in r for r in reasons)


def test_tier_stop_when_any_axis_severe() -> None:
    tier, _ = _decide_tier(pair_count=20, rows=[_row("trajectory", severity="severe")])
    assert tier is Tier.STOP


def test_tier_hold_when_only_moderate() -> None:
    tier, _ = _decide_tier(pair_count=20, rows=[_row("verbosity", severity="moderate")])
    assert tier is Tier.HOLD


def test_tier_ship_when_all_below_moderate() -> None:
    tier, _ = _decide_tier(
        pair_count=20,
        rows=[_row("verbosity", severity="minor"), _row("latency", severity="none")],
    )
    assert tier is Tier.SHIP


def test_probe_floor_beats_severe_when_pair_count_low() -> None:
    """A clear probe-tier pair count must override an apparent severe — the
    sample is too small to pin a definitive call regardless."""
    tier, _ = _decide_tier(pair_count=2, rows=[_row("trajectory", severity="severe")])
    assert tier is Tier.PROBE


# ---------------------------------------------------------------------------
# Confidence grading
# ---------------------------------------------------------------------------


def test_confidence_firm_when_ci_far_from_zero() -> None:
    # delta 0.5, CI [0.4, 0.6] -> half-width 0.1, ratio 5.0 -> firm
    assert _classify_confidence(0.5, 0.4, 0.6, n=10) is Confidence.FIRM


def test_confidence_fair_when_ci_close_to_zero() -> None:
    # delta 0.1, CI [0.05, 0.15] -> half-width 0.05, ratio 2.0 — borderline firm.
    # Use a strictly-inside fair example: delta 0.1, CI [0.02, 0.18], hw 0.08,
    # ratio 1.25 -> fair.
    assert _classify_confidence(0.1, 0.02, 0.18, n=10) is Confidence.FAIR


def test_confidence_faint_when_ci_crosses_zero() -> None:
    assert _classify_confidence(0.1, -0.05, 0.25, n=10) is Confidence.FAINT


def test_confidence_faint_when_n_zero() -> None:
    assert _classify_confidence(0.1, 0.05, 0.15, n=0) is Confidence.FAINT


# ---------------------------------------------------------------------------
# Driver extraction
# ---------------------------------------------------------------------------


def test_driver_picks_structural_over_decision() -> None:
    div_struct = {
        "kind": "structural_drift",
        "primary_axis": "trajectory",
        "baseline_turn": 4,
        "explanation": "candidate dropped tool call(s): `lookup_order(id)`",
        "confidence": 0.8,
    }
    div_decision = {
        "kind": "decision_drift",
        "primary_axis": "safety",
        "baseline_turn": 2,
        "explanation": "stop_reason changed",
        "confidence": 0.7,
    }
    driver = _extract_driver([div_decision, div_struct], worst_axes=[])
    assert driver is not None
    assert "structural" in driver.summary
    assert driver.turn == 4


def test_driver_falls_back_to_decision_when_no_structural() -> None:
    div_decision = {
        "kind": "decision_drift",
        "primary_axis": "safety",
        "baseline_turn": 2,
        "explanation": "stop_reason changed: `end_turn` -> `content_filter`",
        "confidence": 0.7,
    }
    driver = _extract_driver([div_decision], worst_axes=[])
    assert driver is not None
    assert "decision" in driver.summary


def test_driver_falls_back_to_worst_axis_when_no_divergences() -> None:
    axes = [
        AxisLine(
            axis="latency",
            delta=314.0,
            ci_low=200.0,
            ci_high=400.0,
            severity="severe",
            confidence=Confidence.FIRM,
            n=20,
        )
    ]
    driver = _extract_driver([], worst_axes=axes)
    assert driver is not None
    assert "latency" in driver.summary
    assert driver.turn is None


def test_driver_none_when_no_movement() -> None:
    assert _extract_driver([], worst_axes=[]) is None


# ---------------------------------------------------------------------------
# Worst-axes summary
# ---------------------------------------------------------------------------


def test_worst_axes_sorted_severe_first_then_by_abs_delta() -> None:
    rows = [
        _row("a", severity="moderate", delta=0.3, ci_low=0.1, ci_high=0.5, n=10),
        _row("b", severity="severe", delta=-0.2, ci_low=-0.4, ci_high=-0.1, n=10),
        _row("c", severity="severe", delta=0.6, ci_low=0.4, ci_high=0.8, n=10),
        _row("d", severity="none"),
    ]
    out = _summarise_worst_axes(rows, limit=5)
    # Severe before moderate; within severe, larger |delta| first.
    assert [a.axis for a in out] == ["c", "b", "a"]


def test_worst_axes_drops_none_severity() -> None:
    rows = [_row("none-axis", severity="none"), _row("real-axis", severity="minor", delta=0.1)]
    out = _summarise_worst_axes(rows, limit=5)
    assert [a.axis for a in out] == ["real-axis"]


# ---------------------------------------------------------------------------
# End-to-end compute_call
# ---------------------------------------------------------------------------


def test_compute_call_on_bundled_fixture_is_probe_with_structural_driver() -> None:
    """The bundled fixtures have only 3 pairs, so the floor rule kicks in
    even though several axes read severe — the probe tier is correct."""
    b, c = _bundled()
    report = _core.compute_diff_report(b, c, None, 42)
    result = compute_call(report)
    assert result.tier is Tier.PROBE
    assert result.driver is not None
    assert "structural" in result.driver.summary
    assert result.exit_code() == 0  # probe never blocks by default


def test_compute_call_handles_empty_report_gracefully() -> None:
    result = compute_call({})
    assert result.tier is Tier.PROBE
    assert result.driver is None
    assert result.pair_count == 0


def test_call_result_to_dict_round_trips() -> None:
    b, c = _bundled()
    report = _core.compute_diff_report(b, c, None, 42)
    result = compute_call(report)
    payload = result.to_dict()
    assert payload["tier"] == result.tier.value
    assert payload["anchor_id"] == result.anchor_id
    assert isinstance(payload["worst_axes"], list)
    # JSON-serialisable so callers can pipe it.
    json.dumps(payload)


def test_short_id_strips_sha256_prefix() -> None:
    assert _short_id("sha256:abcd1234deadbeef") == "abcd1234"
    assert _short_id("abcd1234deadbeef") == "abcd1234"
    assert _short_id("") == ""


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def test_render_call_emits_tier_label_and_driver() -> None:
    b, c = _bundled()
    report = _core.compute_diff_report(b, c, None, 42)
    result = compute_call(report)
    buf = Console(record=True, width=120)
    render_call(result, console=buf)
    out = buf.export_text()
    assert "PROBE" in out
    assert "anchor" in out
    assert "candidate" in out
    assert "Driver" in out
    assert "What to do" in out


def test_render_call_ship_panel_omits_driver_section_when_clean(tmp_path: Path) -> None:
    """A clean ship-tier call has no driver — the renderer must suppress
    the Driver block instead of rendering 'None'."""
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    # 5 identical pairs each side -> ship tier, no movement.
    _record_with_stop_reason(baseline, stop_reason="end_turn", pairs=5)
    _record_with_stop_reason(candidate, stop_reason="end_turn", pairs=5)
    b = _core.parse_agentlog(baseline.read_bytes())
    c = _core.parse_agentlog(candidate.read_bytes())
    report = _core.compute_diff_report(b, c, None, 42)
    result = compute_call(report)
    assert result.tier is Tier.SHIP
    assert result.driver is None

    buf = Console(record=True, width=120)
    render_call(result, console=buf)
    out = buf.export_text()
    assert "SHIP" in out
    assert "Driver" not in out  # suppressed when no driver present


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_call_renders_panel_against_bundled_fixtures(tmp_path: Path) -> None:
    """End-to-end: writing the bundled fixtures to disk and running the
    CLI command must produce a panel and exit 0 (probe is not blocking)."""
    root = resources.files(_qs_data) / "fixtures"
    b_path = tmp_path / "anchor.agentlog"
    c_path = tmp_path / "candidate.agentlog"
    b_path.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c_path.write_bytes(root.joinpath("candidate.agentlog").read_bytes())

    result = runner.invoke(app, ["call", str(b_path), str(c_path)])
    assert result.exit_code == 0, result.output
    assert "PROBE" in result.output
    assert "structural" in result.output


def test_cli_call_strict_mode_blocks_on_probe(tmp_path: Path) -> None:
    """`--strict` must turn probe / hold into merge-blocking exits."""
    root = resources.files(_qs_data) / "fixtures"
    b_path = tmp_path / "anchor.agentlog"
    c_path = tmp_path / "candidate.agentlog"
    b_path.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c_path.write_bytes(root.joinpath("candidate.agentlog").read_bytes())
    result = runner.invoke(app, ["call", str(b_path), str(c_path), "--strict"])
    assert result.exit_code == 1


def test_cli_call_json_mode_emits_machine_readable_payload(tmp_path: Path) -> None:
    """`--json` swaps the panel for a JSON dump that pipelines can consume."""
    root = resources.files(_qs_data) / "fixtures"
    b_path = tmp_path / "anchor.agentlog"
    c_path = tmp_path / "candidate.agentlog"
    b_path.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c_path.write_bytes(root.joinpath("candidate.agentlog").read_bytes())
    result = runner.invoke(app, ["call", str(b_path), str(c_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload["tier"] == "probe"
    assert "driver" in payload
    assert "worst_axes" in payload


def test_cli_call_blocks_on_severe_when_pair_count_meets_floor(tmp_path: Path) -> None:
    """A 5+-pair severe regression must produce a stop call (exit 1)."""
    baseline = tmp_path / "anchor.agentlog"
    candidate = tmp_path / "candidate.agentlog"
    _record_with_stop_reason(baseline, stop_reason="end_turn", latency_ms=10, pairs=6)
    _record_with_stop_reason(candidate, stop_reason="end_turn", latency_ms=10000, pairs=6)

    result = runner.invoke(app, ["call", str(baseline), str(candidate)])
    assert result.exit_code == 1, result.output
    assert "STOP" in result.output


def test_cli_call_ship_on_identical_traces(tmp_path: Path) -> None:
    """Identical traces (5+ pairs) must produce a ship call (exit 0)."""
    baseline = tmp_path / "anchor.agentlog"
    candidate = tmp_path / "candidate.agentlog"
    _record_with_stop_reason(baseline, stop_reason="end_turn", pairs=5)
    _record_with_stop_reason(candidate, stop_reason="end_turn", pairs=5)
    result = runner.invoke(app, ["call", str(baseline), str(candidate)])
    assert result.exit_code == 0, result.output
    assert "SHIP" in result.output


def test_cli_call_missing_baseline_emits_friendly_hint(tmp_path: Path) -> None:
    """The shared `_fail()` hint for FileNotFoundError must surface."""
    candidate = tmp_path / "candidate.agentlog"
    _record_with_stop_reason(candidate, stop_reason="end_turn")
    result = runner.invoke(app, ["call", str(tmp_path / "missing.agentlog"), str(candidate)])
    assert result.exit_code == 1
    assert "shadow demo" in result.output  # the existing hint
