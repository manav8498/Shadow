"""Tests for `shadow.heal` and the `shadow heal` CLI command.

Three layers:

* Pure classifier — every gate, every tier, hard-refusal correctness.
* Renderer — panel rendering for each tier.
* CLI — end-to-end via Typer's CliRunner, plus the JSON and --log paths.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from rich.console import Console
from typer.testing import CliRunner

import shadow.quickstart_data as _qs_data
from shadow.cli.app import app
from shadow.heal import HealTier, classify, render_decision
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
    n: int = 5,
) -> dict[str, Any]:
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
    rows: list[dict[str, Any]] | None = None,
    divergences: list[dict[str, Any]] | None = None,
    anchor: str = "sha256:abcd1234deadbeef",
    candidate: str = "sha256:1111222233334444",
) -> dict[str, Any]:
    return {
        "pair_count": pairs,
        "rows": rows or [],
        "divergences": divergences or [],
        "baseline_trace_id": anchor,
        "candidate_trace_id": candidate,
    }


def _record_pairs(path: Path, *, n: int, latency_ms: int = 100, text: str = "ok") -> None:
    """Write N identical chat_response pairs into `path` in one Session."""
    with Session(output_path=path, tags={"env": "test"}) as s:
        for _ in range(n):
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


# ---------------------------------------------------------------------------
# Hard-refusal gates — every refusal must produce HOLD with the right reason
# ---------------------------------------------------------------------------


def test_classifier_holds_below_pair_count_floor() -> None:
    """3 pairs < 5 floor — must HOLD before any other gate runs."""
    decision = classify(_report(pairs=3))
    assert decision.tier is HealTier.HOLD
    assert any(c.name == "pair_count_floor" and not c.passed for c in decision.checks)
    assert "probe floor" in decision.rationale


def test_classifier_holds_on_structural_drift() -> None:
    """Structural drift means tool sequence changed — never auto-healable."""
    decision = classify(
        _report(
            pairs=10,
            divergences=[
                {
                    "kind": "structural_drift",
                    "primary_axis": "trajectory",
                    "baseline_turn": 4,
                    "explanation": "candidate dropped tool call(s): `lookup_order(id)`",
                    "confidence": 0.9,
                }
            ],
        )
    )
    assert decision.tier is HealTier.HOLD
    assert any(c.name == "no_structural_drift" and not c.passed for c in decision.checks)
    assert "behaviour, not implementation" in decision.rationale


def test_classifier_holds_on_safety_decision_drift() -> None:
    """Refusal flips on the safety axis are behavioural, never auto-healed."""
    decision = classify(
        _report(
            pairs=10,
            divergences=[
                {
                    "kind": "decision_drift",
                    "primary_axis": "safety",
                    "baseline_turn": 2,
                    "explanation": "stop_reason changed: `end_turn` -> `content_filter`",
                    "confidence": 0.7,
                }
            ],
        )
    )
    assert decision.tier is HealTier.HOLD
    assert any(c.name == "no_safety_decision_drift" and not c.passed for c in decision.checks)


def test_classifier_holds_on_severe_axis_with_ci_excluding_zero() -> None:
    """A severe axis whose 95% CI excludes zero is a real regression."""
    decision = classify(
        _report(
            pairs=10,
            rows=[
                _row("latency", severity="severe", delta=300.0, ci_low=200.0, ci_high=400.0),
            ],
        )
    )
    assert decision.tier is HealTier.HOLD
    assert any(c.name == "no_severe_with_signal" and not c.passed for c in decision.checks)


def test_classifier_passes_severe_axis_when_ci_crosses_zero() -> None:
    """Severe axis whose CI crosses zero is "directional only" — not a hard refusal."""
    decision = classify(
        _report(
            pairs=10,
            rows=[
                _row("verbosity", severity="severe", delta=5.0, ci_low=-10.0, ci_high=20.0),
            ],
        )
    )
    # No structural / safety / hard-signal refusal; with no divergences -> HEAL.
    assert decision.tier is HealTier.HEAL


# ---------------------------------------------------------------------------
# Tier upgrades — refinement gate (heal vs propose)
# ---------------------------------------------------------------------------


def test_classifier_heals_on_no_divergences() -> None:
    """Clean diff — every gate passes, no divergences. HEAL."""
    decision = classify(_report(pairs=10))
    assert decision.tier is HealTier.HEAL
    # Every check should have passed.
    assert all(c.passed for c in decision.checks)


def test_classifier_heals_on_style_drift_only() -> None:
    """Cosmetic divergences only — implementation-only change."""
    decision = classify(
        _report(
            pairs=10,
            divergences=[
                {
                    "kind": "style_drift",
                    "primary_axis": "semantic",
                    "baseline_turn": 1,
                    "explanation": "cosmetic wording change",
                    "confidence": 0.3,
                }
            ],
        )
    )
    assert decision.tier is HealTier.HEAL


def test_classifier_proposes_on_decision_drift_outside_safety() -> None:
    """Decision drift on a non-safety axis (e.g. trajectory arg-value
    change) passes the hard gates but is non-cosmetic — PROPOSE."""
    decision = classify(
        _report(
            pairs=10,
            divergences=[
                {
                    "kind": "decision_drift",
                    "primary_axis": "trajectory",
                    "baseline_turn": 4,
                    "explanation": "tool arg value changed",
                    "confidence": 0.6,
                }
            ],
        )
    )
    assert decision.tier is HealTier.PROPOSE
    # The cosmetic_only check should appear in the audit trail and be False.
    assert any(c.name == "cosmetic_only" and not c.passed for c in decision.checks)


# ---------------------------------------------------------------------------
# HealDecision shape + serialisation
# ---------------------------------------------------------------------------


def test_decision_to_dict_round_trips_through_json() -> None:
    """JSON-serialisable so callers can pipe it through `jq` etc."""
    decision = classify(_report(pairs=10))
    payload = decision.to_dict()
    rehydrated = json.loads(json.dumps(payload))
    assert rehydrated == payload


def test_short_id_strips_sha256_prefix() -> None:
    """Trace ids in the decision must be 8-char prefixes, not the full hash."""
    decision = classify(
        _report(
            pairs=10,
            anchor="sha256:abcdef0123456789abcdef0123456789",
            candidate="sha256:1234567890abcdef1234567890abcdef",
        )
    )
    assert decision.anchor_id == "abcdef01"
    assert decision.candidate_id == "12345678"


def test_decision_with_zero_pairs_returns_hold_not_crash() -> None:
    """Edge case: empty report must not crash, must return HOLD."""
    decision = classify({})
    assert decision.tier is HealTier.HOLD
    assert decision.pair_count == 0


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def test_render_heal_panel_includes_tier_label_and_checks() -> None:
    decision = classify(_report(pairs=10))
    buf = Console(record=True, width=120)
    render_decision(decision, console=buf)
    out = buf.export_text()
    assert "HEAL" in out
    assert "Checks" in out
    assert "Why" in out
    assert "What to do" in out


def test_render_hold_panel_includes_refusal_reason() -> None:
    decision = classify(_report(pairs=3))  # below floor -> HOLD
    buf = Console(record=True, width=120)
    render_decision(decision, console=buf)
    out = buf.export_text()
    assert "HOLD" in out
    assert "probe floor" in out


def test_render_propose_panel_suggests_autopr() -> None:
    """PROPOSE tier's What-to-do should point at shadow autopr — that's
    the user's manual path to pin the regression."""
    decision = classify(
        _report(
            pairs=10,
            divergences=[
                {
                    "kind": "decision_drift",
                    "primary_axis": "trajectory",
                    "baseline_turn": 4,
                    "explanation": "tool arg value changed",
                    "confidence": 0.6,
                }
            ],
        )
    )
    assert decision.tier is HealTier.PROPOSE
    buf = Console(record=True, width=120)
    render_decision(decision, console=buf)
    out = buf.export_text()
    assert "PROPOSE" in out
    assert "shadow autopr" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_heal_renders_panel_against_bundled_fixtures(tmp_path: Path) -> None:
    """End-to-end: bundled fixtures land at HOLD (probe floor); the
    panel must render correctly and exit 0."""
    root = resources.files(_qs_data) / "fixtures"
    a = tmp_path / "anchor.agentlog"
    c = tmp_path / "candidate.agentlog"
    a.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c.write_bytes(root.joinpath("candidate.agentlog").read_bytes())

    result = runner.invoke(app, ["heal", str(a), str(c)])
    assert result.exit_code == 0, result.output
    assert "HOLD" in result.output  # bundled fixtures hit the probe floor


def test_cli_heal_json_mode_emits_machine_readable(tmp_path: Path) -> None:
    """`--json` swaps the panel for a JSON dump."""
    root = resources.files(_qs_data) / "fixtures"
    a = tmp_path / "anchor.agentlog"
    c = tmp_path / "candidate.agentlog"
    a.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c.write_bytes(root.joinpath("candidate.agentlog").read_bytes())

    result = runner.invoke(app, ["heal", str(a), str(c), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload["tier"] == "hold"
    assert "checks" in payload


def test_cli_heal_default_writes_no_ledger(tmp_path: Path, monkeypatch: Any) -> None:
    """Zero-regression invariant: `shadow heal` without `--log` must
    not create `.shadow/ledger/` or any file inside it."""
    monkeypatch.chdir(tmp_path)
    root = resources.files(_qs_data) / "fixtures"
    a = tmp_path / "anchor.agentlog"
    c = tmp_path / "candidate.agentlog"
    a.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c.write_bytes(root.joinpath("candidate.agentlog").read_bytes())

    result = runner.invoke(app, ["heal", str(a), str(c)])
    assert result.exit_code == 0
    assert not (tmp_path / ".shadow" / "ledger").exists()


def test_cli_heal_log_flag_writes_one_ledger_entry(tmp_path: Path, monkeypatch: Any) -> None:
    """`--log` must land exactly one heal entry."""
    monkeypatch.chdir(tmp_path)
    root = resources.files(_qs_data) / "fixtures"
    a = tmp_path / "anchor.agentlog"
    c = tmp_path / "candidate.agentlog"
    a.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c.write_bytes(root.joinpath("candidate.agentlog").read_bytes())

    result = runner.invoke(app, ["heal", str(a), str(c), "--log"])
    assert result.exit_code == 0
    base = tmp_path / ".shadow" / "ledger"
    assert base.exists()
    entries = list(base.rglob("*-heal-*.json"))
    assert len(entries) == 1
    payload = json.loads(entries[0].read_text())
    assert payload["kind"] == "heal"
    assert payload["tier"] == "hold"


def test_cli_heal_missing_baseline_emits_friendly_hint(tmp_path: Path) -> None:
    """The shared `_fail()` hint for FileNotFoundError must surface."""
    candidate = tmp_path / "c.agentlog"
    _record_pairs(candidate, n=5)
    result = runner.invoke(app, ["heal", str(tmp_path / "missing.agentlog"), str(candidate)])
    assert result.exit_code == 1
    assert "shadow demo" in result.output


def test_cli_heal_high_pair_identical_traces_lands_heal(tmp_path: Path) -> None:
    """With 5+ identical pairs each side, the decision must be HEAL."""
    a = tmp_path / "a.agentlog"
    c = tmp_path / "c.agentlog"
    _record_pairs(a, n=5)
    _record_pairs(c, n=5)

    result = runner.invoke(app, ["heal", str(a), str(c), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload["tier"] == "heal"
