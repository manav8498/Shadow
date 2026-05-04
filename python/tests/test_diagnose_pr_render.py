"""Tests for `shadow.diagnose_pr.render`.

The renderer is the human-facing surface — every word matters.
Tests pin the structure (verdict header, affected-trace count,
suggested fix block), not the exact prose, so we can iterate on
voice without breaking CI."""

from __future__ import annotations

from shadow.diagnose_pr import SCHEMA_VERSION
from shadow.diagnose_pr.models import CauseEstimate, DiagnosePrReport
from shadow.diagnose_pr.render import render_pr_comment


def _empty_report(verdict: str = "ship") -> DiagnosePrReport:
    return DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict=verdict,  # type: ignore[arg-type]
        total_traces=0,
        affected_traces=0,
        blast_radius=0.0,
        dominant_cause=None,
        top_causes=[],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=0,
        worst_policy_rule=None,
        suggested_fix=None,
        flags=[],
    )


def test_ship_verdict_renders_short_and_friendly() -> None:
    md = render_pr_comment(_empty_report("ship"))
    assert "Shadow verdict: SHIP" in md
    assert "no behavior regression detected" in md.lower()


def test_hold_verdict_includes_dominant_cause_block() -> None:
    cause = CauseEstimate(
        delta_id="system_prompt.md:47",
        axis="trajectory",
        ate=0.31,
        ci_low=0.22,
        ci_high=0.44,
        e_value=2.8,
        confidence=1.0,
    )
    r = DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict="hold",
        total_traces=1247,
        affected_traces=84,
        blast_radius=84 / 1247,
        dominant_cause=cause,
        top_causes=[cause],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=6,
        worst_policy_rule="confirm-before-refund",
        suggested_fix="Restore the refund confirmation instruction.",
        flags=[],
    )
    md = render_pr_comment(r)
    assert "Shadow verdict: HOLD" in md
    assert "84" in md and "1,247" in md
    assert "system_prompt.md:47" in md
    assert "trajectory" in md
    assert "0.31" in md
    assert "[0.22, 0.44]" in md
    assert "2.8" in md
    assert "confirm-before-refund" in md
    assert "Restore the refund confirmation instruction" in md
    assert "shadow verify-fix" in md.lower()


def test_probe_verdict_explains_uncertainty() -> None:
    r = DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict="probe",
        total_traces=5,
        affected_traces=1,
        blast_radius=0.2,
        dominant_cause=None,
        top_causes=[],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=0,
        worst_policy_rule=None,
        suggested_fix=None,
        flags=[],
    )
    md = render_pr_comment(r)
    assert "PROBE" in md
    assert "uncertain" in md.lower() or "low confidence" in md.lower()


def test_low_power_flag_surfaces_in_comment() -> None:
    r = DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict="probe",
        total_traces=5,
        affected_traces=1,
        blast_radius=0.2,
        dominant_cause=None,
        top_causes=[],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=0,
        worst_policy_rule=None,
        suggested_fix=None,
        flags=["low_power"],
    )
    md = render_pr_comment(r)
    assert "low statistical power" in md.lower() or "few traces" in md.lower()


def test_renderer_includes_hidden_marker_for_pr_comment_dedup() -> None:
    """The GitHub Action's comment.py looks for a hidden marker so
    it can update the previous comment instead of stacking new ones."""
    md = render_pr_comment(_empty_report("ship"))
    assert "<!-- shadow-diagnose-pr -->" in md


def test_likely_candidates_renders_when_no_dominant_cause() -> None:
    """When attribution can't crown a single cause (ties at the top),
    the renderer falls back to a candidate list rather than silently
    skipping the cause section. This is the recorded-mode case where
    multiple deltas all share confidence=0.5."""
    r = DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict="hold",
        total_traces=3,
        affected_traces=2,
        blast_radius=2 / 3,
        dominant_cause=None,
        top_causes=[
            CauseEstimate(
                delta_id="prompt.system",
                axis="trajectory",
                ate=0.01,
                ci_low=None,
                ci_high=None,
                e_value=None,
                confidence=0.5,
            ),
            CauseEstimate(
                delta_id="model:gpt-4.1->gpt-4.1-mini",
                axis="trajectory",
                ate=0.01,
                ci_low=None,
                ci_high=None,
                e_value=None,
                confidence=0.5,
            ),
        ],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=0,
        worst_policy_rule=None,
        suggested_fix=None,
        flags=[],
    )
    md = render_pr_comment(r)
    assert "Likely cause candidates" in md
    assert "prompt.system" in md
    assert "model:gpt-4.1->gpt-4.1-mini" in md
    # Honesty about why no single cause was crowned.
    assert "no single one stands out" in md.lower()
    # Should NOT crown one as "appears to be the main cause".
    assert "appears to be the main cause" not in md.lower()


def test_softens_language_when_attribution_confidence_is_weak() -> None:
    """When a single dominant cause exists but bootstrap CI doesn't
    exclude zero (confidence < 1.0), the headline must say 'most
    likely candidate' not 'appears to be the main cause'."""
    r = DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict="probe",
        total_traces=10,
        affected_traces=4,
        blast_radius=0.4,
        dominant_cause=CauseEstimate(
            delta_id="params.temperature:0.2->0.7",
            axis="trajectory",
            ate=0.15,
            ci_low=-0.05,
            ci_high=0.35,
            e_value=1.4,
            confidence=0.5,
        ),
        top_causes=[],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=0,
        worst_policy_rule=None,
        suggested_fix=None,
        flags=[],
    )
    md = render_pr_comment(r)
    assert "most likely candidate" in md.lower()
    assert "appears to be the main cause" not in md.lower()


def test_synthetic_mock_flag_surfaces_disclosure_block() -> None:
    """When --backend mock was used, the renderer must disclose
    that cause magnitudes are synthetic, not grounded in real LLM
    behavior. This prevents a casual reader from treating the mock
    ATE / CI as evidence."""
    r = DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict="hold",
        total_traces=3,
        affected_traces=3,
        blast_radius=1.0,
        dominant_cause=CauseEstimate(
            delta_id="prompt.system",
            axis="trajectory",
            ate=0.6,
            ci_low=0.6,
            ci_high=0.6,
            e_value=6.7,
            confidence=1.0,
        ),
        top_causes=[],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=0,
        worst_policy_rule=None,
        suggested_fix=None,
        flags=["low_power", "synthetic_mock"],
    )
    md = render_pr_comment(r)
    assert "synthetic" in md.lower() or "mock backend" in md.lower()
    assert "--backend live" in md
