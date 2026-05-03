"""Tests for the `shadow.diagnose_pr.models` dataclasses.

These dataclasses are the public report shape — every consumer (PR
comment renderer, JSON writer, future verify-fix command) depends on
their field names. The test pins those names so a careless rename
fails CI before it ships."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def test_config_delta_is_frozen_and_named() -> None:
    from shadow.diagnose_pr.models import ConfigDelta

    d = ConfigDelta(
        id="model:gpt-4.1->gpt-4.1-mini",
        kind="model",
        path="model",
        old_hash=None,
        new_hash=None,
        display="model: gpt-4.1 → gpt-4.1-mini",
    )
    assert d.kind == "model"
    with pytest.raises(FrozenInstanceError):
        d.path = "params"  # type: ignore[misc]


def test_trace_diagnosis_carries_per_trace_state() -> None:
    from shadow.diagnose_pr.models import TraceDiagnosis

    diag = TraceDiagnosis(
        trace_id="sha256:abc",
        affected=True,
        risk=78.4,
        worst_axis="trajectory",
        first_divergence={"pair_index": 2, "axis": "trajectory"},
        policy_violations=[{"rule_id": "x", "severity": "error"}],
    )
    assert diag.affected is True
    assert diag.risk == pytest.approx(78.4)


def test_cause_estimate_has_ate_and_ci_fields() -> None:
    from shadow.diagnose_pr.models import CauseEstimate

    c = CauseEstimate(
        delta_id="prompt:system",
        axis="trajectory",
        ate=0.31,
        ci_low=0.22,
        ci_high=0.44,
        e_value=2.8,
        confidence=1.0,
    )
    assert c.ci_low is not None and c.ci_high is not None
    assert c.ci_low < c.ate < c.ci_high


def test_diagnose_pr_report_has_schema_version_and_verdict() -> None:
    from shadow.diagnose_pr import SCHEMA_VERSION
    from shadow.diagnose_pr.models import DiagnosePrReport

    r = DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict="ship",
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
    assert r.schema_version == "diagnose-pr/v0.1"
    assert r.verdict == "ship"
    assert r.blast_radius == 0.0


def test_verdict_literal_rejects_unknown_values_at_type_level() -> None:
    """Compile-time, not runtime — but document the contract."""
    from shadow.diagnose_pr.models import Verdict

    # Just an existence check; the Literal["ship", "probe", "hold", "stop"]
    # guarantee is enforced by mypy --strict in CI.
    assert "ship" in Verdict.__args__  # type: ignore[attr-defined]
    assert "stop" in Verdict.__args__  # type: ignore[attr-defined]
