"""Tests for shadow.conformal — conformal prediction coverage bounds."""

from __future__ import annotations

import pytest

from shadow.conformal import (
    ConformalCoverageReport,
    build_conformal_coverage,
    conformal_calibrate,
)

# ---- helpers ----------------------------------------------------------------


def _row(axis: str, delta: float, n: int, ci_low: float = 0.0, ci_high: float = 0.0) -> dict:
    return {
        "axis": axis,
        "delta": delta,
        "n": n,
        "severity": "minor",
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


# ---- build_conformal_coverage -----------------------------------------------


class TestBuildConformalCoverage:
    def test_returns_report_type(self):
        rows = [_row("semantic", 0.05, 10)]
        report = build_conformal_coverage(rows)
        assert isinstance(report, ConformalCoverageReport)

    def test_empty_rows_returns_empty_axes(self):
        report = build_conformal_coverage([])
        assert report.axes == []
        assert report.worst_axis == ""

    def test_skips_n_zero_rows(self):
        rows = [_row("semantic", 0.05, 0), _row("trajectory", 0.10, 5)]
        report = build_conformal_coverage(rows)
        assert len(report.axes) == 1
        assert report.axes[0].axis == "trajectory"

    def test_target_coverage_preserved(self):
        rows = [_row("semantic", 0.05, 20)]
        report = build_conformal_coverage(rows, target_coverage=0.80)
        assert report.target_coverage == 0.80

    def test_confidence_preserved(self):
        rows = [_row("semantic", 0.05, 20)]
        report = build_conformal_coverage(rows, confidence=0.99)
        assert report.confidence == 0.99

    def test_worst_axis_has_largest_q_hat(self):
        rows = [
            _row("semantic", 0.10, 20, -0.05, 0.25),
            _row("trajectory", 0.50, 20, 0.20, 0.80),
        ]
        report = build_conformal_coverage(rows)
        # trajectory has larger delta → larger q_hat → worst axis
        assert report.worst_axis == "trajectory"

    def test_axes_sorted_by_q_hat_descending(self):
        rows = [
            _row("cost", 0.01, 20),
            _row("trajectory", 0.40, 20, 0.20, 0.60),
            _row("semantic", 0.10, 20, 0.05, 0.15),
        ]
        report = build_conformal_coverage(rows)
        q_hats = [ax.q_hat for ax in report.axes]
        assert q_hats == sorted(q_hats, reverse=True)

    def test_n1_degenerate_q_hat_equals_abs_delta(self):
        rows = [_row("semantic", 0.15, 1)]
        report = build_conformal_coverage(rows)
        ax = report.axes[0]
        assert abs(ax.q_hat - 0.15) < 1e-9

    def test_sufficient_n_flag_when_n_below_n_min(self):
        rows = [_row("semantic", 0.05, 2)]
        report = build_conformal_coverage(rows, target_coverage=0.95, confidence=0.99)
        # n_min at 95% coverage, 99% confidence is ceil(log(0.01)/log(0.95)) ≈ 90
        assert report.n_min > 2
        assert not report.sufficient_n

    def test_sufficient_n_flag_when_n_above_n_min(self):
        rows = [_row("semantic", 0.05, 200)]
        report = build_conformal_coverage(rows, target_coverage=0.80, confidence=0.80)
        # n_min at 80% coverage, 80% conf is small
        assert report.sufficient_n

    def test_q_hat_increases_with_target_coverage(self):
        rows = [_row("semantic", 0.10, 30, 0.05, 0.15)]
        r80 = build_conformal_coverage(rows, target_coverage=0.80)
        r95 = build_conformal_coverage(rows, target_coverage=0.95)
        assert r95.axes[0].q_hat >= r80.axes[0].q_hat

    def test_invalid_target_coverage_raises(self):
        with pytest.raises(ValueError, match="target_coverage"):
            build_conformal_coverage([], target_coverage=1.5)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            build_conformal_coverage([], confidence=0.0)

    def test_achieved_coverage_gte_target_coverage(self):
        # By the conformal guarantee, empirical coverage on the
        # calibration set must be ≥ target. Only holds for non-trivial
        # (n > 1) calibration sets with n ≥ n_min.
        rows = [_row("semantic", 0.08, 50, 0.04, 0.12)]
        report = build_conformal_coverage(rows, target_coverage=0.90)
        ax = report.axes[0]
        # Achieved coverage must meet or exceed target on calibration.
        assert ax.achieved_coverage >= ax.target_coverage - 1e-9

    def test_to_dict_round_trips(self):
        rows = [_row("trajectory", 0.30, 10, 0.10, 0.50)]
        report = build_conformal_coverage(rows)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["target_coverage"] == report.target_coverage
        assert isinstance(d["axes"], list)
        assert len(d["axes"]) == len(report.axes)

    def test_marginal_claim_contains_axis_name(self):
        rows = [_row("verbosity", 0.20, 15, 0.10, 0.30)]
        report = build_conformal_coverage(rows)
        assert "verbosity" in report.axes[0].marginal_claim

    def test_note_contains_insufficient_warning_when_n_low(self):
        rows = [_row("semantic", 0.05, 1)]
        report = build_conformal_coverage(rows, target_coverage=0.95, confidence=0.99)
        assert "n_min" in report.note or "below" in report.note

    def test_note_describes_binding_axis_when_sufficient(self):
        rows = [_row("semantic", 0.05, 200)]
        report = build_conformal_coverage(rows, target_coverage=0.80, confidence=0.80)
        assert "semantic" in report.note or "binding" in report.note

    def test_nine_axes_all_present(self):
        axis_names = [
            "semantic", "trajectory", "safety", "verbosity",
            "latency", "cost", "reasoning", "judge", "conformance",
        ]
        rows = [_row(name, 0.05 * (i + 1), 20) for i, name in enumerate(axis_names)]
        report = build_conformal_coverage(rows)
        assert len(report.axes) == 9
        reported_names = {ax.axis for ax in report.axes}
        assert reported_names == set(axis_names)


# ---- AxisCoverage -----------------------------------------------------------


class TestAxisCoverage:
    def test_to_dict_has_required_keys(self):
        rows = [_row("cost", 0.02, 10)]
        report = build_conformal_coverage(rows)
        d = report.axes[0].to_dict()
        required = {
            "axis", "n_calibration", "target_coverage",
            "q_hat", "achieved_coverage", "pac_delta", "marginal_claim",
        }
        assert required <= set(d)

    def test_pac_delta_in_unit_interval(self):
        rows = [_row("cost", 0.02, 10)]
        report = build_conformal_coverage(rows)
        ax = report.axes[0]
        assert 0.0 <= ax.pac_delta <= 1.0


# ---- conformal_calibrate (real distribution-free path) ---------------------


class TestConformalCalibrate:
    def test_returns_distribution_free_report(self):
        scores = {"latency": [0.1, 0.2, 0.3, 0.4, 0.5]}
        report = conformal_calibrate(scores)
        assert isinstance(report, ConformalCoverageReport)
        assert report.is_distribution_free is True

    def test_q_hat_is_max_for_small_n_at_high_coverage(self):
        # n=10, alpha=0.10 → idx = ceil(11*0.9)/10 = 10/10 = 1.0 → max
        scores = {"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        report = conformal_calibrate(scores, target_coverage=0.90)
        assert abs(report.axes[0].q_hat - 1.0) < 1e-9

    def test_q_hat_below_max_for_large_n(self):
        # n=100, alpha=0.05 → idx = ceil(101*0.95)/100 = 96/100 = 0.96
        # The 96th-of-100 percentile of [0..99]/99 ≈ 0.96 — strictly below max.
        scores = {"x": [i / 99.0 for i in range(100)]}
        report = conformal_calibrate(scores, target_coverage=0.95)
        ax = report.axes[0]
        assert ax.q_hat < 1.0
        # Empirical coverage on calibration must meet or exceed target.
        assert ax.achieved_coverage >= ax.target_coverage - 1e-9

    def test_empty_axis_skipped(self):
        scores = {"x": [], "y": [0.1, 0.2, 0.3]}
        report = conformal_calibrate(scores)
        assert len(report.axes) == 1
        assert report.axes[0].axis == "y"

    def test_invalid_target_coverage_raises(self):
        with pytest.raises(ValueError, match="target_coverage"):
            conformal_calibrate({"x": [0.1]}, target_coverage=0.0)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            conformal_calibrate({"x": [0.1]}, confidence=1.0)

    def test_negative_scores_become_absolute(self):
        # Nonconformity scores are conventionally non-negative; signed
        # input should be absolutised so q̂ is a magnitude.
        scores = {"x": [-1.0, -2.0, -3.0]}
        report = conformal_calibrate(scores)
        assert report.axes[0].q_hat >= 0

    def test_axes_sorted_by_q_hat_descending(self):
        scores = {
            "small": [0.01, 0.02, 0.03, 0.04, 0.05],
            "big": [1.0, 2.0, 3.0, 4.0, 5.0],
            "mid": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        report = conformal_calibrate(scores)
        q_hats = [ax.q_hat for ax in report.axes]
        assert q_hats == sorted(q_hats, reverse=True)
        assert report.worst_axis == "big"

    def test_validates_coverage_on_held_out_simulation(self):
        """Empirical validation: q̂ from n calibration draws covers
        ≥ target_coverage of held-out draws from the same distribution.

        With Gaussian scores and n=200 calibration + 1000 test, the
        marginal coverage should be very close to the target."""
        import random
        rng = random.Random(0)
        n_cal = 200
        cal_scores = [abs(rng.gauss(0, 1.0)) for _ in range(n_cal)]
        report = conformal_calibrate({"x": cal_scores}, target_coverage=0.90)
        q_hat = report.axes[0].q_hat
        n_test = 1000
        test_scores = [abs(rng.gauss(0, 1.0)) for _ in range(n_test)]
        empirical = sum(1 for s in test_scores if s <= q_hat) / n_test
        # Coverage should be ≥ target − slack; in practice typically ≥ 0.88.
        assert empirical >= 0.85, (
            f"Held-out coverage {empirical:.3f} below target 0.90"
        )

    def test_n_calibration_in_report_matches_max_axis_n(self):
        scores = {
            "a": [1.0, 2.0, 3.0],          # n=3
            "b": [0.1, 0.2, 0.3, 0.4, 0.5],  # n=5
        }
        report = conformal_calibrate(scores)
        assert report.n_calibration == 5

    def test_marginal_claim_well_formed(self):
        scores = {"semantic": [0.1, 0.2, 0.3, 0.4, 0.5]}
        report = conformal_calibrate(scores, target_coverage=0.80, confidence=0.95)
        ax = report.axes[0]
        assert "semantic" in ax.marginal_claim
        assert "80%" in ax.marginal_claim
        assert "95%" in ax.marginal_claim


# ---- parametric path is_distribution_free flag -----------------------------


class TestParametricFlag:
    def test_build_conformal_coverage_marks_parametric(self):
        rows = [_row("semantic", 0.1, 20, 0.05, 0.15)]
        report = build_conformal_coverage(rows)
        assert report.is_distribution_free is False

    def test_parametric_note_warns_about_non_distribution_free(self):
        rows = [_row("semantic", 0.1, 50, 0.05, 0.15)]
        report = build_conformal_coverage(rows, target_coverage=0.80, confidence=0.80)
        assert "parametric" in report.note.lower()


# ---- integration with certify -----------------------------------------------


class TestCertifyConformal:
    def test_build_certificate_without_conformal(self):
        from shadow.certify import build_certificate

        trace = [
            self._make_record("metadata", "m1", {}),
            self._make_record("chat_response", "r1", {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hello"}],
                "model": "gpt-4o",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }),
        ]
        cert = build_certificate(trace=trace, agent_id="test-agent")
        assert cert.regression_suite is None

    def test_build_certificate_with_conformal_none_when_no_baseline(self):
        from shadow.certify import build_certificate

        trace = [
            self._make_record("metadata", "m1", {}),
            self._make_record("chat_response", "r1", {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hello"}],
                "model": "gpt-4o",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }),
        ]
        # conformal_coverage specified but no baseline_trace — conformal is None.
        cert = build_certificate(trace=trace, agent_id="test", conformal_coverage=0.90)
        assert cert.regression_suite is None

    def _make_record(self, kind: str, rid: str, payload: dict) -> dict:
        return {
            "version": "0.1",
            "id": rid,
            "kind": kind,
            "ts": "2026-01-01T00:00:00Z",
            "parent": None,
            "payload": payload,
        }

    def test_build_certificate_with_conformal_embedded(self):
        from shadow.certify import build_certificate

        response = self._make_record("chat_response", "r1", {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "the answer is 42"}],
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "latency_ms": 300,
        })
        baseline = [
            self._make_record("metadata", "base-m", {}),
            response,
        ]
        candidate = [
            self._make_record("metadata", "cand-m", {}),
            response,
        ]

        cert = build_certificate(
            trace=candidate,
            agent_id="test-agent",
            baseline_trace=baseline,
            conformal_coverage=0.90,
            conformal_confidence=0.95,
        )
        assert cert.regression_suite is not None
        conformal = cert.regression_suite.get("conformal")
        assert conformal is not None
        assert conformal["target_coverage"] == 0.90
        assert conformal["confidence"] == 0.95
        assert isinstance(conformal["axes"], list)

    def test_cert_version_is_0_2(self):
        from shadow.certify import CERT_VERSION

        assert CERT_VERSION == "0.2"

    def test_verify_certificate_accepts_both_versions(self):

        # v0.1 cert (no conformal field) should still verify.

        from shadow.certify import AgentCertificate, _hash_payload, verify_certificate

        cert_01 = AgentCertificate(
            cert_version="0.1",
            agent_id="old-agent",
            released_at="2025-01-01T00:00:00Z",
            trace_id="sha256:abc",
        )
        cert_01.cert_id = _hash_payload(cert_01)
        payload = cert_01.to_dict()
        ok, msg = verify_certificate(payload)
        assert ok, msg

    def test_render_terminal_shows_conformal(self):
        from shadow.certify import build_certificate, render_terminal

        response = self._make_record("chat_response", "r1", {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "latency_ms": 200,
        })
        baseline = [self._make_record("metadata", "bm", {}), response]
        candidate = [self._make_record("metadata", "cm", {}), response]
        cert = build_certificate(
            trace=candidate,
            agent_id="test",
            baseline_trace=baseline,
            conformal_coverage=0.90,
        )
        rendered = render_terminal(cert)
        assert "conformal" in rendered.lower()
        assert "90%" in rendered
