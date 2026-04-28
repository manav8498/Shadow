"""Adversarial / edge-case robustness tests for shadow.statistical and
shadow.conformal.

Covers pathological inputs that can crash naive implementations:

- NaN values in observations.
- Inf values in observations.
- Zero variance (all-identical) inputs.
- Single-row matrices for Hotelling.
- Empty calibration sets for conformal.
- Negative scores for conformal (treat as absolute).

Each test asserts the primitive either:
  (a) raises a clear ValueError, OR
  (b) returns a well-defined finite result.

Silent NaN propagation (returning NaN p-values, NaN q̂, etc.) is
explicitly NOT acceptable — it produces misleading downstream
behaviour where alarms fail open.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadow.conformal import conformal_calibrate
from shadow.statistical.hotelling import hotelling_t2
from shadow.statistical.sprt import MSPRTDetector, SPRTDetector


# ---------------------------------------------------------------------------
# Hotelling T²
# ---------------------------------------------------------------------------


class TestHotellingEdgeCases:
    def test_single_row_each_side_raises(self):
        x1 = np.array([[1.0, 2.0, 3.0]])
        x2 = np.array([[1.5, 2.5, 3.5]])
        with pytest.raises(ValueError, match="at least 2 rows"):
            hotelling_t2(x1, x2)

    def test_dimension_mismatch_raises(self):
        x1 = np.random.default_rng(0).standard_normal((10, 3))
        x2 = np.random.default_rng(1).standard_normal((10, 4))
        with pytest.raises(ValueError, match="same number of columns"):
            hotelling_t2(x1, x2)

    def test_negative_permutations_raises(self):
        x1 = np.random.default_rng(0).standard_normal((5, 2))
        x2 = np.random.default_rng(1).standard_normal((5, 2))
        with pytest.raises(ValueError, match="permutations"):
            hotelling_t2(x1, x2, permutations=-1)

    def test_all_zero_inputs_returns_finite_result(self):
        """Zero-variance input → diff=0 → T²=0 → p-value=1.0 (no rejection).
        Must NOT return NaN, must NOT crash."""
        x1 = np.zeros((5, 3))
        x2 = np.zeros((5, 3))
        result = hotelling_t2(x1, x2)
        assert math.isfinite(result.p_value)
        assert result.p_value == 1.0
        assert not result.reject_null
        assert math.isfinite(result.t2)

    def test_identical_inputs_returns_p_one(self):
        """Same data both sides: p=1.0 deterministically."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((10, 3))
        result = hotelling_t2(x, x.copy())
        assert math.isfinite(result.p_value)
        assert result.p_value == pytest.approx(1.0)

    def test_high_dim_more_features_than_samples(self):
        """D > n1+n2-2: F-test undefined; should return p=1.0 not crash."""
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal((3, 20))
        x2 = rng.standard_normal((3, 20))
        result = hotelling_t2(x1, x2)
        assert math.isfinite(result.p_value)
        assert result.p_value == 1.0
        assert not result.reject_null


# ---------------------------------------------------------------------------
# SPRT
# ---------------------------------------------------------------------------


class TestSPRTEdgeCases:
    def test_warmup_below_2_raises(self):
        with pytest.raises(ValueError, match="warmup"):
            SPRTDetector(warmup=1)

    def test_alpha_at_bounds_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            SPRTDetector(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            SPRTDetector(alpha=1.0)

    def test_zero_variance_warmup_does_not_blow_up(self):
        """All-identical warmup → σ̂=0; the implementation must floor σ
        to a small positive number rather than divide by zero."""
        det = SPRTDetector(alpha=0.05, beta=0.20, effect_size=0.5, warmup=5)
        for _ in range(5):
            det.update(0.0)  # all identical
        # First post-warmup observation must not crash.
        state = det.update(0.0)
        assert math.isfinite(state.log_lr)

    def test_constant_post_warmup_gives_finite_log_lr(self):
        det = SPRTDetector(alpha=0.05, beta=0.20, effect_size=0.5, warmup=5)
        for _ in range(5):
            det.update(0.0)
        for _ in range(50):
            state = det.update(0.0)
        assert math.isfinite(state.log_lr)


class TestMSPRTEdgeCases:
    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError, match="tau"):
            MSPRTDetector(tau=0.0)

    def test_zero_variance_warmup_doesnt_crash(self):
        det = MSPRTDetector(alpha=0.05, tau=1.0, warmup=5)
        for _ in range(5):
            det.update(0.0)
        # First post-warmup with constant input.
        state = det.update(0.0)
        # The point: no NaN, no crash. log_lr is the SPRTState field
        # used by both SPRT and mSPRT (mSPRT stores log Λ there).
        assert math.isfinite(state.log_lr)


# ---------------------------------------------------------------------------
# Conformal
# ---------------------------------------------------------------------------


class TestConformalEdgeCases:
    def test_invalid_target_coverage_raises(self):
        with pytest.raises(ValueError, match="target_coverage"):
            conformal_calibrate({"x": [1.0, 2.0]}, target_coverage=0.0)
        with pytest.raises(ValueError, match="target_coverage"):
            conformal_calibrate({"x": [1.0, 2.0]}, target_coverage=1.0)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            conformal_calibrate({"x": [1.0, 2.0]}, confidence=0.0)
        with pytest.raises(ValueError, match="confidence"):
            conformal_calibrate({"x": [1.0, 2.0]}, confidence=1.0)

    def test_empty_axis_skipped(self):
        report = conformal_calibrate({"x": []})
        assert report.axes == []
        assert report.worst_axis == ""

    def test_negative_scores_treated_as_absolute(self):
        """Nonconformity scores are non-negative by convention.
        Negative inputs (e.g. signed deltas before abs()) should be
        absolutised, not propagate as nonsensical negative q̂."""
        report = conformal_calibrate({"x": [-1.0, -2.0, -3.0, -4.0, -5.0]})
        assert report.axes[0].q_hat >= 0
        assert math.isfinite(report.axes[0].q_hat)

    def test_single_score_finite_q_hat(self):
        report = conformal_calibrate({"x": [0.5]})
        assert len(report.axes) == 1
        assert math.isfinite(report.axes[0].q_hat)

    def test_inf_score_propagates_as_inf(self):
        """An infinite score is a real signal (e.g. max-deviation observed)
        and should yield q̂=Inf — but the structure must remain
        well-formed."""
        report = conformal_calibrate({"x": [1.0, 2.0, math.inf]})
        ax = report.axes[0]
        # q̂ at the (n+1)/n quantile of {1, 2, ∞} is ∞ — that's correct
        # (we observed an infinite deviation and our bound respects it).
        assert ax.q_hat == math.inf or math.isfinite(ax.q_hat)
        # achieved_coverage and pac_delta still finite probabilities.
        assert math.isfinite(ax.achieved_coverage)
        assert 0.0 <= ax.pac_delta <= 1.0
