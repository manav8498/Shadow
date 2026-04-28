"""Tests for power estimation, underpowered detection, and exact-
permutation small-n path on Hotelling T².

Production criteria for A+ practical utility:
- The implementation must NEVER silently report "no regression" on
  data where statistical power to detect the effect is near zero.
  Instead it must report ``decision == "underpowered"`` so the
  caller knows their sample is too small to draw a conclusion.
- Exact (not Monte-Carlo) permutation tests must run when the total
  number of label permutations is small enough to enumerate.
- Power estimation must agree with the closed-form non-central-F
  formula on synthetic fixtures with known effect size.
"""

from __future__ import annotations

import numpy as np

from shadow.statistical.hotelling import (
    HotellingResult,
    decision_label,
    hotelling_t2,
)


class TestDecisionStates:
    def test_decision_field_exists(self) -> None:
        x1 = np.random.default_rng(0).standard_normal((30, 4))
        x2 = np.random.default_rng(1).standard_normal((30, 4)) + 2.0
        result = hotelling_t2(x1, x2, alpha=0.05)
        assert result.decision in {"reject", "fail_to_reject", "underpowered"}

    def test_clear_signal_is_reject(self) -> None:
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal((50, 4))
        x2 = rng.standard_normal((50, 4)) + 2.0  # large effect
        result = hotelling_t2(x1, x2, alpha=0.05)
        assert result.decision == "reject"
        assert result.power > 0.9

    def test_truly_no_signal_is_fail_to_reject(self) -> None:
        rng = np.random.default_rng(2)
        x1 = rng.standard_normal((100, 4))
        x2 = rng.standard_normal((100, 4))  # null
        result = hotelling_t2(x1, x2, alpha=0.05)
        assert result.decision == "fail_to_reject"
        # Power on null data is the size of the test, ≈ alpha = 0.05.
        # We don't assert that exactly; we just verify the decision
        # state is correct.

    def test_small_n_high_d_is_underpowered(self) -> None:
        """At n=5 per arm with D=12 (the v2.7 fingerprint dim), df2=-3.
        The implementation must classify this as ``underpowered`` rather
        than silently returning fail_to_reject."""
        rng = np.random.default_rng(3)
        x1 = rng.standard_normal((5, 12))
        x2 = rng.standard_normal((5, 12)) + 0.3
        result = hotelling_t2(x1, x2, alpha=0.05)
        assert result.decision == "underpowered", (
            f"n=5/D=12 must be flagged underpowered, got {result.decision!r} "
            f"with power={result.power}"
        )

    def test_decision_label_helper(self) -> None:
        """Public helper that turns a HotellingResult into a 1-line
        human label for the diff report renderer."""
        rng = np.random.default_rng(4)
        x1 = rng.standard_normal((50, 4))
        x2 = rng.standard_normal((50, 4)) + 2.0
        result = hotelling_t2(x1, x2)
        label = decision_label(result)
        assert "behavioral drift detected" in label.lower()


class TestPowerEstimation:
    def test_power_in_unit_interval(self) -> None:
        rng = np.random.default_rng(5)
        x1 = rng.standard_normal((30, 4))
        x2 = rng.standard_normal((30, 4)) + 0.5
        result = hotelling_t2(x1, x2)
        assert 0.0 <= result.power <= 1.0

    def test_power_increases_with_n(self) -> None:
        """Doubling sample size at the same effect should increase
        power monotonically."""
        rng = np.random.default_rng(6)
        # Small sample
        x1_small = rng.standard_normal((10, 4))
        x2_small = rng.standard_normal((10, 4)) + 1.0
        result_small = hotelling_t2(x1_small, x2_small)

        # Larger sample, SAME effect size
        rng = np.random.default_rng(6)
        x1_large = rng.standard_normal((100, 4))
        x2_large = rng.standard_normal((100, 4)) + 1.0
        result_large = hotelling_t2(x1_large, x2_large)

        assert result_large.power >= result_small.power

    def test_power_increases_with_effect_size(self) -> None:
        # Use a sample size where power is in the unsaturated regime so
        # different effect sizes produce different power values. At
        # n=10/D=4, a 0.1 effect gives power ≈ 0.1 and a 1.0 effect
        # gives power > 0.9.
        rng = np.random.default_rng(7)
        x1 = rng.standard_normal((10, 4))
        x2_small = rng.standard_normal((10, 4)) + 0.1
        rng2 = np.random.default_rng(7)
        x1b = rng2.standard_normal((10, 4))
        x2_large = rng2.standard_normal((10, 4)) + 1.0
        result_small = hotelling_t2(x1, x2_small)
        result_large = hotelling_t2(x1b, x2_large)
        assert result_large.power > result_small.power, (
            f"large effect power={result_large.power}, " f"small effect power={result_small.power}"
        )


class TestExactPermutation:
    """When n1 + n2 ≤ ~12, the total combinations C(n1+n2, n1) is
    small enough to enumerate. The implementation must use the exact
    permutation distribution rather than sampling, and report the
    exact p-value."""

    def test_exact_permutation_used_for_tiny_n(self) -> None:
        """Force the exact path with a small fixture and verify the
        p-value is exact (matches enumerated null)."""
        from itertools import combinations

        rng = np.random.default_rng(11)
        x1 = rng.standard_normal((4, 2))
        x2 = rng.standard_normal((4, 2)) + 1.5
        result = hotelling_t2(x1, x2, alpha=0.05, permutations=-1)

        # Verify against a hand-rolled exact enumeration.
        from shadow.statistical.hotelling import _t2_statistic

        combined = np.vstack([x1, x2])
        n1 = x1.shape[0]
        n_total = combined.shape[0]
        all_indices = list(range(n_total))
        observed_t2, _ = _t2_statistic(x1, x2)
        ge_count = 0
        total = 0
        for combo in combinations(all_indices, n1):
            mask = np.zeros(n_total, dtype=bool)
            mask[list(combo)] = True
            xa = combined[mask]
            xb = combined[~mask]
            t2_perm, _ = _t2_statistic(xa, xb)
            if t2_perm >= observed_t2:
                ge_count += 1
            total += 1
        # Exact p-value definition: ge_count / total (no Phipson-Smyth
        # +1 adjustment when we genuinely enumerated everything).
        expected_p = ge_count / total
        assert result.p_value == expected_p, (
            f"exact path should return enumerated p-value; "
            f"got {result.p_value}, expected {expected_p}"
        )

    def test_monte_carlo_used_for_larger_n(self) -> None:
        """Above the exact-enumeration threshold, the Monte-Carlo
        path with explicit B is used."""
        rng = np.random.default_rng(12)
        x1 = rng.standard_normal((20, 3))
        x2 = rng.standard_normal((20, 3)) + 0.5
        # B=200 explicit MC
        result = hotelling_t2(x1, x2, permutations=200, rng=rng)
        # The Monte-Carlo path uses Phipson-Smyth (b+1)/(B+1), so the
        # smallest possible p is 1/201 ≈ 0.005, never zero.
        assert result.p_value >= 1 / 201


class TestHotellingResultBackcompat:
    """Existing callers using only t2 / p_value / reject_null must
    keep working unchanged after the field additions."""

    def test_t2_p_reject_null_still_present(self) -> None:
        rng = np.random.default_rng(20)
        x1 = rng.standard_normal((30, 4))
        x2 = rng.standard_normal((30, 4)) + 1.0
        result = hotelling_t2(x1, x2)
        assert isinstance(result, HotellingResult)
        assert isinstance(result.t2, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.reject_null, bool)

    def test_to_dict_contains_new_fields(self) -> None:
        rng = np.random.default_rng(21)
        x1 = rng.standard_normal((30, 4))
        x2 = rng.standard_normal((30, 4)) + 1.0
        result = hotelling_t2(x1, x2)
        d = result.to_dict()
        assert "power" in d
        assert "decision" in d
        assert d["decision"] in {"reject", "fail_to_reject", "underpowered"}
