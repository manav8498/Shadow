"""Tests for shadow.statistical — fingerprint, Hotelling T², SPRT."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadow.statistical.fingerprint import DIM, fingerprint_trace, mean_fingerprint
from shadow.statistical.hotelling import HotellingResult, hotelling_t2
from shadow.statistical.sprt import MultiSPRT, SPRTDetector

# ---- helpers ----------------------------------------------------------------


def _make_response(
    stop_reason: str = "end_turn",
    tool_names: list[str] | None = None,
    output_tokens: int = 100,
    latency_ms: float = 500.0,
) -> dict:
    content = []
    if tool_names:
        for name in tool_names:
            content.append({"type": "tool_use", "name": name, "input": {}})
    content.append({"type": "text", "text": "some output"})
    return {
        "kind": "chat_response",
        "id": "r1",
        "payload": {
            "stop_reason": stop_reason,
            "content": content,
            "usage": {"output_tokens": output_tokens},
            "latency_ms": latency_ms,
        },
    }


def _make_trace(responses: list[dict]) -> list[dict]:
    return [{"kind": "metadata", "id": "m1", "payload": {}}, *responses]


# ---- fingerprint_trace ------------------------------------------------------


class TestFingerprintTrace:
    def test_empty_trace_returns_zero_rows(self):
        mat = fingerprint_trace([])
        assert mat.shape == (0, DIM)

    def test_metadata_only_returns_zero_rows(self):
        mat = fingerprint_trace([{"kind": "metadata", "id": "x", "payload": {}}])
        assert mat.shape == (0, DIM)

    def test_one_response_produces_one_row(self):
        trace = _make_trace([_make_response()])
        mat = fingerprint_trace(trace)
        assert mat.shape == (1, DIM)

    def test_all_values_in_unit_interval(self):
        trace = _make_trace(
            [
                _make_response(stop_reason="end_turn", tool_names=["search"], output_tokens=500),
                _make_response(stop_reason="tool_use", tool_names=["search", "fetch"]),
                _make_response(stop_reason="content_filter"),
            ]
        )
        mat = fingerprint_trace(trace)
        assert mat.shape == (3, DIM)
        assert np.all(mat >= 0.0 - 1e-9)
        assert np.all(mat <= 1.0 + 1e-9)

    def test_stop_reason_one_hot_exclusive(self):
        trace_end = _make_trace([_make_response(stop_reason="end_turn")])
        trace_tool = _make_trace([_make_response(stop_reason="tool_use")])
        trace_filter = _make_trace([_make_response(stop_reason="content_filter")])

        v_end = fingerprint_trace(trace_end)[0]
        v_tool = fingerprint_trace(trace_tool)[0]
        v_filter = fingerprint_trace(trace_filter)[0]

        # stop_end_turn is feature 2, stop_tool_use is 3, refusal_flag is 7
        assert v_end[2] == 1.0 and v_end[3] == 0.0 and v_end[7] == 0.0
        assert v_tool[2] == 0.0 and v_tool[3] == 1.0 and v_tool[7] == 0.0
        assert v_filter[2] == 0.0 and v_filter[3] == 0.0 and v_filter[7] == 1.0

    def test_tool_call_rate_saturates_at_max(self):
        # Default max_tool_calls=8 → ≥8 tools saturates dim to 1.0.
        names = [f"t{i}" for i in range(10)]
        trace = _make_trace([_make_response(tool_names=names)])
        mat = fingerprint_trace(trace)
        assert mat[0, 0] == pytest.approx(1.0, abs=1e-9)

    def test_tool_call_rate_log_scaled_discriminates(self):
        # 1 tool < 4 tools < 8 tools → strictly increasing on this axis.
        v1 = fingerprint_trace(_make_trace([_make_response(tool_names=["a"])]))[0, 0]
        v4 = fingerprint_trace(_make_trace([_make_response(tool_names=["a", "b", "c", "d"])]))[0, 0]
        v8 = fingerprint_trace(
            _make_trace([_make_response(tool_names=[f"t{i}" for i in range(8)])])
        )[0, 0]
        assert v1 < v4 < v8 + 1e-9

    def test_no_tools_zero_call_rate_and_zero_distinct(self):
        trace = _make_trace([_make_response(tool_names=None)])
        mat = fingerprint_trace(trace)
        assert mat[0, 0] == 0.0  # tool_call_rate
        assert mat[0, 1] == 0.0  # distinct_tool_frac

    def test_fingerprint_config_overrides_scales(self):
        from shadow.statistical.fingerprint import FingerprintConfig

        # Same observation, two configs: a smaller max_tool_calls makes
        # the dim saturate sooner.
        trace = _make_trace([_make_response(tool_names=["a", "b", "c", "d"])])
        small_cfg = FingerprintConfig(max_tool_calls=4)
        big_cfg = FingerprintConfig(max_tool_calls=64)
        small = fingerprint_trace(trace, small_cfg)[0, 0]
        big = fingerprint_trace(trace, big_cfg)[0, 0]
        assert small > big

    def test_distinct_tool_frac_two_same(self):
        trace = _make_trace([_make_response(tool_names=["x", "x"])])
        mat = fingerprint_trace(trace)
        # 1 distinct / 2 total = 0.5
        assert abs(mat[0, 1] - 0.5) < 1e-9

    def test_output_len_log_scales_with_tokens(self):
        trace_small = _make_trace([_make_response(output_tokens=10)])
        trace_large = _make_trace([_make_response(output_tokens=4000)])
        v_small = fingerprint_trace(trace_small)[0]
        v_large = fingerprint_trace(trace_large)[0]
        assert v_small[5] < v_large[5]

    def test_mean_fingerprint_matches_matrix_mean(self):
        trace = _make_trace(
            [
                _make_response(stop_reason="end_turn", output_tokens=100),
                _make_response(stop_reason="tool_use", output_tokens=200),
            ]
        )
        mat = fingerprint_trace(trace)
        mean = mean_fingerprint(trace)
        np.testing.assert_allclose(mean, mat.mean(axis=0))

    def test_mean_fingerprint_empty_returns_zeros(self):
        mean = mean_fingerprint([])
        assert mean.shape == (DIM,)
        assert np.all(mean == 0.0)


# ---- hotelling_t2 -----------------------------------------------------------


class TestHotellingT2:
    def _identical_samples(self, n: int = 20, d: int = 4) -> tuple:
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal((n, d))
        x2 = rng.standard_normal((n, d))  # same distribution, different samples
        return x1, x2

    def _shifted_samples(self, n: int = 30, d: int = 4, shift: float = 2.0) -> tuple:
        rng = np.random.default_rng(1)
        x1 = rng.standard_normal((n, d))
        x2 = rng.standard_normal((n, d)) + shift  # clear shift
        return x1, x2

    def test_returns_hotelling_result(self):
        x1, x2 = self._identical_samples()
        result = hotelling_t2(x1, x2)
        assert isinstance(result, HotellingResult)

    def test_p_value_in_unit_interval(self):
        x1, x2 = self._identical_samples()
        result = hotelling_t2(x1, x2)
        assert 0.0 <= result.p_value <= 1.0

    def test_clearly_shifted_data_rejects_null(self):
        x1, x2 = self._shifted_samples(n=50, shift=3.0)
        result = hotelling_t2(x1, x2, alpha=0.05)
        assert result.reject_null, f"Expected H1 rejection; p={result.p_value:.4f}"

    def test_same_data_does_not_reject_at_low_alpha(self):
        # Generate from the same distribution 10 times; at α=0.001 we
        # should almost never reject (p > 0.001 on > 99.9% of seeds).
        rng = np.random.default_rng(42)
        rejections = 0
        for _ in range(20):
            x1 = rng.standard_normal((30, 4))
            x2 = rng.standard_normal((30, 4))
            r = hotelling_t2(x1, x2, alpha=0.001)
            if r.reject_null:
                rejections += 1
        # At α=0.001 over 20 trials we expect 0 rejections, allow ≤2.
        assert rejections <= 2, f"Too many false rejections: {rejections}/20"

    def test_correct_degrees_of_freedom(self):
        x1 = np.random.default_rng(5).standard_normal((15, 3))
        x2 = np.random.default_rng(6).standard_normal((15, 3))
        result = hotelling_t2(x1, x2)
        assert result.df1 == 3
        assert result.df2 == 15 + 15 - 3 - 1  # n1+n2-D-1 = 26

    def test_raises_on_single_row_input(self):
        x1 = np.random.standard_normal((1, 4))
        x2 = np.random.standard_normal((10, 4))
        with pytest.raises(ValueError, match="at least 2 rows"):
            hotelling_t2(x1, x2)

    def test_raises_on_dimension_mismatch(self):
        x1 = np.random.standard_normal((10, 4))
        x2 = np.random.standard_normal((10, 5))
        with pytest.raises(ValueError, match="same number of columns"):
            hotelling_t2(x1, x2)

    def test_high_dim_graceful_p1_when_df2_nonpositive(self):
        # D > n1+n2-2: df2 ≤ 0, must return p=1 instead of crashing.
        rng = np.random.default_rng(7)
        x1 = rng.standard_normal((3, 10))  # n1=3, D=10 → df2 = 3+3-10-1 < 0
        x2 = rng.standard_normal((3, 10))
        result = hotelling_t2(x1, x2)
        assert result.p_value == 1.0
        assert not result.reject_null

    def test_permutation_p_value_uses_phipson_smyth_correction(self):
        """Permutation p-value follows (b+1)/(B+1) — the Phipson-Smyth (2010)
        unbiased estimator. Verify the formula directly: with B=99 and a
        known clearly-separated input, the p-value should be 1/100 (the
        minimum non-zero value) when no permutation matches the observed."""
        rng = np.random.default_rng(42)
        # Strong shift: candidate is N(5, 1) — every permutation of
        # combined data will have a smaller T² than the observed.
        x1 = rng.standard_normal((20, 3))
        x2 = rng.standard_normal((20, 3)) + 5.0
        result = hotelling_t2(x1, x2, alpha=0.05, permutations=99, rng=np.random.default_rng(0))
        # Lower bound (1/(B+1)) is achieved when count==0; never exactly 0.
        assert result.p_value >= 1 / 100
        # And should be near the floor for this strong shift.
        assert result.p_value <= 5 / 100

    def test_permutation_p_value_never_zero(self):
        """Phipson-Smyth ensures p-value > 0 even when no permutation
        exceeds observed — important because p=0 would over-state evidence."""
        rng = np.random.default_rng(43)
        x1 = rng.standard_normal((10, 2))
        x2 = rng.standard_normal((10, 2)) + 100.0  # absurdly large shift
        result = hotelling_t2(x1, x2, permutations=50, rng=np.random.default_rng(1))
        assert result.p_value > 0

    def test_shrinkage_nonzero_when_high_dim(self):
        rng = np.random.default_rng(8)
        x1 = rng.standard_normal((5, 8))  # p/n is high
        x2 = rng.standard_normal((5, 8))
        result = hotelling_t2(x1, x2)
        assert result.shrinkage > 0.0

    def test_to_dict_has_required_keys(self):
        x1, x2 = self._identical_samples()
        d = hotelling_t2(x1, x2).to_dict()
        for key in ("t2", "f_stat", "p_value", "df1", "df2", "n1", "n2", "d", "reject_null"):
            assert key in d


# ---- SPRTDetector -----------------------------------------------------------


class TestSPRTDetector:
    def test_warmup_phase_returns_continue(self):
        det = SPRTDetector(warmup=5)
        for _ in range(4):
            state = det.update(0.1)
            assert state.decision == "continue"
            assert state.in_warmup

    def test_warmup_completes_at_n_warmup(self):
        det = SPRTDetector(warmup=3)
        for _ in range(3):
            det.update(0.0)
        # After warmup, still continue until boundary crossed.
        assert not det.update(0.0).in_warmup

    def test_drift_detected_on_large_shift(self):
        # Feed warmup with near-zero scores, then large positive values.
        det = SPRTDetector(alpha=0.05, beta=0.20, effect_size=1.0, warmup=5)
        for _ in range(5):
            det.update(0.0)  # warmup: μ0 ≈ 0, σ ≈ small
        # Saturate with large values to push log_LR above log_B.
        for _ in range(30):
            state = det.update(5.0)
            if state.decision == "h1":
                break
        assert det.decision == "h1"

    def test_no_drift_accepted_on_identical_stream(self):
        # Feed values from the same distribution as warmup; should accept H0.
        rng = __import__("random").Random(42)
        det = SPRTDetector(alpha=0.05, beta=0.20, effect_size=0.5, warmup=5)
        for _ in range(5):
            det.update(rng.gauss(0, 0.01))  # warmup from near-zero stream
        for _ in range(50):
            state = det.update(rng.gauss(0, 0.01))  # same distribution
            if state.decision == "h0":
                break
        assert det.decision == "h0"

    def test_reset_clears_state(self):
        det = SPRTDetector(warmup=3)
        for _ in range(10):
            det.update(1.0)
        det.reset()
        assert det.n_observations == 0
        assert det.decision == "continue"
        assert det.log_lr == 0.0

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            SPRTDetector(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            SPRTDetector(alpha=0.6)

    def test_invalid_warmup_raises(self):
        with pytest.raises(ValueError, match="warmup"):
            SPRTDetector(warmup=1)

    def test_state_dict_has_required_keys(self):
        det = SPRTDetector(warmup=3)
        det.update(0.5)
        state = det.update(0.5)
        d = state.to_dict()
        assert {"log_lr", "n_observations", "decision", "in_warmup"} <= set(d)

    def test_boundaries_wald_formula(self):
        det = SPRTDetector(alpha=0.05, beta=0.20)
        log_a, log_b = det.boundaries
        expected_a = math.log(0.20 / 0.95)
        expected_b = math.log(0.80 / 0.05)
        assert abs(log_a - expected_a) < 1e-9
        assert abs(log_b - expected_b) < 1e-9


# ---- MSPRTDetector ----------------------------------------------------------


class TestMSPRTDetector:
    def test_warmup_returns_continue(self):
        from shadow.statistical.sprt import MSPRTDetector

        det = MSPRTDetector(alpha=0.05, tau=1.0, warmup=5)
        for i in range(5):
            state = det.update(0.0)
            assert state.in_warmup is True
            assert state.decision == "continue"

    def test_threshold_log_one_over_alpha(self):
        from shadow.statistical.sprt import MSPRTDetector

        det = MSPRTDetector(alpha=0.05)
        assert abs(det.threshold - math.log(1.0 / 0.05)) < 1e-9

    def test_invalid_alpha_raises(self):
        from shadow.statistical.sprt import MSPRTDetector

        with pytest.raises(ValueError, match="alpha"):
            MSPRTDetector(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            MSPRTDetector(alpha=1.0)

    def test_invalid_tau_raises(self):
        from shadow.statistical.sprt import MSPRTDetector

        with pytest.raises(ValueError, match="tau"):
            MSPRTDetector(tau=0.0)

    def test_invalid_warmup_raises(self):
        from shadow.statistical.sprt import MSPRTDetector

        with pytest.raises(ValueError, match="warmup"):
            MSPRTDetector(warmup=1)

    def test_detects_strong_drift(self):
        from shadow.statistical.sprt import MSPRTDetector

        det = MSPRTDetector(alpha=0.05, tau=1.0, warmup=5)
        # Warmup with mean 0, std 1.
        for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            det.update(v)
        # Then push observations far from the calibrated mean.
        for _ in range(50):
            state = det.update(5.0)
            if state.decision == "h1":
                break
        assert det.decision == "h1"

    def test_decision_is_absorbing(self):
        from shadow.statistical.sprt import MSPRTDetector

        det = MSPRTDetector(alpha=0.05, tau=1.0, warmup=3)
        for v in [-1.0, 0.0, 1.0]:
            det.update(v)
        for _ in range(50):
            det.update(10.0)
            if det.decision == "h1":
                break
        assert det.decision == "h1"
        # Subsequent observations near 0 must NOT flip the decision.
        for _ in range(20):
            state = det.update(0.0)
            assert state.decision == "h1"

    def test_reset_clears_state(self):
        from shadow.statistical.sprt import MSPRTDetector

        det = MSPRTDetector(warmup=3)
        for _ in range(10):
            det.update(1.0)
        det.reset()
        assert det.n_observations == 0
        assert det.decision == "continue"
        assert det.log_lambda == 0.0

    def test_log_lambda_increases_with_drift_magnitude(self):
        from shadow.statistical.sprt import MSPRTDetector

        det_small = MSPRTDetector(alpha=0.05, tau=1.0, warmup=3)
        det_large = MSPRTDetector(alpha=0.05, tau=1.0, warmup=3)
        for v in [-1.0, 0.0, 1.0]:
            det_small.update(v)
            det_large.update(v)
        # Same number of post-warmup steps; different magnitudes.
        for _ in range(10):
            det_small.update(0.5)
            det_large.update(2.0)
        assert det_large.log_lambda > det_small.log_lambda


# ---- MSPRTtDetector (variance-adaptive) -------------------------------------


class TestMSPRTtDetector:
    def test_warmup_returns_continue(self):
        from shadow.statistical.sprt import MSPRTtDetector

        det = MSPRTtDetector(alpha=0.05, tau=1.0, warmup=5)
        for _ in range(5):
            state = det.update(0.0)
            assert state.in_warmup is True
            assert state.decision == "continue"

    def test_invalid_alpha_raises(self):
        from shadow.statistical.sprt import MSPRTtDetector

        with pytest.raises(ValueError, match="alpha"):
            MSPRTtDetector(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            MSPRTtDetector(alpha=1.0)

    def test_invalid_tau_raises(self):
        from shadow.statistical.sprt import MSPRTtDetector

        with pytest.raises(ValueError, match="tau"):
            MSPRTtDetector(tau=0.0)

    def test_invalid_warmup_raises(self):
        from shadow.statistical.sprt import MSPRTtDetector

        with pytest.raises(ValueError, match="warmup"):
            MSPRTtDetector(warmup=1)

    def test_running_variance_updates_per_observation(self):
        """Welford running variance should match np.var(ddof=1) on the
        sample at each step."""
        from shadow.statistical.sprt import MSPRTtDetector

        det = MSPRTtDetector(warmup=5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        for v in values:
            det.update(v)
            n = det.n_observations
            if n >= 2:
                expected = float(np.var(values[:n], ddof=1))
                assert abs(det.running_variance - expected) < 1e-9

    def test_detects_strong_drift(self):
        from shadow.statistical.sprt import MSPRTtDetector

        rng = __import__("random").Random(0)
        det = MSPRTtDetector(alpha=0.05, tau=1.0, warmup=10)
        # Warmup with N(0, 1)
        for _ in range(10):
            det.update(rng.gauss(0, 1))
        # Then large shift
        for _ in range(50):
            state = det.update(rng.gauss(5.0, 1))
            if state.decision == "h1":
                break
        assert det.decision == "h1"

    def test_decision_is_absorbing(self):
        from shadow.statistical.sprt import MSPRTtDetector

        rng = __import__("random").Random(1)
        det = MSPRTtDetector(alpha=0.05, tau=1.0, warmup=5)
        for _ in range(5):
            det.update(rng.gauss(0, 1))
        # Drive to h1
        for _ in range(50):
            det.update(10.0)
            if det.decision == "h1":
                break
        assert det.decision == "h1"
        # Subsequent observations close to mean must NOT flip the decision.
        for _ in range(20):
            state = det.update(0.0)
            assert state.decision == "h1"

    def test_reset_clears_all_state(self):
        from shadow.statistical.sprt import MSPRTtDetector

        det = MSPRTtDetector(warmup=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            det.update(v)
        det.reset()
        assert det.n_observations == 0
        assert det.decision == "continue"
        assert det.log_lambda == 0.0
        assert det.running_variance == 0.0

    def test_threshold_is_log_one_over_alpha(self):
        from shadow.statistical.sprt import MSPRTtDetector

        det = MSPRTtDetector(alpha=0.05)
        assert abs(det.threshold - math.log(1.0 / 0.05)) < 1e-9


# ---- MultiSPRT --------------------------------------------------------------


class TestMultiSPRT:
    def test_any_drift_detected_when_one_axis_crosses_h1(self):
        multi = MultiSPRT(["axis_a", "axis_b"], warmup=3)
        for _ in range(3):
            multi.update({"axis_a": 0.0, "axis_b": 0.0})
        # Push axis_a into H1 territory.
        for _ in range(50):
            multi.update({"axis_a": 10.0, "axis_b": 0.0})
            if multi.any_drift_detected:
                break
        assert multi.any_drift_detected

    def test_all_null_accepted_when_no_drift(self):
        # Feed observations exactly at the calibrated null mean → each
        # update contributes the deterministic drift -δ²/(2σ²) per step,
        # driving log_LR monotonically toward log_A. This isolates the
        # absorbing-decision behavior from warmup-calibration noise.
        multi = MultiSPRT(["axis_a", "axis_b"], effect_size=0.5, warmup=5)
        # Warmup with a fixed non-zero variance so σ̂ is well-defined.
        warm = [-0.5, -0.25, 0.0, 0.25, 0.5]  # mean=0, std≈0.4
        for v in warm:
            multi.update({"axis_a": v, "axis_b": v})
        # Stream observations at the warmup mean. Per-step increment is
        # exactly -δ²/(2σ²) < 0, so log_LR strictly decreases.
        for _ in range(200):
            multi.update({"axis_a": 0.0, "axis_b": 0.0})
            if multi.all_null_accepted:
                break
        assert multi.all_null_accepted

    def test_reset_all_resets_all_detectors(self):
        multi = MultiSPRT(["x", "y"], warmup=3)
        for _ in range(10):
            multi.update({"x": 1.0, "y": 1.0})
        multi.reset_all()
        assert not multi.any_drift_detected
        assert not multi.all_null_accepted


# ---- integration: fingerprint → Hotelling ----------------------------------


class TestFingerprintToHotelling:
    def test_same_agent_no_drift(self):
        rng = np.random.default_rng(99)

        # Build two traces from the same distribution.
        def _random_trace(n: int) -> list[dict]:
            responses = []
            for _ in range(n):
                out_tokens = int(rng.integers(80, 200))
                latency = float(rng.uniform(200, 800))
                responses.append(
                    _make_response(
                        stop_reason="end_turn",
                        output_tokens=out_tokens,
                        latency_ms=latency,
                    )
                )
            return _make_trace(responses)

        x1 = fingerprint_trace(_random_trace(30))
        x2 = fingerprint_trace(_random_trace(30))
        result = hotelling_t2(x1, x2, alpha=0.05)
        # p > 0.05 for same-distribution traces (may occasionally fail
        # but expectation is no rejection).  We use a loose check.
        assert result.p_value > 0.001, f"p={result.p_value:.4f} unexpectedly low"

    def test_different_stop_reason_shifts_distribution(self):
        rng = np.random.default_rng(7)

        def _trace_end_turn(n: int) -> list[dict]:
            return _make_trace(
                [
                    _make_response(
                        stop_reason="end_turn",
                        output_tokens=int(rng.integers(80, 200)),
                        latency_ms=float(rng.uniform(200, 800)),
                    )
                    for _ in range(n)
                ]
            )

        def _trace_tool_use(n: int) -> list[dict]:
            return _make_trace(
                [
                    _make_response(
                        stop_reason="tool_use",
                        tool_names=["search"],
                        output_tokens=int(rng.integers(80, 200)),
                        latency_ms=float(rng.uniform(200, 800)),
                    )
                    for _ in range(n)
                ]
            )

        x1 = fingerprint_trace(_trace_end_turn(40))
        x2 = fingerprint_trace(_trace_tool_use(40))
        result = hotelling_t2(x1, x2, alpha=0.05)
        assert result.reject_null, (
            f"Expected drift detected between end_turn and tool_use traces; "
            f"p={result.p_value:.4f}"
        )
