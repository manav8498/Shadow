"""Tests for the bootstrapped LASSO attribution with 95% CIs."""

from __future__ import annotations

import numpy as np

from shadow.bisect.attribution import AXIS_NAMES, rank_attributions_with_ci
from shadow.bisect.design import full_factorial


def _build_divergence(
    design: np.ndarray, driver_idx: int, axis_idx: int, scale: float = 1.0
) -> np.ndarray:
    """Synthesize a divergence matrix where only `driver_idx` drives `axis_idx`."""
    runs = design.shape[0]
    div = np.zeros((runs, len(AXIS_NAMES)), dtype=float)
    for i in range(runs):
        if design[i, driver_idx] == 1:
            div[i, axis_idx] = scale
    return div


def test_ci_shape_matches_labels() -> None:
    design = full_factorial(2)
    labels = ["a", "b"]
    div = np.random.default_rng(0).random((design.shape[0], len(AXIS_NAMES)))
    result = rank_attributions_with_ci(design, div, labels, seed=1, n_bootstrap=50)
    assert set(result) == set(AXIS_NAMES)
    for axis in AXIS_NAMES:
        assert len(result[axis]) == 2
        for row in result[axis]:
            assert {"delta", "weight", "ci95_low", "ci95_high", "significant"} <= row.keys()


def test_ci_marks_true_driver_as_significant() -> None:
    # 3 deltas; delta #1 is the only cause of latency movement.
    design = full_factorial(3)
    labels = ["model", "prompt", "tools"]
    latency_idx = AXIS_NAMES.index("latency")
    div = _build_divergence(design, driver_idx=0, axis_idx=latency_idx, scale=10.0)
    result = rank_attributions_with_ci(design, div, labels, seed=1, n_bootstrap=200)
    latency_rows = {r["delta"]: r for r in result["latency"]}
    assert latency_rows["model"]["significant"] is True
    assert latency_rows["model"]["ci95_low"] > 0.5  # dominant driver
    # The non-drivers should be non-significant.
    assert latency_rows["prompt"]["significant"] is False
    assert latency_rows["tools"]["significant"] is False


def test_ci_zero_divergence_axis_returns_zero_weights_non_significant() -> None:
    design = full_factorial(2)
    labels = ["a", "b"]
    div = np.zeros((design.shape[0], len(AXIS_NAMES)), dtype=float)
    result = rank_attributions_with_ci(design, div, labels, seed=1, n_bootstrap=50)
    for row in result["latency"]:
        assert row["weight"] == 0.0
        assert row["ci95_low"] == 0.0
        assert row["ci95_high"] == 0.0
        assert row["significant"] is False


def test_stability_selection_flags_true_driver_not_noise() -> None:
    """Meinshausen-Bühlmann selection frequency should be high for the true
    driver and low for the noise categories — a proper inferential signal
    that survives LASSO's zero-length-CI failure mode."""
    design = full_factorial(4)  # n=16 → sub_size=8 (MB-compatible)
    labels = ["model", "prompt", "params", "tools"]
    axis_idx = AXIS_NAMES.index("latency")
    # Only `tools` drives the axis; every other column is pure noise.
    div = _build_divergence(design, driver_idx=3, axis_idx=axis_idx, scale=10.0)
    result = rank_attributions_with_ci(
        design, div, labels, seed=1, n_bootstrap=100, stability_threshold=0.6
    )
    rows_by_cat = {r["delta"]: r for r in result["latency"]}
    assert rows_by_cat["tools"]["significant"] is True
    assert rows_by_cat["tools"]["selection_frequency"] >= 0.9
    for noise_cat in ("model", "prompt", "params"):
        assert rows_by_cat[noise_cat]["significant"] is False
        assert rows_by_cat[noise_cat]["selection_frequency"] < 0.6


def test_stability_selection_zero_variance_axes_get_zero_frequency() -> None:
    design = full_factorial(3)
    labels = ["a", "b", "c"]
    div = np.zeros((design.shape[0], len(AXIS_NAMES)), dtype=float)
    result = rank_attributions_with_ci(design, div, labels, seed=1, n_bootstrap=50)
    for row in result["latency"]:
        assert row["selection_frequency"] == 0.0
        assert row["significant"] is False


def test_ci_rows_sorted_descending_by_weight() -> None:
    design = full_factorial(3)
    labels = ["a", "b", "c"]
    axis_idx = AXIS_NAMES.index("verbosity")
    div = _build_divergence(design, driver_idx=2, axis_idx=axis_idx, scale=5.0)
    result = rank_attributions_with_ci(design, div, labels, seed=1, n_bootstrap=100)
    weights = [r["weight"] for r in result["verbosity"]]
    assert weights == sorted(weights, reverse=True)
