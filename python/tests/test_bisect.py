"""Tests for shadow.bisect (Phase 4)."""

from __future__ import annotations

import numpy as np

from shadow.bisect import Delta, diff_configs, full_factorial, plackett_burman
from shadow.bisect.attribution import AXIS_NAMES, rank_attributions
from shadow.bisect.design import is_orthogonal
from shadow.bisect.runner import choose_design, score_corners_synthetic


def test_diff_configs_finds_leaf_differences() -> None:
    a = {"model": "claude-opus-4-7", "params": {"temperature": 0.2}}
    b = {"model": "claude-opus-4-7", "params": {"temperature": 0.7, "top_p": 0.9}}
    deltas = diff_configs(a, b)
    # Two differing leaves: params.temperature and params.top_p.
    paths = {d.path for d in deltas}
    assert paths == {"params.temperature", "params.top_p"}
    by_path = {d.path: d for d in deltas}
    assert by_path["params.temperature"].old_value == 0.2
    assert by_path["params.temperature"].new_value == 0.7
    assert by_path["params.top_p"].old_value is None  # added in b
    assert by_path["params.top_p"].new_value == 0.9


def test_diff_configs_identical_returns_empty() -> None:
    a = {"model": "x", "params": {"t": 0.2}}
    assert diff_configs(a, a) == []


def test_full_factorial_k2_matches_manual_table() -> None:
    # Expected 4 rows x 2 cols: (-1,-1), (1,-1), (-1,1), (1,1).
    expected = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.int8)
    assert np.array_equal(full_factorial(2), expected)


def test_full_factorial_k_up_to_6_has_right_shape() -> None:
    for k in range(1, 7):
        d = full_factorial(k)
        assert d.shape == (1 << k, k)
        # Orthogonality: X^T X = runs * I
        assert is_orthogonal(d)


def test_full_factorial_rejects_k_gt_6() -> None:
    import pytest

    with pytest.raises(ValueError, match="capped at k=6"):
        full_factorial(7)


def test_plackett_burman_k8_has_12_runs_and_is_orthogonal() -> None:
    d = plackett_burman(8)
    assert d.shape == (12, 8)
    assert is_orthogonal(d)


def test_plackett_burman_k7_has_8_runs() -> None:
    d = plackett_burman(7)
    assert d.shape == (8, 7)
    assert is_orthogonal(d)


def test_choose_design_picks_factorial_for_small_k() -> None:
    d = choose_design(3)
    assert d.shape == (8, 3)


def test_choose_design_picks_pb_for_large_k() -> None:
    d = choose_design(8)
    assert d.shape == (12, 8)


def test_ground_truth_lasso_recovers_the_driver() -> None:
    """Synthetic scenario with 8 deltas; only delta #2 moves 'trajectory'.

    The LASSO attribution must give delta #2 the dominant share of the
    trajectory axis (≥0.9) and give every other delta at most 0.05.
    """
    k = 8
    design = plackett_burman(k)
    # Only delta #2 drives the trajectory axis (axis index 1).
    axis_weights = {2: {1: 1.0}}
    divergence = score_corners_synthetic(design, axis_weights)
    labels = [f"delta_{i}" for i in range(k)]
    attributions = rank_attributions(design, divergence, labels, alpha=0.01)

    # Look up trajectory-axis attributions.
    trajectory = dict(attributions["trajectory"])
    # Deltas that shouldn't move trajectory must get ≤ 0.05 weight.
    for i in range(k):
        if i == 2:
            continue
        assert (
            trajectory[f"delta_{i}"] <= 0.05
        ), f"non-driver delta_{i} got weight {trajectory[f'delta_{i}']:.3f}"
    # The actual driver gets ≥ 0.9.
    assert trajectory["delta_2"] >= 0.9

    # All other axes should be all-zero attributions (no signal).
    for axis in AXIS_NAMES:
        if axis == "trajectory":
            continue
        axis_attr = dict(attributions[axis])
        for i in range(k):
            assert axis_attr[f"delta_{i}"] == 0.0


def test_delta_kind_extracts_top_level_prefix() -> None:
    d = Delta(path="params.temperature", old_value=0.2, new_value=0.7)
    assert d.kind == "params"
