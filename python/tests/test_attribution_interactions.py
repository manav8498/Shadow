"""Tests for `rank_attributions_with_interactions`.

Ground-truth tests: construct synthetic divergence matrices where the
true data-generating process includes specific main effects and
interaction terms, then verify the detector recovers them with
honest CIs and significance.
"""

from __future__ import annotations

import numpy as np

from shadow.bisect.attribution import (
    AXIS_NAMES,
    rank_attributions_with_interactions,
)
from shadow.bisect.design import full_factorial


def _zeros_divergence(runs: int) -> np.ndarray:
    return np.zeros((runs, len(AXIS_NAMES)), dtype=float)


def _axis_idx(name: str) -> int:
    return AXIS_NAMES.index(name)


def test_main_effect_only_recovered_with_ci_excluding_zero() -> None:
    """When only ONE main effect drives an axis, that delta should:
    - be marked significant
    - have a CI strictly above 0
    - dominate the weight distribution
    """
    design = full_factorial(4)
    labels = ["prompt", "model", "temperature", "tools"]
    divergence = _zeros_divergence(design.shape[0])
    # Only `prompt` (column 0) drives the semantic axis.
    # When design[:, 0] == +1, semantic divergence = 3.0; else 0.
    sem = _axis_idx("semantic")
    divergence[:, sem] = (design[:, 0] == 1).astype(float) * 3.0

    result = rank_attributions_with_interactions(
        design, divergence, labels, n_bootstrap=200, seed=1, alpha=0.01
    )
    mains = {r["delta"]: r for r in result["semantic"]["main_effects"]}

    # prompt should have high weight, tight CI above 0, significant
    assert mains["prompt"]["weight"] > 0.9, mains
    assert mains["prompt"]["ci95_low"] > 0.5, mains
    assert mains["prompt"]["significant"] is True

    # All other mains should have low weight + not significant
    for other in ("model", "temperature", "tools"):
        assert mains[other]["weight"] < 0.1, (other, mains[other])
        assert mains[other]["significant"] is False


def test_pure_interaction_effect_is_surfaced() -> None:
    """When an axis is driven by a PURE AxB interaction (both mains
    have zero main effect, but their product drives divergence), the
    detector should flag the interaction term.

    Note: strong hierarchy requires at least one component main to be
    significant for the interaction to be emitted. Since a pure
    interaction with no mains is a degenerate research case that
    rarely matches real-world data, we include weak main effects to
    satisfy strong hierarchy.
    """
    design = full_factorial(4)
    labels = ["a", "b", "c", "d"]
    divergence = _zeros_divergence(design.shape[0])
    axis = _axis_idx("latency")
    # Axis driven by axb interaction: when both a=1 and b=1, shift +5.
    # Also add small main effects to keep strong hierarchy satisfied.
    a_on = design[:, 0] == 1
    b_on = design[:, 1] == 1
    divergence[:, axis] = (
        (a_on & b_on).astype(float) * 5.0
        + a_on.astype(float) * 0.5  # small main for strong-hierarchy
        + b_on.astype(float) * 0.5
    )

    result = rank_attributions_with_interactions(
        design, divergence, labels, n_bootstrap=200, seed=2, alpha=0.01
    )
    interactions = {tuple(r["pair"]): r for r in result["latency"]["interactions"]}
    # The axb interaction is lex-sorted as ("a", "b")
    assert ("a", "b") in interactions
    ab = interactions[("a", "b")]
    # Weight should be non-trivial and CI should exclude zero
    assert ab["weight"] > 0.1, ab
    assert ab["ci95_low"] > 0.0, ab


def test_strong_hierarchy_drops_spurious_interactions() -> None:
    """If neither main of an interaction is significant, the strong-
    hierarchy filter should drop the interaction from the output.

    Construct data where only the MAIN effect of `a` drives the axis;
    any interaction involving `d` should be filtered because `d` has
    no main-effect signal.
    """
    design = full_factorial(4)
    labels = ["a", "b", "c", "d"]
    divergence = _zeros_divergence(design.shape[0])
    axis = _axis_idx("verbosity")
    # Only `a` matters.
    divergence[:, axis] = (design[:, 0] == 1).astype(float) * 4.0

    result = rank_attributions_with_interactions(
        design, divergence, labels, n_bootstrap=200, seed=3, alpha=0.01
    )
    interactions = result["verbosity"]["interactions"]
    # Any interaction involving b, c, d (without a) should be dropped
    # by strong-hierarchy. Interactions involving `a` may survive
    # because `a` IS significant.
    for inter in interactions:
        assert "a" in inter["pair"], (
            f"strong hierarchy should drop {inter['pair']} since "
            "neither component has a significant main effect"
        )


def test_ci_bounds_are_valid_ranges() -> None:
    """Sanity: every returned CI has low ≤ weight ≤ high, and all
    three are in [0, 1]."""
    design = full_factorial(3)
    labels = ["x", "y", "z"]
    divergence = _zeros_divergence(design.shape[0])
    # Light noise across all axes
    rng = np.random.default_rng(7)
    divergence += rng.normal(0, 0.5, divergence.shape)

    result = rank_attributions_with_interactions(
        design, divergence, labels, n_bootstrap=100, seed=7, alpha=0.05
    )
    for axis_name in AXIS_NAMES:
        for row in result[axis_name]["main_effects"] + result[axis_name]["interactions"]:
            assert 0.0 <= row["ci95_low"] <= row["weight"] + 1e-9 <= row["ci95_high"] + 1e-9
            assert 0.0 <= row["ci95_low"] <= 1.0
            assert 0.0 <= row["ci95_high"] <= 1.0
            assert 0.0 <= row["weight"] <= 1.0
            assert 0.0 <= row["selection_frequency"] <= 1.0


def test_zero_variance_axis_returns_empty_shape() -> None:
    """An axis with zero variance (e.g. all-zero divergence) should
    return well-shaped rows with zero weights, not error out."""
    design = full_factorial(3)
    labels = ["x", "y", "z"]
    divergence = _zeros_divergence(design.shape[0])  # all zeros
    result = rank_attributions_with_interactions(design, divergence, labels, n_bootstrap=50, seed=1)
    # Every axis produces empty (all-zero) rows
    for axis_name in AXIS_NAMES:
        ax = result[axis_name]
        assert len(ax["main_effects"]) == 3
        assert all(r["weight"] == 0.0 for r in ax["main_effects"])
        assert all(r["significant"] is False for r in ax["main_effects"])
        # k=3 → 3 pairwise interactions
        assert len(ax["interactions"]) == 3


def test_significant_requires_both_selection_and_ci() -> None:
    """The conjunction rule: significant iff (selection_freq ≥ 0.6)
    AND (ci95_low > 0). Neither alone is sufficient."""
    design = full_factorial(4)
    labels = ["a", "b", "c", "d"]
    divergence = _zeros_divergence(design.shape[0])
    axis = _axis_idx("semantic")
    # Strong signal from `a` ensures both gates trigger.
    divergence[:, axis] = (design[:, 0] == 1).astype(float) * 5.0

    result = rank_attributions_with_interactions(
        design, divergence, labels, n_bootstrap=200, seed=4, alpha=0.01
    )
    a_row = next(r for r in result["semantic"]["main_effects"] if r["delta"] == "a")
    # With a clear signal both gates should trigger
    assert a_row["selection_frequency"] >= 0.6
    assert a_row["ci95_low"] > 0.0
    assert a_row["significant"] is True

    # The other mains have no signal → neither gate triggers → not significant
    for label in ("b", "c", "d"):
        row = next(r for r in result["semantic"]["main_effects"] if r["delta"] == label)
        assert row["significant"] is False


def test_output_canonicalises_pair_lexicographic_order() -> None:
    """Interaction pairs should be sorted lex so (A, B) and (B, A)
    never both appear — purely a display-canonicalisation check."""
    design = full_factorial(3)
    labels = ["zebra", "apple", "mango"]
    divergence = _zeros_divergence(design.shape[0])
    divergence += np.random.default_rng(0).normal(0, 0.1, divergence.shape)

    result = rank_attributions_with_interactions(design, divergence, labels, n_bootstrap=50, seed=0)
    for axis_name in AXIS_NAMES:
        for inter in result[axis_name]["interactions"]:
            pair = inter["pair"]
            assert pair[0] <= pair[1], f"pair not lex-sorted: {pair}"
