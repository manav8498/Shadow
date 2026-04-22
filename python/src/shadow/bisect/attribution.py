"""LASSO attribution: which delta caused which axis to move."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Lasso  # type: ignore[import-untyped]

AXIS_NAMES = (
    "semantic",
    "trajectory",
    "safety",
    "verbosity",
    "latency",
    "cost",
    "reasoning",
    "judge",
    "conformance",
)


def rank_attributions(
    design: NDArray[np.int8] | NDArray[np.float64],
    divergence: NDArray[np.float64],
    delta_labels: list[str],
    alpha: float = 0.01,
) -> dict[str, list[tuple[str, float]]]:
    """Fit a LASSO per axis and return attribution weights.

    Parameters
    ----------
    design:
        `(runs, k)` design matrix with entries in `{-1, +1}` (from
        [`full_factorial`][] or [`plackett_burman`][]).
    divergence:
        `(runs, 9)` matrix of per-axis divergence scores for each
        corner configuration.
    delta_labels:
        Length-`k` list of human-readable delta identifiers (e.g.
        `params.temperature`).
    alpha:
        LASSO regularization strength. Defaults to 0.01.

    Returns
    -------
    dict[axis_name, list[(delta_label, normalized_weight)]]
        Weights are normalized so that the sum of absolute weights per
        axis is 1 (or 0 if no delta has any effect). Sorted descending
        by absolute weight.
    """
    runs, k = design.shape
    if divergence.shape != (runs, len(AXIS_NAMES)):
        raise ValueError(f"divergence shape {divergence.shape} != ({runs}, {len(AXIS_NAMES)})")
    if len(delta_labels) != k:
        raise ValueError(f"delta_labels length {len(delta_labels)} != k={k}")
    out: dict[str, list[tuple[str, float]]] = {}
    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        y = divergence[:, axis_idx]
        # If the axis doesn't vary across corners, return zero attributions.
        if np.ptp(y) < 1e-12:
            out[axis_name] = [(label, 0.0) for label in delta_labels]
            continue
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10_000)
        model.fit(design.astype(float), y)
        coefs = np.abs(model.coef_)
        total = float(coefs.sum())
        normalized = coefs / total if total > 1e-12 else coefs
        ranked = sorted(
            zip(delta_labels, normalized.tolist(), strict=True),
            key=lambda p: -p[1],
        )
        out[axis_name] = list(ranked)
    return out
