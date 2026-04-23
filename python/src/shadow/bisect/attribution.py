"""LASSO attribution: which delta caused which axis to move."""

from __future__ import annotations

from typing import Any

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
        # If the axis doesn't vary across corners OR has any NaN/Inf
        # (which would make np.ptp also NaN and poison the subsequent
        # LASSO fit), return zero attributions.
        if not np.all(np.isfinite(y)) or np.ptp(y) < 1e-12:
            out[axis_name] = [(label, 0.0) for label in delta_labels]
            continue
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10_000)
        model.fit(design.astype(float), y)
        coefs = np.abs(model.coef_)
        if not np.all(np.isfinite(coefs)):
            # LASSO converged to NaN (extremely rare — numerical instability).
            out[axis_name] = [(label, 0.0) for label in delta_labels]
            continue
        total = float(coefs.sum())
        normalized = coefs / total if total > 1e-12 else coefs
        ranked = sorted(
            zip(delta_labels, normalized.tolist(), strict=True),
            key=lambda p: -p[1],
        )
        out[axis_name] = list(ranked)
    return out


def rank_attributions_with_ci(
    design: NDArray[np.int8] | NDArray[np.float64],
    divergence: NDArray[np.float64],
    delta_labels: list[str],
    alpha: float = 0.01,
    n_bootstrap: int = 1000,
    seed: int = 42,
    stability_threshold: float = 0.6,
    lambda_path: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1),
) -> dict[str, list[dict[str, Any]]]:
    """LASSO attribution with bootstrap CIs **and** stability-selection significance.

    Returns per-axis rows with:
      - `weight`: median normalised |coef| across bootstrap resamples
      - `ci95_low`/`ci95_high`: 2.5/97.5 percentile of normalised weights
      - `significant`: True iff selected in >= `stability_threshold` of
         Meinshausen-Bühlmann subsamples across any λ in the path

    Why both? The percentile bootstrap produces **zero-length CIs for
    weakly-nonzero coefficients** (LASSO zeroes them out in most
    resamples), which hides real effects. Stability Selection
    (Meinshausen-Bühlmann 2010) fixes that by using subsamples of size
    n/2 and reporting selection frequency across a λ path — true
    effects get selected at many λ, noise gets selected at few.

    The `weight` + CI fields are kept for ranking / display. The
    `significant` flag is the correct inferential signal.
    """
    runs, k = design.shape
    if divergence.shape != (runs, len(AXIS_NAMES)):
        raise ValueError(f"divergence shape {divergence.shape} != ({runs}, {len(AXIS_NAMES)})")
    if len(delta_labels) != k:
        raise ValueError(f"delta_labels length {len(delta_labels)} != k={k}")
    rng = np.random.default_rng(seed)
    design_f = design.astype(float)

    out: dict[str, list[dict[str, Any]]] = {}
    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        y = divergence[:, axis_idx]
        # Bail on non-finite data OR zero-variance. np.ptp on a NaN-bearing
        # array returns NaN, which makes the <1e-12 comparison false and
        # poisons the subsequent LASSO fit.
        if not np.all(np.isfinite(y)) or np.ptp(y) < 1e-12:
            out[axis_name] = [
                {
                    "delta": label,
                    "weight": 0.0,
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                    "significant": False,
                    "selection_frequency": 0.0,
                }
                for label in delta_labels
            ]
            continue
        # Point estimate from the full sample.
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10_000)
        model.fit(design_f, y)
        point = _normalised_abs(model.coef_)

        # Bootstrap: resample corners with replacement, refit.
        samples = np.zeros((n_bootstrap, k), dtype=float)
        for i in range(n_bootstrap):
            idx = rng.integers(0, runs, size=runs)
            x_b = design_f[idx]
            y_b = y[idx]
            if np.ptp(y_b) < 1e-12:
                # Degenerate resample — skip (weights will be 0).
                continue
            bm = Lasso(alpha=alpha, fit_intercept=True, max_iter=10_000)
            bm.fit(x_b, y_b)
            samples[i] = _normalised_abs(bm.coef_)

        ci_low = np.percentile(samples, 2.5, axis=0)
        ci_high = np.percentile(samples, 97.5, axis=0)

        # Stability selection: for each λ in the path, draw n_subsamples
        # size-n/2 subsamples, fit LASSO, record which coefficients were
        # non-zero. Selection frequency is max across the path.
        selection_freq = _stability_selection(
            design_f, y, rng, lambda_path=lambda_path, n_subsamples=100
        )

        rows: list[dict[str, Any]] = []
        for j, label in enumerate(delta_labels):
            rows.append(
                {
                    "delta": label,
                    "weight": float(point[j]),
                    "ci95_low": float(ci_low[j]),
                    "ci95_high": float(ci_high[j]),
                    # Stability-selection significance: selected at high
                    # frequency across the λ path. Survives the zero-length
                    # CI failure mode of plain bootstrap LASSO.
                    "significant": bool(selection_freq[j] >= stability_threshold),
                    "selection_frequency": float(selection_freq[j]),
                }
            )
        rows.sort(key=lambda r: -r["weight"])
        out[axis_name] = rows
    return out


def _stability_selection(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    lambda_path: tuple[float, ...],
    n_subsamples: int = 100,
) -> NDArray[np.float64]:
    """Meinshausen-Bühlmann stability selection frequencies.

    For each λ in `lambda_path`, draw `n_subsamples` size-`n/2` random
    subsamples of the rows, fit LASSO on each, and count the fraction
    of subsamples in which each coefficient is non-zero. Return the
    per-coefficient MAX frequency across the λ path.

    Reference: Meinshausen & Bühlmann (2010), "Stability selection",
    J. R. Statist. Soc. B 72: 417-473.
    """
    n, k = design.shape
    if n < 4:
        # Subsampling to n/2 < 2 is statistically meaningless; bail.
        empty: NDArray[np.float64] = np.zeros(k, dtype=np.float64)
        return empty
    sub_size = n // 2
    # Accumulate selection indicators per λ; take max frequency across λ.
    per_lambda = np.zeros((len(lambda_path), k), dtype=np.float64)
    for lam_idx, lam in enumerate(lambda_path):
        counts = np.zeros(k, dtype=np.float64)
        for _ in range(n_subsamples):
            # Subsample WITHOUT replacement (MB prescribes).
            idx = rng.choice(n, size=sub_size, replace=False)
            x_b = design[idx]
            y_b = y[idx]
            if np.ptp(y_b) < 1e-12:
                continue
            try:
                mb = Lasso(alpha=lam, fit_intercept=True, max_iter=10_000)
                mb.fit(x_b, y_b)
                counts += (np.abs(mb.coef_) > 1e-8).astype(np.float64)
            except Exception:
                continue
        per_lambda[lam_idx] = counts / n_subsamples
    result: NDArray[np.float64] = per_lambda.max(axis=0).astype(np.float64)
    return result


def _normalised_abs(coefs: NDArray[np.float64]) -> NDArray[np.float64]:
    """abs(coefs) normalised to sum to 1 (or all-zero if no signal).

    NaN-safe: any NaN in the input produces an all-zero vector rather
    than poisoning downstream bootstrap CIs with NaN weights.
    """
    abs_coefs: NDArray[np.float64] = np.abs(coefs).astype(np.float64)
    if not np.all(np.isfinite(abs_coefs)):
        return np.zeros_like(abs_coefs)
    total = float(abs_coefs.sum())
    if total < 1e-12:
        return abs_coefs
    return (abs_coefs / total).astype(np.float64)
