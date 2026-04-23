"""LASSO attribution: which delta caused which axis to move.

## Three levels of attribution shipped

1. **`rank_attributions`** — plain LASSO point estimate, backward-compat API.
2. **`rank_attributions_with_ci`** — adds bootstrap CIs and Meinshausen-
   Bühlmann stability selection (2010). Significance = selection
   frequency ≥ threshold.
3. **`rank_attributions_with_interactions`** (new) — hardened engine
   that also fits **pairwise interactions** (delta A x delta B). Uses
   residual bootstrap (Chatterjee & Lahiri, Annals of Stats 2011) which
   avoids the point-mass-at-0 pathology of pairs bootstrap on LASSO
   coefficients. Alpha is fixed by outer LassoCV once on the original
   data — re-tuning per resample artificially inflates CI width
   (Efron 2014, JASA). Normalization happens **per resample** (before
   percentile) to preserve bootstrap CI validity; normalizing point
   estimates then bootstrapping breaks CI independence. A strong-
   hierarchy post-filter drops interactions AxB where neither main
   effect survived stability selection (Lim & Hastie 2015,
   *glinternet*-style).

## Significance rule (conjunction — screening + magnitude)

A term is `significant` iff BOTH:

  - stability-selection frequency ≥ `stability_threshold` (default 0.6)
  - bootstrap 95% CI excludes 0 (lower bound > 0)

This is a screening-plus-magnitude conjunction, not two independent
tests; documented as such so users don't treat it as a multiplicity-
adjusted p-value. Reference: R `stabs` package, Meinshausen-Bühlmann
(2010), Dezeure et al. (2017, *Statistical Science*).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Lasso, LassoCV  # type: ignore[import-untyped]
from sklearn.preprocessing import PolynomialFeatures  # type: ignore[import-untyped]

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


# ---------------------------------------------------------------------------
# Hardened engine with interactions + residual bootstrap
# ---------------------------------------------------------------------------


def rank_attributions_with_interactions(
    design: NDArray[np.int8] | NDArray[np.float64],
    divergence: NDArray[np.float64],
    delta_labels: list[str],
    alpha: float | None = None,
    n_bootstrap: int = 500,
    seed: int = 42,
    stability_threshold: float = 0.6,
    lambda_path: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1),
    strong_hierarchy: bool = True,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Hardened LASSO attribution with pairwise interactions + residual-
    bootstrap CIs + strong-hierarchy filter.

    Parameters
    ----------
    design, divergence, delta_labels :
        Same shape contract as [`rank_attributions`]. `design` is the
        main-effect design matrix (corners x k); interactions are
        derived internally via [`sklearn.preprocessing.PolynomialFeatures`].
    alpha :
        LASSO regularization strength for the point fit AND every
        bootstrap resample. If `None`, picked once via [`LassoCV`] on
        the original data — re-tuning per resample inflates CI width
        artificially (Efron 2014).
    n_bootstrap :
        Number of residual-bootstrap resamples. 500 is a reasonable
        default for 95% CI stability; 1000 for tighter bounds if the
        noise floor is suspect.
    stability_threshold :
        Meinshausen-Bühlmann selection-frequency threshold for the
        "screening" half of the `significant` conjunction.
    lambda_path :
        λ grid used by stability selection.
    strong_hierarchy :
        When True (default), drop an interaction AxB from the output
        if NEITHER of its component main effects survived stability
        selection. Strong hierarchy is the most-cited heuristic for
        tamed interaction selection (Lim & Hastie 2015, *glinternet*).

    Returns
    -------
    dict[axis_name, {"main_effects": [...], "interactions": [...]}]
        Each entry in `main_effects` and `interactions` has the fields
        `weight`, `ci95_low`, `ci95_high`, `significant`,
        `selection_frequency`, and a label:
          - main effects: `"delta"` names the single delta (e.g.
            `"params.temperature"`)
          - interactions: `"pair"` holds the lex-sorted tuple of the
            two interacting delta labels (e.g. `["model", "prompt"]`)
        Weights are raw |coef| normalised **per resample** and then
        summarised, NOT normalised-then-bootstrapped (which would
        break CI validity).
    """
    runs, k = design.shape
    if divergence.shape != (runs, len(AXIS_NAMES)):
        raise ValueError(f"divergence shape {divergence.shape} != ({runs}, {len(AXIS_NAMES)})")
    if len(delta_labels) != k:
        raise ValueError(f"delta_labels length {len(delta_labels)} != k={k}")

    # Augment the design matrix with pairwise interactions. The
    # `interaction_only=True` flag tells PolynomialFeatures to emit
    # X_i * X_j terms WITHOUT squared terms X_i^2 (which for ±1-valued
    # design columns are always 1 and thus useless).
    design_f = design.astype(float)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    design_aug = poly.fit_transform(design_f)
    # Build human-readable labels aligned with augmented columns.
    # PolynomialFeatures returns main effects first, then interactions
    # in lex index order (0*1, 0*2, ..., (k-2)*(k-1)).
    labels: list[str] = list(delta_labels)
    interaction_pairs: list[tuple[str, str]] = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = delta_labels[i], delta_labels[j]
            # Canonicalise pair display by lex order so (A, B) and
            # (B, A) are never confused for distinct interactions.
            pair = (a, b) if a < b else (b, a)
            interaction_pairs.append(pair)
            labels.append(f"{pair[0]} x {pair[1]}")
    # Total feature count including interactions.
    p = design_aug.shape[1]
    assert p == k + len(
        interaction_pairs
    ), f"feature count mismatch: {p} vs {k + len(interaction_pairs)}"

    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        y = divergence[:, axis_idx]
        # Guard against non-finite / zero-variance — return empty per-axis.
        if not np.all(np.isfinite(y)) or np.ptp(y) < 1e-12:
            out[axis_name] = _empty_axis(delta_labels, interaction_pairs)
            continue

        # 1. Fix alpha once on the original data via LassoCV.
        fixed_alpha = alpha if alpha is not None else _pick_alpha(design_aug, y, lambda_path)

        # 2. Point-estimate fit on the full data.
        point_model = Lasso(alpha=fixed_alpha, fit_intercept=True, max_iter=10_000)
        point_model.fit(design_aug, y)
        point_coefs = point_model.coef_
        point_residuals = y - point_model.predict(design_aug)
        # Per-resample normalisation of the point estimate for display.
        point_norm = _normalised_abs(point_coefs)

        # 3. Residual bootstrap — resample residuals with replacement,
        #    add them to the fitted values, refit LASSO at the SAME alpha.
        #    This produces an honest bootstrap distribution of LASSO
        #    coefficients without the pairs-bootstrap pathology.
        samples = np.zeros((n_bootstrap, p), dtype=float)
        fitted = point_model.predict(design_aug)
        for i in range(n_bootstrap):
            eps_b = rng.choice(point_residuals, size=runs, replace=True)
            y_b = fitted + eps_b
            if np.ptp(y_b) < 1e-12:
                continue
            bm = Lasso(alpha=fixed_alpha, fit_intercept=True, max_iter=10_000)
            bm.fit(design_aug, y_b)
            # Per-resample normalisation, THEN aggregate — NOT the other
            # way around. Normalising point estimates first then bootstrapping
            # residuals breaks CI independence (research brief §7).
            samples[i] = _normalised_abs(bm.coef_)

        ci_low = np.percentile(samples, 2.5, axis=0)
        ci_high = np.percentile(samples, 97.5, axis=0)

        # 4. Stability selection on the augmented design.
        selection_freq = _stability_selection(
            design_aug, y, rng, lambda_path=lambda_path, n_subsamples=100
        )

        # 5. Assemble per-term rows, split into main + interaction.
        main_rows: list[dict[str, Any]] = []
        inter_rows: list[dict[str, Any]] = []
        for idx in range(p):
            ci_excludes_zero = ci_low[idx] > 1e-9
            # Conjunction rule: significant iff BOTH gates agree.
            is_sig = bool(selection_freq[idx] >= stability_threshold and ci_excludes_zero)
            row: dict[str, Any] = {
                "weight": float(point_norm[idx]),
                "ci95_low": float(ci_low[idx]),
                "ci95_high": float(ci_high[idx]),
                "significant": is_sig,
                "selection_frequency": float(selection_freq[idx]),
            }
            if idx < k:
                row["delta"] = delta_labels[idx]
                main_rows.append(row)
            else:
                pair = interaction_pairs[idx - k]
                row["pair"] = [pair[0], pair[1]]
                row["label"] = labels[idx]
                inter_rows.append(row)

        # 6. Strong-hierarchy post-filter: drop an interaction row if
        #    NEITHER of its two main effects is significant. This mirrors
        #    glinternet's "strong hierarchy" constraint and kills the
        #    main class of spurious interactions at small-n.
        if strong_hierarchy and inter_rows:
            sig_mains = {r["delta"] for r in main_rows if r["significant"]}
            inter_rows = [
                r for r in inter_rows if (r["pair"][0] in sig_mains or r["pair"][1] in sig_mains)
            ]

        # 7. Sort each list by weight desc for display.
        main_rows.sort(key=lambda r: -r["weight"])
        inter_rows.sort(key=lambda r: -r["weight"])
        out[axis_name] = {"main_effects": main_rows, "interactions": inter_rows}

    return out


def _pick_alpha(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
    lambda_path: tuple[float, ...],
) -> float:
    """Pick alpha for LASSO via LassoCV on the original data.

    Held constant across all bootstrap resamples to avoid CI-width
    inflation (Efron 2014, *Estimation and Accuracy After Model
    Selection*, JASA).
    """
    n = design.shape[0]
    # cv folds must be ≥ 2 and ≤ n; clamp for small designs.
    cv = max(2, min(5, n))
    try:
        model = LassoCV(alphas=list(lambda_path), cv=cv, max_iter=10_000)
        model.fit(design, y)
        a = float(model.alpha_)
        # Fallback to middle of path on pathological data.
        if not np.isfinite(a) or a <= 0.0:
            return float(lambda_path[len(lambda_path) // 2])
        return a
    except Exception:
        # Any numeric / shape issue falls back to a safe middle-of-path
        # value rather than propagating a CV-specific exception.
        return float(lambda_path[len(lambda_path) // 2])


def _empty_axis(
    delta_labels: list[str], interaction_pairs: list[tuple[str, str]]
) -> dict[str, list[dict[str, Any]]]:
    """Degenerate output for an axis with non-finite or zero-variance y.

    Same shape as a normal response so downstream consumers don't
    branch on missing keys.
    """
    mains = [
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
    inters = [
        {
            "pair": [a, b],
            "label": f"{a} x {b}",
            "weight": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "significant": False,
            "selection_frequency": 0.0,
        }
        for (a, b) in interaction_pairs
    ]
    return {"main_effects": mains, "interactions": inters}
