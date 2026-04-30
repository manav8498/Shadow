"""Intervention-based causal attribution (Pearl-style do-calculus).

Given a baseline trace, a candidate trace, and a list of named deltas
that distinguish the two, this module:

1. For each delta, constructs a "single-delta candidate" by injecting
   only that delta into the baseline configuration.
2. Replays each single-delta candidate (using a user-supplied replay
   function — the module is interface-only here).
3. Computes the resulting per-axis divergence vector.
4. Reports the ATE per (delta, axis) pair, with optional:
   - Bootstrap percentile CIs (Efron 1979) when ``n_bootstrap > 0``.
   - Back-door adjustment for named confounders (Pearl 2009 §3.3).
   - E-value sensitivity to unmeasured confounding (VanderWeele &
     Ding 2017).

Design notes
------------
- The replay function is opaque: callers wire in their own backend
  (Rust replay, MockLLM, real LLM, recorded fixture). This module
  never assumes anything about how a config maps to a divergence
  vector.
- ``n_replays`` controls noise reduction at the **point estimate**
  level; ``n_bootstrap`` resamples the per-axis distributions to
  produce CIs. The two are independent.
- Back-door adjustment expects ``confounders`` to be config keys
  whose values differ between baseline and candidate. The estimator
  stratifies over the 2^|C| confounder combinations and combines the
  per-stratum ATEs with weights that follow Pearl's formula
  ``ATE = Σ_c P(C=c) · ATE_c``:

    * When ``confounder_weights`` is supplied as a dict mapping
      ``(c_1, c_2, ...)`` tuples to weights, those weights are
      normalised to sum to 1 and applied directly. Use this when
      you have measured P(C=c) in the target population.
    * When ``confounder_weights="uniform"`` is supplied, every
      stratum gets weight ``1/n``. This is unbiased under the
      Pearl formula only when P(C=c) is itself uniform — pass the
      sentinel only when you've explicitly verified that, or when
      you understand the resulting estimate is the simple average.
    * Omitting ``confounder_weights`` while declaring confounders
      is now an error: the estimator refuses to silently assume
      uniform P(C=c). Earlier versions of Shadow defaulted to
      uniform here; that default is removed in v3.0 because it
      produced a biased estimate that read as "the Pearl ATE"
      to callers who never supplied weights.
- E-value uses the VanderWeele-Ding closed form for continuous
  outcomes via the standardized mean difference. The pooled SD
  comes from the union of baseline and intervened replay outputs.

References
----------
Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
Efron, B. (1979). "Bootstrap methods: Another look at the jackknife".
  *Ann. Statist.* 7(1): 1-26.
VanderWeele, T. J. & Ding, P. (2017). "Sensitivity analysis in
  observational research: introducing the E-value". *Annals of
  Internal Medicine* 167(4): 268-274.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

import numpy as np

# Type aliases for clarity.
DeltaName = str
AxisName = str
DivergenceVector = dict[AxisName, float]
ConfounderWeights = dict[tuple[Any, ...], float] | Literal["uniform"]


@dataclass(frozen=True)
class InterventionResult:
    """Result of running ONE intervention (replay with one delta toggled)."""

    delta: DeltaName
    """The delta that was injected from candidate config into baseline."""
    divergence: DivergenceVector
    """Per-axis divergence of the resulting replay vs the baseline run."""

    def to_dict(self) -> dict[str, Any]:
        return {"delta": self.delta, "divergence": dict(self.divergence)}


@dataclass(frozen=True)
class CausalAttribution:
    """Per-(delta, axis) ATE estimate, sorted by absolute effect.

    Always populated:
      - ``ate``: point ATE per (delta, axis).
      - ``interventions``: raw per-intervention divergence vectors.

    Populated when ``n_bootstrap > 0``:
      - ``ci_low``, ``ci_high``: bootstrap percentile CI bounds.

    Populated when ``sensitivity=True``:
      - ``e_values``: VanderWeele-Ding E-value per (delta, axis).
    """

    ate: dict[DeltaName, dict[AxisName, float]] = field(default_factory=dict)
    """Average treatment effect per delta, per axis."""
    interventions: list[InterventionResult] = field(default_factory=list)
    """Raw per-intervention divergence vectors (for diagnostics)."""
    ci_low: dict[DeltaName, dict[AxisName, float]] = field(default_factory=dict)
    """Lower bound of the bootstrap percentile CI."""
    ci_high: dict[DeltaName, dict[AxisName, float]] = field(default_factory=dict)
    """Upper bound of the bootstrap percentile CI."""
    e_values: dict[DeltaName, dict[AxisName, float]] = field(default_factory=dict)
    """VanderWeele-Ding E-value: smallest unmeasured-confounder effect
    that could explain away the observed ATE. ≥ 1; larger = more
    robust to unmeasured confounding."""

    def top(self, axis: AxisName, k: int = 5) -> list[tuple[DeltaName, float]]:
        """Return top-k deltas by |ATE| on the given axis."""
        scored = [(d, abs(self.ate.get(d, {}).get(axis, 0.0))) for d in self.ate]
        scored.sort(key=lambda x: -x[1])
        return [(d, self.ate.get(d, {}).get(axis, 0.0)) for d, _ in scored[:k]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ate": {d: dict(axes) for d, axes in self.ate.items()},
            "interventions": [iv.to_dict() for iv in self.interventions],
            "ci_low": {d: dict(axes) for d, axes in self.ci_low.items()},
            "ci_high": {d: dict(axes) for d, axes in self.ci_high.items()},
            "e_values": {d: dict(axes) for d, axes in self.e_values.items()},
        }


# A replay function takes a "config" (a dict of named deltas) and
# returns a per-axis divergence vector. Callers wire this to their
# backend — mock, Rust replay, live LLM.
ReplayFn = Callable[[dict[DeltaName, Any]], DivergenceVector]


def causal_attribution(
    *,
    baseline_config: dict[DeltaName, Any],
    candidate_config: dict[DeltaName, Any],
    replay_fn: ReplayFn,
    n_replays: int = 1,
    n_bootstrap: int = 0,
    confounders: list[DeltaName] | None = None,
    confounder_weights: ConfounderWeights | None = None,
    sensitivity: bool = False,
    confidence_level: float = 0.95,
    seed: int = 42,
    axes: list[AxisName] | None = None,
) -> CausalAttribution:
    """Estimate per-delta ATE via single-delta interventions, with
    optional bootstrap CIs, back-door adjustment, and E-value sensitivity.

    Parameters
    ----------
    baseline_config, candidate_config :
        The two configurations to compare. The set of differing keys
        defines the deltas under attribution.
    replay_fn :
        Callable mapping a config dict to a per-axis divergence vector.
    n_replays :
        Number of independent replays per cell (control and intervention).
        Increases noise resilience of the point estimate.
    n_bootstrap :
        Number of bootstrap resamples for CI computation. ``0`` (default)
        skips CI computation entirely. ``≥200`` recommended for stable
        95% percentile bounds.
    confounders :
        Optional list of config keys whose effect should be controlled
        for via Pearl's back-door criterion. The estimator stratifies
        over the 2^|C| combinations of confounder values (drawn from
        ``baseline_config`` and ``candidate_config``) and combines the
        per-stratum ATEs via Pearl's formula
        ``ATE = Σ_c P(C=c) · ATE_c``. See ``confounder_weights`` for
        how P(C=c) is supplied.
    confounder_weights :
        Per-stratum weights for the back-door combination. Required
        when ``confounders`` is non-empty. One of:

          * ``dict[tuple, float]`` — explicit weights keyed by the
            tuple of confounder values in the same order as
            ``confounders``. Every cartesian combination of
            (baseline_value, candidate_value) across confounders
            must have a weight. Raw counts are accepted; the
            estimator normalises to sum to 1.
          * ``"uniform"`` — sentinel that distributes weight
            equally across strata (1/n). Only correct when P(C=c)
            is itself uniform; pass it explicitly to acknowledge
            the assumption.

        Omitting this argument while declaring confounders raises
        ``ValueError``. The previous silent uniform default is
        removed.
    sensitivity :
        When True, compute the VanderWeele-Ding E-value per (delta, axis).
        Requires ``n_replays ≥ 2`` for a non-degenerate pooled SD; with
        ``n_replays = 1`` and a deterministic replay the E-value is
        reported as 1.0 (null) when ATE = 0 and ``+inf`` otherwise.
    confidence_level :
        Two-sided CI level. Default 0.95.
    seed :
        Bootstrap RNG seed.
    axes :
        Optional whitelist of axis names to report. Defaults to "all
        axes the replay function returned".

    Returns
    -------
    :class:`CausalAttribution`

    Raises
    ------
    ValueError
        For invalid ``n_replays``, ``n_bootstrap``, ``confidence_level``,
        identical configs, or confounders that are not config keys / are
        the target delta.
    """
    if n_replays < 1:
        raise ValueError(f"n_replays must be >= 1; got {n_replays}")
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be >= 0; got {n_bootstrap}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1); got {confidence_level}")

    differing = [k for k in candidate_config if baseline_config.get(k) != candidate_config[k]]
    if not differing:
        raise ValueError(
            "no differing keys between baseline_config and candidate_config — "
            "nothing to attribute. Did you pass two identical configs?"
        )

    confounder_keys = list(confounders) if confounders else []
    if confounder_keys:
        for c in confounder_keys:
            if c not in baseline_config or c not in candidate_config:
                raise ValueError(
                    f"confounder {c!r} not found in baseline_config and candidate_config"
                )
        # Disallow targeting a delta as its own confounder — the
        # back-door adjustment would integrate over the very value we
        # are trying to attribute.
        for c in confounder_keys:
            if c in differing and c not in [k for k in differing if k not in confounder_keys]:
                # We allow confounders that themselves differ between
                # configs (that's the typical case — e.g. `model`
                # changes alongside the prompt). What we forbid is
                # asking for a confounder when the **target** delta
                # has nothing to attribute beyond it. Detected here:
                pass
        # The forbidden case: confounders == differing (no remaining
        # target delta).
        target_deltas = [k for k in differing if k not in confounder_keys]
        if not target_deltas:
            raise ValueError(
                "all differing keys were declared as confounders — "
                "no target delta remains for attribution"
            )

    rng = np.random.default_rng(seed)

    interventions: list[InterventionResult] = []
    ate: dict[DeltaName, dict[AxisName, float]] = {}
    ci_low: dict[DeltaName, dict[AxisName, float]] = {}
    ci_high: dict[DeltaName, dict[AxisName, float]] = {}
    e_values: dict[DeltaName, dict[AxisName, float]] = {}

    target_deltas = [k for k in differing if k not in confounder_keys]

    normalised_weights = _validate_and_normalise_weights(
        confounder_keys, baseline_config, candidate_config, confounder_weights
    )

    for delta in target_deltas:
        per_axis_ate, per_axis_runs = _ate_for_delta(
            target_delta=delta,
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            replay_fn=replay_fn,
            n_replays=n_replays,
            confounder_keys=confounder_keys,
            confounder_weights=normalised_weights,
            axes=axes,
        )
        ate[delta] = per_axis_ate

        # Use the last (or only) intervention run for the diagnostic record.
        last_intervention_runs = per_axis_runs["__last_intervention__"]
        interventions.append(
            InterventionResult(delta=delta, divergence=_mean_divergence(last_intervention_runs))
        )

        if n_bootstrap > 0:
            ci_low[delta], ci_high[delta] = _bootstrap_ci_per_axis(
                per_axis_runs=per_axis_runs,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                rng=rng,
                axes_filter=axes,
                confounder_weights=normalised_weights,
            )

        if sensitivity:
            e_values[delta] = _e_values_per_axis(
                per_axis_runs=per_axis_runs,
                axes_filter=axes,
            )

    return CausalAttribution(
        ate=ate,
        interventions=interventions,
        ci_low=ci_low,
        ci_high=ci_high,
        e_values=e_values,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _ate_for_delta(
    *,
    target_delta: DeltaName,
    baseline_config: dict[DeltaName, Any],
    candidate_config: dict[DeltaName, Any],
    replay_fn: ReplayFn,
    n_replays: int,
    confounder_keys: list[DeltaName],
    confounder_weights: dict[tuple[Any, ...], float] | Literal["uniform"] | None,
    axes: list[AxisName] | None,
) -> tuple[
    dict[AxisName, float],
    dict[str, Any],
]:
    """Compute the ATE for a single target delta, optionally adjusting
    for confounders via back-door stratification.

    Returns
    -------
    (ate_per_axis, runs_dict) where ``runs_dict`` contains the raw
    per-stratum control / intervention replay outputs keyed for downstream
    bootstrap CI / E-value computations:
        - ``"control_runs__<stratum>"``  : list of control replays
        - ``"intervention_runs__<stratum>"`` : list of intervention replays
        - ``"__last_intervention__"`` : the last intervention run set
          (for the diagnostic ``InterventionResult``).
        - ``"__stratum_combos__"`` : per-stratum confounder-value tuple,
          indexed by the stratum integer; only set when confounders are
          declared. Lets the bootstrap CI honour the same per-stratum
          weights as the point estimate.
    """
    runs: dict[str, Any] = {}

    if not confounder_keys:
        # No back-door adjustment: a single stratum.
        control_cfg = dict(baseline_config)
        intervened_cfg = dict(baseline_config)
        intervened_cfg[target_delta] = candidate_config[target_delta]

        control_runs = [replay_fn(dict(control_cfg)) for _ in range(n_replays)]
        intervention_runs = [replay_fn(dict(intervened_cfg)) for _ in range(n_replays)]

        runs["control_runs__base"] = control_runs
        runs["intervention_runs__base"] = intervention_runs
        runs["__last_intervention__"] = intervention_runs

        ate = _ate_from_runs(control_runs, intervention_runs, axes)
        return ate, runs

    # Back-door adjustment: stratify over confounder combinations and
    # combine via Pearl's formula ATE = Σ_c P(C=c) · ATE_c. Per-stratum
    # weights come from `confounder_weights` (already validated and
    # normalised by the caller) or default to uniform when None.
    stratum_ates: list[tuple[tuple[Any, ...], dict[AxisName, float]]] = []

    confounder_value_grids = [(baseline_config[c], candidate_config[c]) for c in confounder_keys]
    last_intervention_runs: list[DivergenceVector] = []

    for stratum_idx, combo in enumerate(product(*confounder_value_grids)):
        fixed_cfg = dict(baseline_config)
        for c, v in zip(confounder_keys, combo, strict=True):
            fixed_cfg[c] = v

        control_cfg = dict(fixed_cfg)
        intervened_cfg = dict(fixed_cfg)
        intervened_cfg[target_delta] = candidate_config[target_delta]

        control_runs = [replay_fn(dict(control_cfg)) for _ in range(n_replays)]
        intervention_runs = [replay_fn(dict(intervened_cfg)) for _ in range(n_replays)]

        runs[f"control_runs__{stratum_idx}"] = control_runs
        runs[f"intervention_runs__{stratum_idx}"] = intervention_runs
        last_intervention_runs = intervention_runs

        stratum_ates.append((combo, _ate_from_runs(control_runs, intervention_runs, axes)))

    runs["__last_intervention__"] = last_intervention_runs
    # Stratum index → confounder combo, so the bootstrap CI can apply
    # the same weights as the point estimate. Stored as a sentinel
    # alongside the run lists rather than passed as a separate arg, so
    # callers downstream of `_ate_for_delta` see the mapping next to
    # the runs themselves.
    runs["__stratum_combos__"] = [combo for combo, _ in stratum_ates]

    all_axes: set[str] = set()
    for _, s in stratum_ates:
        all_axes.update(s.keys())
    if axes is not None:
        all_axes &= set(axes)

    if confounder_weights == "uniform":
        # Caller explicitly opted into 1/n weighting. Documented as
        # "unbiased only when P(C=c) is itself uniform" in the public
        # docstring; the sentinel forces the caller to acknowledge it.
        weights = {combo: 1.0 / len(stratum_ates) for combo, _ in stratum_ates}
    elif confounder_weights is None:
        # Should be unreachable: _validate_and_normalise_weights raises
        # before we get here when confounders are declared without
        # weights. Kept defensive in case the validation contract drifts.
        raise ValueError(
            "confounder_weights is required when confounders are declared; "
            "pass 'uniform' to acknowledge a uniform-P(C=c) assumption or "
            "provide a dict of per-stratum weights"
        )
    else:
        weights = confounder_weights

    ate = {ax: sum(weights[combo] * s.get(ax, 0.0) for combo, s in stratum_ates) for ax in all_axes}
    return ate, runs


def _validate_and_normalise_weights(
    confounder_keys: list[DeltaName],
    baseline_config: dict[DeltaName, Any],
    candidate_config: dict[DeltaName, Any],
    confounder_weights: dict[tuple[Any, ...], float] | Literal["uniform"] | None,
) -> dict[tuple[Any, ...], float] | Literal["uniform"] | None:
    """Validate user-supplied weights against the cartesian product of
    confounder values, normalise to sum to 1, and return the result.

    Contract:
      * ``None`` + no confounders → ``None`` passes through.
      * ``None`` + 1+ confounders → raises (the caller must opt into
        a P(C=c) assumption explicitly; previously this silently used
        uniform 1/n weights, which biased the back-door estimate
        toward the simple average for non-uniform stratum
        distributions).
      * ``"uniform"`` + no confounders → raises (no strata to weight).
      * ``"uniform"`` + 1+ confounders → ``"uniform"`` passes through;
        the back-door step interprets the sentinel as 1/n.
      * dict + no confounders → raises (no strata to weight).
      * dict + 1+ confounders → validated against the cartesian
        product of (baseline, candidate) values, then normalised.
    """
    if confounder_weights is None:
        if confounder_keys:
            raise ValueError(
                "confounder_weights is required when confounders are declared. "
                "Pass 'uniform' to acknowledge a uniform-P(C=c) assumption, "
                "or supply a dict of per-stratum weights derived from your "
                "target population. The previous silent uniform default was "
                "removed because it produced a biased estimate that read as "
                "'the Pearl ATE' to callers who never supplied weights."
            )
        return None

    if not confounder_keys:
        # No back-door adjustment requested but weights were supplied —
        # weights are meaningless without confounders. Treat as a usage
        # error so the caller doesn't get silently-ignored configuration.
        raise ValueError(
            "confounder_weights supplied but `confounders` is empty — "
            "weights have no strata to apply to"
        )

    if confounder_weights == "uniform":
        return "uniform"

    grids = [(baseline_config[c], candidate_config[c]) for c in confounder_keys]
    expected_combos = list(product(*grids))

    missing = [combo for combo in expected_combos if combo not in confounder_weights]
    if missing:
        raise ValueError(
            "confounder_weights is missing weight(s) for stratum/strata "
            f"{missing!r}; back-door requires every (baseline, candidate) "
            "combination across confounders to have an explicit weight"
        )

    total = sum(float(confounder_weights[c]) for c in expected_combos)
    if total <= 0.0:
        raise ValueError(
            "confounder_weights must sum to a positive value; "
            f"got {total} across {len(expected_combos)} strata"
        )
    return {c: float(confounder_weights[c]) / total for c in expected_combos}


def _ate_from_runs(
    control_runs: list[DivergenceVector],
    intervention_runs: list[DivergenceVector],
    axes: list[AxisName] | None,
) -> dict[AxisName, float]:
    """Per-axis ATE = mean(intervention) − mean(control)."""
    control_mean = _mean_divergence(control_runs)
    intervention_mean = _mean_divergence(intervention_runs)
    all_axes = set(control_mean.keys()) | set(intervention_mean.keys())
    if axes is not None:
        all_axes &= set(axes)
    return {ax: intervention_mean.get(ax, 0.0) - control_mean.get(ax, 0.0) for ax in all_axes}


def _bootstrap_ci_per_axis(
    *,
    per_axis_runs: dict[str, Any],
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
    axes_filter: list[AxisName] | None,
    confounder_weights: dict[tuple[Any, ...], float] | Literal["uniform"] | None,
) -> tuple[dict[AxisName, float], dict[AxisName, float]]:
    """Percentile-bootstrap CI on the ATE.

    Resamples the per-stratum control and intervention runs with
    replacement, computes the resampled ATE on each draw, and reports
    the (alpha/2, 1-alpha/2) percentiles across draws.

    For multi-stratum (back-door-adjusted) ATEs, the resampling is
    done within each stratum and the stratum ATEs are recombined
    using the same ``confounder_weights`` as the point estimate.
    Without the weights match, the CI bounds would be uniform-weighted
    while the point estimate honours the user's P(C=c) — they would
    not refer to the same statistic.
    """
    # Identify the strata. Sorted as integers so the order matches
    # the enumerate() order in `_ate_for_delta` (the strata indices
    # are written as integer-valued strings).
    stratum_keys = sorted(
        {k.removeprefix("control_runs__") for k in per_axis_runs if k.startswith("control_runs__")},
        key=lambda s: int(s) if s.isdigit() else 0,
    )

    # Discover all axes present.
    all_axes: set[str] = set()
    for k in per_axis_runs:
        if k.startswith("control_runs__") or k.startswith("intervention_runs__"):
            runs_list = per_axis_runs[k]
            if isinstance(runs_list, list):
                for run in runs_list:
                    if isinstance(run, dict):
                        all_axes.update(run.keys())
    if axes_filter is not None:
        all_axes &= set(axes_filter)

    # Resolve per-stratum weights so the CI uses the same combination
    # rule as the point estimate.
    stratum_combos = per_axis_runs.get("__stratum_combos__")
    weight_per_stratum: list[float]
    if confounder_weights == "uniform" or confounder_weights is None or stratum_combos is None:
        # Uniform fallback covers: caller opted into "uniform"; no
        # confounders (single stratum, weight = 1); or the combo
        # mapping is missing (legacy callers calling this internal
        # helper directly).
        n = len(stratum_keys) or 1
        weight_per_stratum = [1.0 / n] * len(stratum_keys)
    else:
        # Map stratum index → combo → weight. Missing combos fall to
        # 0; this matches the validation contract that
        # `_validate_and_normalise_weights` already enforces.
        weight_per_stratum = [
            float(confounder_weights.get(stratum_combos[int(s)], 0.0)) for s in stratum_keys
        ]
        total = sum(weight_per_stratum)
        if total > 0:
            weight_per_stratum = [w / total for w in weight_per_stratum]

    alpha = 1.0 - confidence_level
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    # Bootstrap loop: each replicate produces one ATE-per-axis vector.
    bootstrap_ates: dict[AxisName, list[float]] = {ax: [] for ax in all_axes}

    for _ in range(n_bootstrap):
        stratum_ates: list[dict[AxisName, float]] = []
        for stratum in stratum_keys:
            controls = per_axis_runs[f"control_runs__{stratum}"]
            interventions = per_axis_runs[f"intervention_runs__{stratum}"]
            n_c = len(controls)
            n_i = len(interventions)
            if n_c == 0 or n_i == 0:
                stratum_ates.append({ax: 0.0 for ax in all_axes})
                continue
            c_idx = rng.integers(0, n_c, size=n_c)
            i_idx = rng.integers(0, n_i, size=n_i)
            c_resample = [controls[int(j)] for j in c_idx]
            i_resample = [interventions[int(j)] for j in i_idx]
            stratum_ates.append(_ate_from_runs(c_resample, i_resample, axes_filter))

        # Combine across strata using the same weights as the point estimate.
        for ax in all_axes:
            ax_combined = sum(
                w * s.get(ax, 0.0) for w, s in zip(weight_per_stratum, stratum_ates, strict=True)
            )
            bootstrap_ates[ax].append(ax_combined)

    ci_low = {ax: float(np.percentile(vals, lo_pct)) for ax, vals in bootstrap_ates.items()}
    ci_high = {ax: float(np.percentile(vals, hi_pct)) for ax, vals in bootstrap_ates.items()}
    return ci_low, ci_high


def _e_values_per_axis(
    *,
    per_axis_runs: dict[str, list[DivergenceVector]],
    axes_filter: list[AxisName] | None,
) -> dict[AxisName, float]:
    """VanderWeele-Ding (2017) E-value for continuous outcomes.

    Procedure
    ---------
    1. Collect all per-stratum control and intervention runs.
    2. Compute the unstratified pooled ATE. Note: this E-value is a
       sensitivity check on the *unadjusted* effect, not on the
       back-door adjusted estimate. When ``confounder_weights`` is
       non-uniform the two ATEs differ; the E-value here gives a
       conservative robustness floor against unmeasured confounders
       beyond those modelled.
    3. Compute pooled SD over the union of control and intervention
       runs.
    4. Standardize: ``d = ATE / SD_pooled`` (Cohen's d for two
       independent groups, equal n).
    5. Convert to a risk-ratio-equivalent: ``RR ≈ exp(0.91 · |d|)``.
    6. E-value ``= RR + sqrt(RR · (RR − 1))``.

    Edge cases
    ----------
    - ``ATE = 0`` exactly → E-value = 1.0 (no confounding strength
      needed to explain a null effect).
    - ``SD_pooled = 0`` (deterministic outcomes) → E-value is +inf
      when ATE ≠ 0 (any nonzero confounder effect could explain it
      under zero-noise data).
    """
    # Aggregate runs across strata.
    all_controls: list[DivergenceVector] = []
    all_interventions: list[DivergenceVector] = []
    for k, runs in per_axis_runs.items():
        if k.startswith("control_runs__"):
            all_controls.extend(runs)
        elif k.startswith("intervention_runs__"):
            all_interventions.extend(runs)

    all_axes: set[str] = set()
    for r in all_controls + all_interventions:
        all_axes.update(r.keys())
    if axes_filter is not None:
        all_axes &= set(axes_filter)

    out: dict[AxisName, float] = {}
    for ax in all_axes:
        c_vals = [r.get(ax, 0.0) for r in all_controls]
        i_vals = [r.get(ax, 0.0) for r in all_interventions]
        ate = (sum(i_vals) / len(i_vals) if i_vals else 0.0) - (
            sum(c_vals) / len(c_vals) if c_vals else 0.0
        )
        if abs(ate) < 1e-12:
            out[ax] = 1.0
            continue
        pooled = c_vals + i_vals
        sd = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
        if sd < 1e-12:
            out[ax] = math.inf
            continue
        d = ate / sd
        rr = math.exp(0.91 * abs(d))
        out[ax] = float(rr + math.sqrt(rr * (rr - 1.0)))
    return out


def _mean_divergence(runs: list[DivergenceVector]) -> DivergenceVector:
    """Element-wise mean of a list of divergence vectors."""
    if not runs:
        return {}
    keys: set[str] = set()
    for run in runs:
        keys.update(run.keys())
    return {k: sum(run.get(k, 0.0) for run in runs) / len(runs) for k in keys}
