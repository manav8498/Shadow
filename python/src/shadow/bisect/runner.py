"""Orchestrate the bisection: parse configs → design matrix → per-corner
scoring → LASSO attribution.

v0.1 scoring strategy:

The ideal scorer replays each design-matrix corner against a live LLM
and measures the nine-axis divergence per corner. That requires API
keys and O(2^k) LLM calls, which is out of scope for a local-only v0.1.

Instead, this runner uses a **heuristic kind-based allocator** when the
user provides `candidate_traces`:

1. Compute the actual baseline-vs-candidate 9-axis divergence using the
   Rust `compute_diff_report`. This is the "all-deltas-on" corner's
   ground truth divergence.
2. For each nonzero axis, allocate its divergence across the deltas
   that could plausibly have caused it, based on a static mapping from
   delta-kind to affected-axis. `prompt.*` can affect semantic /
   verbosity / safety / conformance / reasoning. `tools[*].*` can
   affect trajectory. `model_id` can affect all. `params.temperature`
   / `params.top_p` affect semantic / verbosity.
3. Normalise per axis so attributions sum to 1.

This is honest about its limits — it's an allocator, not a causal
inference engine — but it produces non-zero, plausible, defensible
attributions instead of placeholder zeros, which is what the customer-
support scenario and every realistic use case needs.

When `candidate_traces` is omitted, we fall back to the previous v0.1
behaviour (zero divergence) and print a warning in the output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shadow import _core
from shadow.bisect.attribution import AXIS_NAMES, rank_attributions
from shadow.bisect.deltas import Delta, diff_configs, load_config
from shadow.bisect.design import full_factorial, plackett_burman
from shadow.errors import ShadowConfigError

# Which axes each delta-kind can plausibly affect, derived strictly from
# the axes' own definitions (not from any particular domain). The logic
# for each cell:
#
#   prompt.*    The system/user prompt is the text the model conditions on;
#               it can move semantic content, verbosity (via length
#               instructions), conformance (format directives), reasoning
#               (CoT prompting), latency/cost (downstream of verbosity),
#               and — because prompts often contain explicit refusal
#               instructions — safety. Prompts CAN instruct tool-calling
#               behaviour too, so trajectory is eligible.
#
#   params.*   Sampling hyperparameters change which token the model
#              picks (→ semantic drift), how long it stays on-topic
#              (→ verbosity, reasoning, latency, cost), and can break
#              structured outputs at high temperature (→ conformance).
#              They don't train a new safety policy, so safety is out.
#              Trajectory is out too: the tool-call *decision* is mostly
#              policy-driven, not sampling-driven, at the usual temps.
#
#   tools.*    Tool-schema edits directly drive tool-call shape
#              (→ trajectory). They also change the input-token count of
#              every turn (→ verbosity of INPUT tokens, latency, cost).
#              They do not retrain the model's refusal policy (safety
#              is out) and they don't alter generated-text semantics
#              except via trajectory-mediated downstream effects.
#
#   model_id   Model swaps can move any axis (different policy, different
#              length priors, different pricing, different refusal
#              thresholds). Eligible for all nine.
#
# This mapping is LLM-general — it encodes how each class of
# configuration knob influences each of the nine axes' definitions,
# with no assumptions about a particular domain (customer support,
# medical, legal, etc.). Users whose deltas land in a category not
# listed here (custom yaml keys) get Delta.kind = their top-level key
# and no allocation until they extend this mapping or use a live-replay
# scorer (v0.2).
DELTA_KIND_AFFECTS: dict[str, frozenset[str]] = {
    "prompt": frozenset(
        {
            "semantic",
            "trajectory",
            "safety",
            "verbosity",
            "reasoning",
            "conformance",
            "latency",
            "cost",
        }
    ),
    "params": frozenset({"semantic", "verbosity", "reasoning", "conformance", "latency", "cost"}),
    "tools": frozenset({"trajectory", "verbosity", "latency", "cost"}),
    "model": frozenset(AXIS_NAMES),
    "model_id": frozenset(AXIS_NAMES),
}


def run_bisect(
    config_a: Path | str,
    config_b: Path | str,
    traces: Path | str,
    candidate_traces: Path | str | None = None,
    backend: Any | None = None,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """High-level entry point for `shadow bisect`.

    Three modes, chosen in priority order:

    1. `lasso_over_corners` (best, real causal attribution) — selected
       when a live `backend` is supplied. Runs the full LASSO-over-corners
       scorer: for each of the 2**k corners of the category-level design
       matrix, replays the baseline against the intermediate config and
       measures per-axis divergence, then fits LASSO per axis. See
       `shadow.bisect.corner_scorer`.
    2. `heuristic_kind_allocator` — selected when `candidate_traces` is
       supplied (and no `backend`). Computes the real
       baseline-vs-candidate divergence and allocates it heuristically
       across the delta categories that could plausibly have caused
       movement on each axis. Fast, offline, no LLM calls.
    3. `lasso_placeholder_zero` — ultimate fallback when neither
       `backend` nor `candidate_traces` is available. Returns the right
       shape with zero weights and a warning.

    Parameters
    ----------
    config_a, config_b:
        YAML configs to diff.
    traces:
        Path to the baseline `.agentlog` file.
    candidate_traces:
        Optional path to a candidate `.agentlog` file recorded under
        `config_b`. Used when `backend` is not supplied.
    backend:
        Optional `shadow.llm.LlmBackend`. When supplied, triggers
        live-replay corner scoring (mode 1 above).
    alpha:
        LASSO regularisation strength for modes 1 and 3.
    """
    from shadow.bisect.apply import active_categories
    from shadow.bisect.corner_scorer import run_sync as corner_run

    a = load_config(config_a)
    b = load_config(config_b)
    deltas = diff_configs(a, b)
    if not deltas:
        raise ShadowConfigError(
            "configs are identical — no deltas to attribute.\n"
            "hint: bisect requires two configs that differ in at least one field."
        )
    labels = [d.path for d in deltas]

    warnings: list[str] = []

    if backend is not None:
        baseline_records = _core.parse_agentlog(Path(traces).read_bytes())
        result = corner_run(baseline_records, a, b, backend, alpha=alpha)
        return {
            "deltas": [{"path": d.path, "old": d.old_value, "new": d.new_value} for d in deltas],
            "mode": "lasso_over_corners",
            "design_runs": int(result["design"].shape[0]),
            "traces_path": str(traces),
            "backend_id": backend.id,
            "active_categories": result["categories"],
            "warnings": warnings,
            "attributions": result["attributions"],
        }

    if candidate_traces is not None:
        axis_divergence, note = _observed_divergence(traces, candidate_traces)
        if note:
            warnings.append(note)
        attributions = _allocate_divergence(deltas, axis_divergence)
        return {
            "deltas": [{"path": d.path, "old": d.old_value, "new": d.new_value} for d in deltas],
            "mode": "heuristic_kind_allocator",
            "design_runs": 1,
            "traces_path": str(traces),
            "candidate_traces_path": str(candidate_traces),
            "active_categories": active_categories(a, b),
            "warnings": warnings,
            "attributions": {
                axis: [{"delta": lbl, "weight": float(w)} for lbl, w in ranked]
                for axis, ranked in attributions.items()
            },
        }

    # Placeholder-zero fallback. Both --backend and --candidate-traces
    # were omitted.
    design = choose_design(len(deltas))
    divergence = np.zeros((design.shape[0], len(AXIS_NAMES)), dtype=float)
    attributions = rank_attributions(design, divergence, labels, alpha=alpha)
    warnings.append(
        "no --backend or --candidate-traces supplied; attributions are zero. "
        "Supply one of them to get real attributions."
    )
    return {
        "deltas": [{"path": d.path, "old": d.old_value, "new": d.new_value} for d in deltas],
        "mode": "lasso_placeholder_zero",
        "design_runs": int(design.shape[0]),
        "traces_path": str(traces),
        "warnings": warnings,
        "attributions": {
            axis: [{"delta": lbl, "weight": float(w)} for lbl, w in ranked]
            for axis, ranked in attributions.items()
        },
    }


def _observed_divergence(
    baseline_path: Path | str, candidate_path: Path | str
) -> tuple[dict[str, float], str | None]:
    """Compute the 9-axis divergence between baseline and candidate traces.

    Returns (axis→|delta|, optional-warning). Uses the Rust-side
    `compute_diff_report` (SPEC-compliant bootstrap CI etc.); the
    returned divergence here is just `abs(delta)` per axis because we
    want positive weights for the allocator.
    """
    baseline = _core.parse_agentlog(Path(baseline_path).read_bytes())
    candidate = _core.parse_agentlog(Path(candidate_path).read_bytes())
    report = _core.compute_diff_report(baseline, candidate, None, 42)
    divergence: dict[str, float] = {axis: 0.0 for axis in AXIS_NAMES}
    n_per_axis: dict[str, int] = {}
    for row in report["rows"]:
        divergence[row["axis"]] = abs(float(row["delta"]))
        n_per_axis[row["axis"]] = int(row["n"])
    warning = None
    if all(n == 0 for n in n_per_axis.values()):
        warning = "baseline and candidate traces had no comparable pairs"
    return divergence, warning


def _allocate_divergence(
    deltas: list[Delta], axis_divergence: dict[str, float]
) -> dict[str, list[tuple[str, float]]]:
    """Allocate each axis's observed divergence across the deltas that
    could plausibly have caused it, based on [`DELTA_KIND_AFFECTS`].

    Within each axis, eligible deltas share the divergence equally.
    Output is normalised so each axis's attributions sum to 1 (or 0
    when no delta is eligible for that axis).
    """
    result: dict[str, list[tuple[str, float]]] = {}
    for axis in AXIS_NAMES:
        axis_mass = axis_divergence.get(axis, 0.0)
        eligible = [d for d in deltas if axis in DELTA_KIND_AFFECTS.get(d.kind, frozenset())]
        if axis_mass < 1e-9 or not eligible:
            # No movement on this axis, or no delta could have caused it.
            result[axis] = [(d.path, 0.0) for d in deltas]
            continue
        per_eligible = 1.0 / len(eligible)
        weights: dict[str, float] = {d.path: 0.0 for d in deltas}
        for d in eligible:
            weights[d.path] = per_eligible
        ranked = sorted(weights.items(), key=lambda p: -p[1])
        result[axis] = ranked
    return result


def choose_design(k: int) -> NDArray[np.int8]:
    """Pick the appropriate design matrix for `k` deltas."""
    if k <= 6:
        return full_factorial(k)
    return plackett_burman(k)


def score_corners_synthetic(
    design: NDArray[np.int8],
    axis_weights: dict[int, dict[int, float]],
) -> NDArray[np.float64]:
    """Build a synthetic divergence matrix for unit tests.

    `axis_weights[delta_idx][axis_idx]` is the per-axis effect of delta
    `delta_idx` when it is active (+1). A corner's divergence on axis
    `a` is the sum over deltas `d` of `axis_weights[d][a]` where the
    corner's design row has a `+1` at column `d`.
    """
    runs, k = design.shape
    out = np.zeros((runs, len(AXIS_NAMES)), dtype=float)
    for run_idx in range(runs):
        for delta_idx in range(k):
            if design[run_idx, delta_idx] == 1:
                for axis_idx in range(len(AXIS_NAMES)):
                    out[run_idx, axis_idx] += axis_weights.get(delta_idx, {}).get(axis_idx, 0.0)
    return out


__all__ = [
    "DELTA_KIND_AFFECTS",
    "choose_design",
    "run_bisect",
    "score_corners_synthetic",
]
