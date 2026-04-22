"""Orchestrate the bisection: parse configs → design matrix → replays → LASSO.

In v0.1, the runner uses a *synthetic* per-corner divergence signal when
no live backend is available: for each corner configuration, the
divergence on each axis is the sum of the axis-weight of each delta
that's active in that corner. In practice production usage wires the
runner up to `shadow.replay.run_replay` with a live or mock backend;
that variant is deferred to v0.2 when live-API replay lands.

This v0.1 behavior is sufficient for the CLI's `shadow bisect` command
to run end-to-end and for the ground-truth synthetic test (where the
axis-weights are known a priori and the LASSO must recover them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shadow.bisect.attribution import AXIS_NAMES, rank_attributions
from shadow.bisect.deltas import diff_configs, load_config
from shadow.bisect.design import full_factorial, plackett_burman
from shadow.errors import ShadowConfigError


def run_bisect(
    config_a: Path | str,
    config_b: Path | str,
    traces: Path | str,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """High-level entry point for `shadow bisect`.

    For v0.1 this produces a "skeleton" attribution report: the deltas
    are real, the design matrix is real, but the per-corner divergence
    is a placeholder (all-zero). Downstream tooling can override
    `_score_corners` to plug in a real replay-based scorer.
    """
    a = load_config(config_a)
    b = load_config(config_b)
    deltas = diff_configs(a, b)
    if not deltas:
        raise ShadowConfigError(
            "configs are identical — no deltas to attribute.\n"
            "hint: bisect requires two configs that differ in at least one field."
        )
    labels = [d.path for d in deltas]
    design = choose_design(len(deltas))
    divergence = np.zeros((design.shape[0], len(AXIS_NAMES)), dtype=float)
    attributions = rank_attributions(design, divergence, labels, alpha=alpha)
    return {
        "deltas": [{"path": d.path, "old": d.old_value, "new": d.new_value} for d in deltas],
        "design_runs": int(design.shape[0]),
        "traces_path": str(traces),
        "attributions": {
            axis: [{"delta": lbl, "weight": w} for lbl, w in ranked]
            for axis, ranked in attributions.items()
        },
    }


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


def _resolve_traces_path(path: Path | str) -> list[Path]:
    """Expand a trace path or dir into a list of `.agentlog` files."""
    p = Path(path)
    if p.is_dir():
        return sorted(p.rglob("*.agentlog"))
    return [p]


__all__ = ["choose_design", "run_bisect", "score_corners_synthetic"]
