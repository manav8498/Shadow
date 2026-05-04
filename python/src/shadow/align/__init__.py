"""Shadow's alignment engine — a reusable trace-comparison primitive.

`shadow.align` is the public-facing import for tools that want to use
Shadow's trace alignment + divergence detection without pulling in the
rest of the diagnose-pr / 9-axis / policy machinery.

Five public functions match the design spec §6 surface:

  * `align_traces(baseline, candidate)` — pair every baseline turn to
    its best-match candidate turn; returns an `Alignment` dataclass.
  * `first_divergence(baseline, candidate)` — find the FIRST point at
    which the two traces meaningfully differ; returns a `Divergence`
    or `None`.
  * `top_k_divergences(baseline, candidate, k=5)` — top-K ranked by
    severity * downstream impact.
  * `trajectory_distance(baseline_tools, candidate_tools)` —
    Levenshtein distance on a flat list of tool-call names; useful
    when you only have the tool-trajectory and not full traces.
  * `tool_arg_delta(a, b)` — structural diff of two JSON values
    (typically tool-call argument dicts); returns a list of typed
    deltas (added / removed / changed) keyed by JSON pointer.

The first three delegate to `shadow._core.compute_diff_report`'s
internal alignment (Rust); the last two are pure-Python so callers
without the Rust extension can still use the core comparators.

Stability: this surface is `v0.1` and intentionally narrow. Future
versions add fields without breaking existing field names.
"""

from __future__ import annotations

from shadow.align._core import (
    AlignedTurn,
    Alignment,
    ArgDelta,
    Divergence,
    align_traces,
    first_divergence,
    tool_arg_delta,
    top_k_divergences,
    trajectory_distance,
)

__all__ = [
    "AlignedTurn",
    "Alignment",
    "ArgDelta",
    "Divergence",
    "align_traces",
    "first_divergence",
    "tool_arg_delta",
    "top_k_divergences",
    "trajectory_distance",
]
