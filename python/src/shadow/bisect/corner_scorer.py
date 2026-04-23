"""LASSO-over-corners bisection scorer.

Given a baseline trace set, two configs, and a live `LlmBackend`,
computes a real causal attribution rather than the heuristic
delta-kind allocator used when no backend is supplied.

Algorithm:

1. Identify which of the four category-level knobs differ between
   config_a and config_b: `{model, prompt, params, tools}`. Call the
   active subset `k` (1 ≤ k ≤ 4).
2. Build a 2**k full-factorial design matrix in {-1, +1}.
3. For each corner (row of the design matrix):
   - Synthesize an intermediate config: for each active category, use
     config_b's value when the column is +1, config_a's otherwise.
   - Replay every baseline chat_request through the backend with the
     intermediate config applied. Produce a candidate trace.
   - Compute the nine-axis divergence between the baseline and this
     corner's candidate trace.
4. Fit LASSO per axis: X = design (runs x k), y = divergence column.
   Normalise coefficients per axis → attribution weights.

Output: one row per axis, weights summing to 1 (per axis) across the
active categories. Zero-variance axes (no movement across corners)
report all-zero weights.

This is the real feature the README pitches. It requires a backend;
the runner falls back to the heuristic allocator when no backend is
supplied (useful when you only have the two recorded traces, no
credentials to replay).
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shadow import _core
from shadow.bisect.apply import (
    CONFIG_CATEGORIES,
    active_categories,
    apply_config_to_request,
    build_intermediate_config,
)
from shadow.bisect.attribution import AXIS_NAMES, rank_attributions_with_ci
from shadow.bisect.design import full_factorial
from shadow.errors import ShadowBackendError, ShadowConfigError
from shadow.llm.base import LlmBackend


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


async def replay_with_config(
    baseline_records: list[dict[str, Any]],
    intermediate_config: dict[str, Any],
    backend: LlmBackend,
) -> list[dict[str, Any]]:
    """Replay a baseline trace through `backend` using `intermediate_config`.

    For each baseline `chat_request`, produces a fresh request whose
    config-derived fields (model, system prompt, params, tools) come
    from `intermediate_config`; other fields (user messages, assistant
    history, tool definitions the user didn't touch) are preserved.
    """
    if not baseline_records:
        raise ShadowConfigError("baseline is empty — need at least a metadata record")

    out: list[dict[str, Any]] = []

    # New metadata root pointing back at the baseline root.
    root = baseline_records[0]
    if root.get("kind") != "metadata":
        raise ShadowConfigError(f"baseline root is {root.get('kind')!r}, expected 'metadata'")
    new_meta_payload = dict(root["payload"])
    new_meta_payload["baseline_of"] = root["id"]
    new_meta_id = _core.content_id(new_meta_payload)
    out.append(
        {
            "version": "0.1",
            "id": new_meta_id,
            "kind": "metadata",
            "ts": _now_iso(),
            "parent": None,
            "payload": new_meta_payload,
        }
    )
    last_parent: str = new_meta_id

    for record in baseline_records:
        if record.get("kind") != "chat_request":
            continue
        new_payload = apply_config_to_request(intermediate_config, record["payload"])
        req_id = _core.content_id(new_payload)
        req_record = {
            "version": "0.1",
            "id": req_id,
            "kind": "chat_request",
            "ts": _now_iso(),
            "parent": last_parent,
            "payload": new_payload,
        }
        out.append(req_record)
        try:
            response_payload = await backend.complete(new_payload)
            resp_record = {
                "version": "0.1",
                "id": _core.content_id(response_payload),
                "kind": "chat_response",
                "ts": _now_iso(),
                "parent": req_id,
                "payload": response_payload,
            }
            out.append(resp_record)
            last_parent = str(resp_record["id"])
        except ShadowBackendError as e:
            err_payload = {
                "source": "llm",
                "code": "backend_error",
                "message": str(e),
                "retriable": False,
            }
            err_record = {
                "version": "0.1",
                "id": _core.content_id(err_payload),
                "kind": "error",
                "ts": _now_iso(),
                "parent": req_id,
                "payload": err_payload,
            }
            out.append(err_record)
            last_parent = str(err_record["id"])

    return out


def _divergence_vector(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    seed: int,
) -> NDArray[np.float64]:
    """Return a `(9,)` vector of `abs(delta)` per axis."""
    report = _core.compute_diff_report(baseline_records, candidate_records, None, seed)
    vec = np.zeros(len(AXIS_NAMES), dtype=float)
    rows_by_axis = {row["axis"]: row for row in report["rows"]}
    for i, axis in enumerate(AXIS_NAMES):
        # Defense-in-depth: the Rust differ always returns 9 rows, but a
        # malformed or future schema change shouldn't crash the scorer.
        row = rows_by_axis.get(axis)
        if row is None:
            continue
        delta = row.get("delta", 0.0)
        try:
            vec[i] = abs(float(delta))
        except (TypeError, ValueError):
            vec[i] = 0.0
    return vec


async def score_corners(
    baseline_records: list[dict[str, Any]],
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    backend: LlmBackend,
    seed: int = 42,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Run the LASSO-over-corners scorer and return an attribution report.

    Returns a dict with keys:
        `categories`        List of active category names (columns of the design matrix).
        `design`            (runs x k) ndarray in {-1, +1}.
        `divergence`        (runs x 9) ndarray, abs(delta) per axis per corner.
        `attributions`      dict[axis_name → list[{"category": str, "weight": float}]]
    """
    categories = active_categories(config_a, config_b)
    if not categories:
        raise ShadowConfigError("configs are identical — no category-level deltas to attribute")
    k = len(categories)
    design = full_factorial(k)  # (2**k, k) in {-1, +1}
    runs = design.shape[0]

    divergence = np.zeros((runs, len(AXIS_NAMES)), dtype=float)
    for run_idx in range(runs):
        mask = {cat: design[run_idx, i] == 1 for i, cat in enumerate(categories)}
        intermediate = build_intermediate_config(config_a, config_b, mask)
        candidate = await replay_with_config(baseline_records, intermediate, backend)
        divergence[run_idx] = _divergence_vector(baseline_records, candidate, seed)

    attributions_ci = rank_attributions_with_ci(
        design, divergence, categories, alpha=alpha, seed=seed
    )
    # Unified schema across all bisect modes (heuristic / placeholder / live):
    # every attribution row has the same keys. The `delta` key names the
    # attributed field (may be a delta.path or a coalesced category label —
    # same conceptual thing from the caller's POV). `category` is kept as
    # an alias for backward compatibility with the v0.1 live-mode API but
    # marked deprecated in the docstring.
    unified_rows = {
        axis: [
            {
                "delta": row["delta"],
                "category": row["delta"],  # deprecated alias
                "weight": float(row["weight"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "significant": bool(row["significant"]),
                "selection_frequency": float(row.get("selection_frequency", 0.0)),
            }
            for row in ranked
        ]
        for axis, ranked in attributions_ci.items()
    }
    return {
        "categories": categories,
        "design": design,
        "divergence": divergence,
        "attributions": unified_rows,
        # Legacy key — kept for v0.1 callers that used it. Same content.
        "attributions_ci": unified_rows,
    }


def run_sync(
    baseline_records: list[dict[str, Any]],
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    backend: LlmBackend,
    seed: int = 42,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Sync wrapper for callers that don't want to deal with an event loop."""
    return asyncio.run(score_corners(baseline_records, config_a, config_b, backend, seed, alpha))


__all__ = [
    "CONFIG_CATEGORIES",
    "replay_with_config",
    "run_sync",
    "score_corners",
]
