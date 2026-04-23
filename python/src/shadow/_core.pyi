"""Type stubs for the `shadow._core` Rust extension module.

The implementation lives in `crates/shadow-core/src/python.rs`; this file
exists so mypy --strict can see the surface.
"""

from __future__ import annotations

from typing import Any, TypedDict

__version__: str
SPEC_VERSION: str

class AxisStat(TypedDict):
    """One axis's statistical result. Shape of every element in
    the `rows` list returned by `compute_diff_report`.

    `axis` is one of: semantic, trajectory, safety, verbosity, latency,
    cost, reasoning, judge, conformance.

    `severity` is one of: none, minor, moderate, severe. See
    `Severity::classify` / `classify_rate` in `crates/shadow-core/src/diff/axes.rs`.

    `flags` may contain: `low_power` (n<5) or `ci_crosses_zero`
    (the bootstrap 95% CI strictly straddles zero — a boundary-touching
    CI like [0.0, 1.0] is NOT flagged).

    This TypedDict is documentation of the shape; the runtime value is
    a plain dict.
    """

    axis: str
    baseline_median: float
    candidate_median: float
    delta: float
    ci95_low: float
    ci95_high: float
    severity: str
    n: int
    flags: list[str]

class FirstDivergence(TypedDict):
    """First turn at which the candidate meaningfully diverged from
    the baseline.

    `kind` ∈ {`style_drift`, `decision_drift`, `structural_drift`}:
      - `style_drift`: cosmetic wording only (semantic similarity ≥ 0.9,
        same tool shape, same stop reason).
      - `decision_drift`: same structure, different decision (different
        arg values, refusal flipped, meaningful semantic shift).
      - `structural_drift`: tool sequence differs (insertion, deletion,
        reorder, or tool-name mismatch at the same position).

    `primary_axis` names which of the nine diff axes carries the
    strongest signal for this divergence (semantic / trajectory /
    safety / conformance).

    `confidence` is 0..1; callers can gate display on >= 0.5.
    """

    baseline_turn: int
    candidate_turn: int
    kind: str
    primary_axis: str
    explanation: str
    confidence: float

class DiffReport(TypedDict):
    """Shape of the dict returned by `compute_diff_report`.

    Documentation only — the runtime return value is a plain
    `dict[str, Any]`. Consumers may cast to this TypedDict for static
    access if they want.

    `first_divergence` is `None` when the two traces agree end-to-end,
    otherwise a `FirstDivergence` describing the first meaningful
    behavioural delta (in alignment order).

    `divergences` is the top-K ranked list of divergences, sorted by
    importance (Structural > Decision > Style, then by confidence).
    Empty when the two traces agree. Capped at `DEFAULT_K=5` entries.
    """

    rows: list[AxisStat]
    baseline_trace_id: str
    candidate_trace_id: str
    pair_count: int
    first_divergence: FirstDivergence | None
    divergences: list[FirstDivergence]

def parse_agentlog(data: bytes) -> list[dict[str, Any]]:
    """Parse a `.agentlog` byte blob into a list of record dicts.

    Each record has keys: `version`, `id`, `kind`, `ts`, `parent`, `meta`
    (optional), `payload`. See SPEC §3.
    """

def write_agentlog(records: list[dict[str, Any]]) -> bytes:
    """Serialize a list of record dicts into `.agentlog` bytes."""

def canonical_bytes(payload: dict[str, Any] | list[Any] | str | int | float | bool | None) -> bytes:
    """Canonical-JSON byte sequence for a payload (SPEC §5)."""

def content_id(payload: dict[str, Any] | list[Any] | str | int | float | bool | None) -> str:
    """Content id for a payload: `"sha256:" + hex(sha256(canonical_bytes))` (SPEC §6)."""

def compute_diff_report(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    pricing: dict[str, tuple[float, float]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute a nine-axis DiffReport.

    Parameters
    ----------
    baseline, candidate:
        Parsed `.agentlog` record lists (from [`parse_agentlog`]).
    pricing:
        Optional per-model `{"model-id": (input_$per_mtok, output_$per_mtok)}`
        used by the `cost` axis. When omitted, the cost axis reports 0.
    seed:
        Bootstrap RNG seed. Use a fixed value for reproducible CIs.

    Returns
    -------
    dict[str, Any]
        Shape is documented by the `DiffReport` TypedDict in this stub.
        Keys: `rows` (list of per-axis results, each shaped like
        `AxisStat`), `baseline_trace_id`, `candidate_trace_id`,
        `pair_count`.
    """
