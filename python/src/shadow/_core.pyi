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
    `DiffReport.rows`.

    `axis` is one of: semantic, trajectory, safety, verbosity, latency,
    cost, reasoning, judge, conformance (CLAUDE.md §4).

    `severity` is one of: none, minor, moderate, severe. See
    `Severity::classify` / `classify_rate` in `crates/shadow-core/src/diff/axes.rs`.

    `flags` may contain: `low_power` (n<5) or `ci_crosses_zero`
    (the bootstrap 95% CI strictly straddles zero, i.e. lower<-epsilon and
    upper>+ε — a boundary-touching CI like [0.0, 1.0] is NOT flagged).
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


class DiffReport(TypedDict):
    """Return shape of [`compute_diff_report`]."""

    rows: list[AxisStat]
    baseline_trace_id: str
    candidate_trace_id: str
    pair_count: int


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
) -> DiffReport:
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
    DiffReport
        With `rows` (one per axis — see CLAUDE.md §4), `pair_count`
        (the number of baseline/candidate turn pairs that aligned), and
        content-addressed `baseline_trace_id` / `candidate_trace_id`.
    """
