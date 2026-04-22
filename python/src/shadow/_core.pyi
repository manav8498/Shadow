"""Type stubs for the `shadow._core` Rust extension module.

The implementation lives in `crates/shadow-core/src/python.rs`; this file
exists so mypy --strict can see the surface.
"""

from __future__ import annotations

from typing import Any

__version__: str
SPEC_VERSION: str

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

    Returns a dict with keys: `rows` (list of AxisStat dicts),
    `baseline_trace_id`, `candidate_trace_id`, `pair_count`. See CLAUDE.md §4
    for the axis list.
    """
