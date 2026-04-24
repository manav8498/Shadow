"""Hierarchical diff: session-level + span-level breakdowns.

Shadow's v0.x reports sit at two levels:

- **trace** — the full nine-axis table, one-per-report.
- **turn** — per-pair drill-down (`drill_down` field).

Two real-world questions those two levels can't answer:

1. "Which *session* did the regression happen in?" — when a trace
   contains multiple user-facing conversations.
2. "Within a regressed turn, which *content block* (text message,
   tool call, tool result) actually drove the divergence?"

This module adds the session-level and span-level layers.

## Session-level diff

`diff_by_session(baseline, candidate, ...)` partitions both traces
by `metadata` record (same partitioning as `cost_attribution`) and
runs the Rust differ on each session pair. Result: one
`DiffReport` per session. Useful when a baseline trace contains
10 conversations and only 1 regressed — the flat trace-level diff
would dilute the signal.

## Span-level diff

`span_diff(baseline_response, candidate_response)` walks the
`content` list of two paired chat_response payloads and classifies
each span:

- `text_block_changed` — text content drifted (reports a similarity
  score and a shortened diff preview).
- `tool_use_added` / `tool_use_removed` — one side invoked a tool
  the other didn't.
- `tool_use_args_changed` — same tool, different argument keys or
  values.
- `tool_result_changed` — tool_result payload differs (is_error
  flip, or content differ).
- `stop_reason_changed` — stop_reason flipped.

The classifier uses greedy index alignment: block #0 of baseline
pairs with block #0 of candidate, #1 with #1, etc. For lists of
different length, extras on either side are flagged as
added/removed. Not Needleman-Wunsch — per-turn span counts are
small (median < 5) so the simple alignment is cheaper and doesn't
introduce alignment-algorithm artefacts into a user-facing report.

Sits alongside the existing drill-down (`drill_down` field) as
the "dig one level deeper" layer of the hierarchy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from shadow import _core
from shadow.cost_attribution import partition_sessions

# ---- session-level diff --------------------------------------------------


@dataclass
class SessionDiff:
    """One session's diff report + identifying context."""

    session_index: int
    baseline_session_id: str  # the metadata record's id, for click-back
    candidate_session_id: str
    pair_count: int
    worst_severity: str  # "none" / "minor" / "moderate" / "severe"
    report: dict[str, Any]  # full DiffReport for this session pair

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diff_by_session(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    pricing: dict[str, Any] | None = None,
    seed: int | None = None,
) -> list[SessionDiff]:
    """Partition both traces by session, run `compute_diff_report` on each.

    Session-N of baseline aligns with session-N of candidate. If the
    two traces contain different session counts, the shorter is
    padded with empty sessions — the resulting `DiffReport` will
    have `pair_count == 0` and all-zero rows.
    """
    base_sessions = partition_sessions(baseline_records)
    cand_sessions = partition_sessions(candidate_records)

    out: list[SessionDiff] = []
    n = max(len(base_sessions), len(cand_sessions))
    for i in range(n):
        base_session = base_sessions[i] if i < len(base_sessions) else []
        cand_session = cand_sessions[i] if i < len(cand_sessions) else []
        report = _core.compute_diff_report(base_session, cand_session, pricing or None, seed)
        worst = _worst_severity(report)
        base_id = base_session[0]["id"] if base_session else ""
        cand_id = cand_session[0]["id"] if cand_session else ""
        out.append(
            SessionDiff(
                session_index=i,
                baseline_session_id=base_id,
                candidate_session_id=cand_id,
                pair_count=int(report.get("pair_count", 0)),
                worst_severity=worst,
                report=report,
            )
        )
    return out


def _worst_severity(report: dict[str, Any]) -> str:
    rank = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
    worst = "none"
    for row in report.get("rows", []):
        sev = row.get("severity", "none")
        if rank.get(sev, 0) > rank.get(worst, 0):
            worst = sev
    return worst


# ---- span-level diff -----------------------------------------------------


@dataclass
class SpanDiff:
    """One classified content-block-level change within a turn."""

    kind: str  # see module docstring for enum
    block_index: int  # index into the response's content list
    baseline: dict[str, Any] | None
    candidate: dict[str, Any] | None
    # Free-form human-readable summary, e.g.
    #   "tool_use `search_orders` args changed: {customer_id -> cid}"
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def span_diff(
    baseline_response: dict[str, Any],
    candidate_response: dict[str, Any],
) -> list[SpanDiff]:
    """Classify every content-block difference between one pair of
    chat_response payloads. Empty list = no span-level changes."""
    base_content = baseline_response.get("content") or []
    cand_content = candidate_response.get("content") or []
    out: list[SpanDiff] = []

    # First pass: index-aligned comparison of overlapping blocks.
    overlap = min(len(base_content), len(cand_content))
    for i in range(overlap):
        b = base_content[i] if isinstance(base_content[i], dict) else {}
        c = cand_content[i] if isinstance(cand_content[i], dict) else {}
        out.extend(_classify_block_pair(i, b, c))

    # Trailing blocks are adds/removes.
    for i in range(overlap, len(base_content)):
        b = base_content[i] if isinstance(base_content[i], dict) else {}
        out.append(
            SpanDiff(
                kind=_addremove_kind(b.get("type"), removed=True),
                block_index=i,
                baseline=b,
                candidate=None,
                summary=_describe_addremove(b, removed=True),
            )
        )
    for i in range(overlap, len(cand_content)):
        c = cand_content[i] if isinstance(cand_content[i], dict) else {}
        out.append(
            SpanDiff(
                kind=_addremove_kind(c.get("type"), removed=False),
                block_index=i,
                baseline=None,
                candidate=c,
                summary=_describe_addremove(c, removed=False),
            )
        )

    # Stop-reason flip is reported as a pseudo-span (block_index=-1)
    # so renderers can treat it consistently.
    base_sr = baseline_response.get("stop_reason")
    cand_sr = candidate_response.get("stop_reason")
    if base_sr != cand_sr:
        out.append(
            SpanDiff(
                kind="stop_reason_changed",
                block_index=-1,
                baseline={"stop_reason": base_sr},
                candidate={"stop_reason": cand_sr},
                summary=f"stop_reason: {base_sr!r} -> {cand_sr!r}",
            )
        )

    return out


def _classify_block_pair(index: int, b: dict[str, Any], c: dict[str, Any]) -> list[SpanDiff]:
    """Classify one pair of same-index content blocks."""
    b_type = b.get("type")
    c_type = c.get("type")
    if b_type != c_type:
        # Type swap is effectively a remove + add at the same index.
        return [
            SpanDiff(
                kind="block_type_changed",
                block_index=index,
                baseline=b,
                candidate=c,
                summary=f"block type: {b_type!r} → {c_type!r}",
            )
        ]
    if b_type == "text":
        b_text = b.get("text") or ""
        c_text = c.get("text") or ""
        if b_text == c_text:
            return []
        sim = _char_similarity(b_text, c_text)
        preview_b = b_text[:60] + ("…" if len(b_text) > 60 else "")
        preview_c = c_text[:60] + ("…" if len(c_text) > 60 else "")
        return [
            SpanDiff(
                kind="text_block_changed",
                block_index=index,
                baseline=b,
                candidate=c,
                summary=(
                    f"text block: similarity {sim:.2f}  "
                    f"baseline={preview_b!r}  candidate={preview_c!r}"
                ),
            )
        ]
    if b_type == "tool_use":
        changes = _tool_use_changes(b, c)
        if not changes:
            return []
        return [
            SpanDiff(
                kind="tool_use_args_changed",
                block_index=index,
                baseline=b,
                candidate=c,
                summary=f"tool_use `{b.get('name')}`: {', '.join(changes)}",
            )
        ]
    if b_type == "tool_result":
        if b.get("content") == c.get("content") and b.get("is_error", False) == c.get(
            "is_error", False
        ):
            return []
        return [
            SpanDiff(
                kind="tool_result_changed",
                block_index=index,
                baseline=b,
                candidate=c,
                summary=(
                    f"tool_result for {b.get('tool_use_id', '?')}: "
                    f"is_error {b.get('is_error', False)} → {c.get('is_error', False)}"
                ),
            )
        ]
    # Unknown block type — fall through if anything differs.
    if b != c:
        return [
            SpanDiff(
                kind="unknown_block_changed",
                block_index=index,
                baseline=b,
                candidate=c,
                summary=f"content block of type {b_type!r} differs",
            )
        ]
    return []


def _tool_use_changes(b: dict[str, Any], c: dict[str, Any]) -> list[str]:
    """Describe how two same-name tool_use blocks differ."""
    changes: list[str] = []
    if b.get("name") != c.get("name"):
        changes.append(f"name {b.get('name')!r} → {c.get('name')!r}")
    b_input = b.get("input") or {}
    c_input = c.get("input") or {}
    if not isinstance(b_input, dict) or not isinstance(c_input, dict):
        if b_input != c_input:
            changes.append("input (non-dict) differs")
        return changes
    b_keys = set(b_input.keys())
    c_keys = set(c_input.keys())
    added = c_keys - b_keys
    removed = b_keys - c_keys
    for k in sorted(removed):
        changes.append(f"arg removed `{k}`")
    for k in sorted(added):
        changes.append(f"arg added `{k}`")
    for k in sorted(b_keys & c_keys):
        if b_input[k] != c_input[k]:
            changes.append(f"arg `{k}` value changed")
    return changes


def _addremove_kind(block_type: Any, *, removed: bool) -> str:
    direction = "removed" if removed else "added"
    if block_type == "tool_use":
        return f"tool_use_{direction}"
    if block_type == "tool_result":
        return f"tool_result_{direction}"
    if block_type == "text":
        return f"text_block_{direction}"
    return f"block_{direction}"


def _describe_addremove(block: dict[str, Any], *, removed: bool) -> str:
    direction = "removed" if removed else "added"
    t = block.get("type")
    if t == "tool_use":
        name = block.get("name", "?")
        return f"tool_use `{name}` {direction}"
    if t == "tool_result":
        tid = block.get("tool_use_id", "?")
        return f"tool_result for {tid} {direction}"
    if t == "text":
        preview = (block.get("text") or "")[:60]
        return f"text block {direction}: {preview!r}"
    return f"{t} block {direction}"


def _char_similarity(a: str, b: str) -> float:
    """Character-set Jaccard — cheap similarity for short text."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 1.0


# ---- terminal rendering --------------------------------------------------


def render_session_summary(session_diffs: list[SessionDiff]) -> str:
    """One-line-per-session rollup for the terminal."""
    if not session_diffs:
        return ""
    lines = [f"Hierarchical diff — {len(session_diffs)} session(s):"]
    for sd in session_diffs:
        marker = {
            "none": "✓",
            "minor": "·",
            "moderate": "!",
            "severe": "✗",
        }.get(sd.worst_severity, "·")
        lines.append(
            f"  {marker}  session #{sd.session_index}: "
            f"{sd.pair_count} pair(s), worst severity {sd.worst_severity}"
        )
    return "\n".join(lines)


def render_spans(spans: list[SpanDiff]) -> str:
    """One-line-per-span rollup (pair-scoped, used inside drill-down)."""
    if not spans:
        return ""
    lines = [f"span-level changes ({len(spans)}):"]
    for s in spans:
        lines.append(f"  · {s.summary}")
    return "\n".join(lines)


__all__ = [
    "SessionDiff",
    "SpanDiff",
    "diff_by_session",
    "render_session_summary",
    "render_spans",
    "span_diff",
]
