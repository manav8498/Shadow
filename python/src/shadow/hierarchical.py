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


# Alignment costs for span-level Needleman-Wunsch. Tuned so a "gap"
# (pure add/remove) and a "type mismatch" have roughly comparable
# cost, nudging the aligner toward reporting an add+remove when
# block types differ (more actionable than a block_type_changed).
_SPAN_GAP_COST = 1.0
_SPAN_TYPE_MISMATCH_COST = 1.5
_SPAN_SAME_TYPE_DIFF_COST = 0.5
_SPAN_IDENTICAL_COST = 0.0


def span_diff(
    baseline_response: dict[str, Any],
    candidate_response: dict[str, Any],
) -> list[SpanDiff]:
    """Classify every content-block difference between one pair of
    chat_response payloads. Empty list = no span-level changes.

    Dispatch by input size:
      - **≤ 5 blocks either side** — greedy index alignment. Optimal
        for short lists and avoids the DP allocation.
      - **> 5 blocks** — Needleman-Wunsch alignment over the block
        lists with a cost model that prefers `add + remove` over
        `block_type_changed` when types mismatch. Catches the
        real-world case where an agent interleaves 20+ tool calls
        per turn and the candidate adds one in the middle —
        greedy would report every downstream block as a
        `block_type_changed`.
    """
    base_content = list(baseline_response.get("content") or [])
    cand_content = list(candidate_response.get("content") or [])
    out: list[SpanDiff] = []

    if max(len(base_content), len(cand_content)) <= 5:
        out.extend(_span_diff_greedy(base_content, cand_content))
    else:
        out.extend(_span_diff_aligned(base_content, cand_content))

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


def _span_diff_greedy(base_content: list[Any], cand_content: list[Any]) -> list[SpanDiff]:
    """Original greedy per-index alignment. Exact for short lists."""
    out: list[SpanDiff] = []
    overlap = min(len(base_content), len(cand_content))
    for i in range(overlap):
        b = base_content[i] if isinstance(base_content[i], dict) else {}
        c = cand_content[i] if isinstance(cand_content[i], dict) else {}
        out.extend(_classify_block_pair(i, b, c))
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
    return out


def _span_diff_aligned(base_content: list[Any], cand_content: list[Any]) -> list[SpanDiff]:
    """Needleman-Wunsch alignment path for long block lists.

    Minimises sum of per-pair costs (identical < same-type-diff <
    type-mismatch) + gap costs. Emits one `SpanDiff` per aligned
    pair / gap. Block indices are from the baseline list for
    removes/matches and from the candidate list for adds — so
    downstream UIs always have a click-back target.
    """
    n = len(base_content)
    m = len(cand_content)
    inf = float("inf")
    # dp[i][j] = min cost aligning base[0..i] with cand[0..j]
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    # back[i][j] = which operation got us here: 'M', 'X' (cand gap), 'Y' (base gap)
    back = [[""] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _SPAN_GAP_COST
        back[i][0] = "Y"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _SPAN_GAP_COST
        back[0][j] = "X"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            b = base_content[i - 1] if isinstance(base_content[i - 1], dict) else {}
            c = cand_content[j - 1] if isinstance(cand_content[j - 1], dict) else {}
            match_cost = dp[i - 1][j - 1] + _span_pair_cost(b, c)
            gap_x = dp[i][j - 1] + _SPAN_GAP_COST  # candidate gap (insert)
            gap_y = dp[i - 1][j] + _SPAN_GAP_COST  # baseline gap (delete)
            best = min(match_cost, gap_x, gap_y)
            dp[i][j] = best
            back[i][j] = "M" if best == match_cost else ("X" if best == gap_x else "Y")

    # Traceback produces pair/gap ops in reverse order.
    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            ops.append(("M", i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "X":
            ops.append(("X", -1, j - 1))
            j -= 1
        else:  # "Y"
            ops.append(("Y", i - 1, -1))
            i -= 1
    ops.reverse()

    out: list[SpanDiff] = []
    for op, bi, ci in ops:
        if op == "M":
            b = base_content[bi] if isinstance(base_content[bi], dict) else {}
            c = cand_content[ci] if isinstance(cand_content[ci], dict) else {}
            out.extend(_classify_block_pair(bi, b, c))
        elif op == "Y":  # baseline block with no candidate counterpart → removed
            b = base_content[bi] if isinstance(base_content[bi], dict) else {}
            out.append(
                SpanDiff(
                    kind=_addremove_kind(b.get("type"), removed=True),
                    block_index=bi,
                    baseline=b,
                    candidate=None,
                    summary=_describe_addremove(b, removed=True),
                )
            )
        else:  # "X": candidate block with no baseline counterpart → added
            c = cand_content[ci] if isinstance(cand_content[ci], dict) else {}
            out.append(
                SpanDiff(
                    kind=_addremove_kind(c.get("type"), removed=False),
                    block_index=ci,
                    baseline=None,
                    candidate=c,
                    summary=_describe_addremove(c, removed=False),
                )
            )
    return out


def _span_pair_cost(b: dict[str, Any], c: dict[str, Any]) -> float:
    """Alignment cost between two candidate same-index content blocks.

    Identical → 0. Same-type-but-different → small. Type mismatch →
    larger (to nudge the aligner toward add+remove + same-type pair).
    """
    if b == c:
        return _SPAN_IDENTICAL_COST
    if b.get("type") != c.get("type"):
        return _SPAN_TYPE_MISMATCH_COST
    return _SPAN_SAME_TYPE_DIFF_COST


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


# ---- token-level diff ----------------------------------------------------


@dataclass
class TokenPairDelta:
    """Per-pair token usage delta (one entry per baseline/candidate turn).

    Zero-ish deltas on every pair is the normal case. Large deltas on a
    few pairs is the signal we care about — that's where a prompt edit
    or a model swap changed generation length.
    """

    pair_index: int
    baseline: dict[str, int]  # {input_tokens, output_tokens, thinking_tokens}
    candidate: dict[str, int]
    delta: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenDistSummary:
    """Distribution summary for one token dimension across a trace."""

    median: float
    p25: float
    p75: float
    p95: float
    maximum: int
    total: int


@dataclass
class TokenDiff:
    """Full token-level report: aggregated distributions + per-pair deltas."""

    dimensions: list[str]  # ["input_tokens", "output_tokens", "thinking_tokens"]
    baseline: dict[str, TokenDistSummary]
    candidate: dict[str, TokenDistSummary]
    pair_count: int
    # Per-dimension shift: candidate median - baseline median, normalised
    # by baseline median (fraction, e.g. 0.42 = 42% increase).
    normalised_shift: dict[str, float]
    # Individual per-pair deltas, sorted by worst absolute shift first
    # (so callers can `[:k]` the top-k offenders).
    worst_pairs: list[TokenPairDelta]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": list(self.dimensions),
            "baseline": {k: asdict(v) for k, v in self.baseline.items()},
            "candidate": {k: asdict(v) for k, v in self.candidate.items()},
            "pair_count": self.pair_count,
            "normalised_shift": dict(self.normalised_shift),
            "worst_pairs": [p.to_dict() for p in self.worst_pairs],
        }


_TOKEN_DIMS = ("input_tokens", "output_tokens", "thinking_tokens")


def token_diff(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    top_k_pairs: int = 10,
) -> TokenDiff:
    """Token-level breakdown: per-pair input/output/thinking token deltas.

    Aligned by index on `chat_response` records — the Rust differ uses
    the same alignment, so the resulting `pair_index` here matches the
    one in the main DiffReport's drill-down.
    """
    base_usages = _extract_usages(baseline_records)
    cand_usages = _extract_usages(candidate_records)
    pair_count = min(len(base_usages), len(cand_usages))

    base_summary = {dim: _summarise(base_usages, dim) for dim in _TOKEN_DIMS}
    cand_summary = {dim: _summarise(cand_usages, dim) for dim in _TOKEN_DIMS}

    normalised: dict[str, float] = {}
    for dim in _TOKEN_DIMS:
        b_med = base_summary[dim].median
        c_med = cand_summary[dim].median
        if b_med == 0 and c_med == 0:
            normalised[dim] = 0.0
        elif b_med == 0:
            normalised[dim] = float("inf") if c_med > 0 else 0.0
        else:
            normalised[dim] = (c_med - b_med) / b_med

    pair_deltas: list[TokenPairDelta] = []
    for i in range(pair_count):
        b_use = base_usages[i]
        c_use = cand_usages[i]
        delta = {dim: c_use[dim] - b_use[dim] for dim in _TOKEN_DIMS}
        pair_deltas.append(
            TokenPairDelta(pair_index=i, baseline=b_use, candidate=c_use, delta=delta)
        )
    # Rank by absolute sum of delta across dimensions (bigger = more surprising).
    pair_deltas.sort(key=lambda p: -sum(abs(v) for v in p.delta.values()))

    return TokenDiff(
        dimensions=list(_TOKEN_DIMS),
        baseline=base_summary,
        candidate=cand_summary,
        pair_count=pair_count,
        normalised_shift=normalised,
        worst_pairs=pair_deltas[: max(0, top_k_pairs)],
    )


def _extract_usages(records: list[dict[str, Any]]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for rec in records:
        if rec.get("kind") != "chat_response":
            continue
        usage = (rec.get("payload") or {}).get("usage") or {}
        out.append({dim: int(usage.get(dim) or 0) for dim in _TOKEN_DIMS})
    return out


def _summarise(values: list[dict[str, int]], dim: str) -> TokenDistSummary:
    series = sorted(v[dim] for v in values)
    if not series:
        return TokenDistSummary(median=0.0, p25=0.0, p75=0.0, p95=0.0, maximum=0, total=0)

    def _pct(p: float) -> float:
        if len(series) == 1:
            return float(series[0])
        # Linear interpolation, matching numpy.percentile default.
        idx = (len(series) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(series) - 1)
        frac = idx - lo
        return series[lo] * (1 - frac) + series[hi] * frac

    return TokenDistSummary(
        median=_pct(0.5),
        p25=_pct(0.25),
        p75=_pct(0.75),
        p95=_pct(0.95),
        maximum=series[-1],
        total=sum(series),
    )


def render_token_diff(diff: TokenDiff) -> str:
    """Human-readable summary for the token-level report."""
    if diff.pair_count == 0:
        return "token-level diff: no paired chat_response records"
    lines = [f"Token-level diff — {diff.pair_count} pair(s):"]
    for dim in diff.dimensions:
        b = diff.baseline[dim]
        c = diff.candidate[dim]
        shift = diff.normalised_shift[dim]
        shift_str = "+inf" if shift == float("inf") else f"{shift:+.2%}"
        lines.append(
            f"  {dim:<18}  baseline median {b.median:>8.1f} "
            f"p95 {b.p95:>8.1f}  →  candidate median {c.median:>8.1f} "
            f"p95 {c.p95:>8.1f}  shift {shift_str}"
        )
    if diff.worst_pairs:
        lines.append("  worst pairs (by absolute token delta):")
        for p in diff.worst_pairs[:5]:
            lines.append(
                f"    · turn #{p.pair_index}: "
                f"input {p.delta['input_tokens']:+d}, "
                f"output {p.delta['output_tokens']:+d}, "
                f"thinking {p.delta['thinking_tokens']:+d}"
            )
    return "\n".join(lines)


# ---- policy-level diff ---------------------------------------------------


@dataclass
class PolicyRule:
    """One declarative constraint a policy can check against a trace.

    `scope` controls how the rule is applied to traces containing multiple
    user-initiated sessions (e.g. a production trace with N tickets
    concatenated). ``"trace"`` (the default, for back-compat) runs the rule
    once across the whole record list. ``"session"`` partitions the trace
    into user-initiated sessions — each request whose most recent
    non-system message is from the user starts a new session — and runs
    the rule independently within each session, emitting one violation
    per offending session. For ordering rules like ``must_call_before``,
    session scope is almost always what you want on multi-ticket traces:
    otherwise a correct ordering in ticket #1 masks every subsequent
    violation.
    """

    id: str  # e.g. "call-backup-before-migration"
    kind: str  # see _POLICY_KINDS
    params: dict[str, Any]
    severity: str = "error"  # "warning" | "error" | "critical"
    description: str = ""
    scope: str = "trace"  # "trace" | "session"


@dataclass
class PolicyViolation:
    """One concrete failure of a `PolicyRule` against a trace."""

    rule_id: str
    kind: str
    severity: str
    pair_index: int | None  # None for whole-trace violations
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyDiff:
    """Per-trace policy check output + the candidate-vs-baseline delta."""

    baseline_violations: list[PolicyViolation]
    candidate_violations: list[PolicyViolation]
    # `regressions` = violations present in candidate but not baseline.
    # `fixes` = the reverse — no longer violated in candidate.
    regressions: list[PolicyViolation]
    fixes: list[PolicyViolation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_violations": [v.to_dict() for v in self.baseline_violations],
            "candidate_violations": [v.to_dict() for v in self.candidate_violations],
            "regressions": [v.to_dict() for v in self.regressions],
            "fixes": [v.to_dict() for v in self.fixes],
        }


_POLICY_KINDS = {
    "must_call_before",
    "must_call_once",
    "no_call",
    "max_turns",
    "required_stop_reason",
    "max_total_tokens",
    "must_include_text",
    "forbidden_text",
}


def load_policy(data: Any) -> list[PolicyRule]:
    """Parse a policy overlay (dict or list) into a list of rules.

    Accepts either:
    - `{"rules": [ {...rule...}, ... ]}` — typical YAML overlay shape.
    - `[ {...rule...}, ... ]` — bare rule list.
    """
    from shadow.errors import ShadowConfigError

    if isinstance(data, dict):
        raw_rules = data.get("rules")
        if not isinstance(raw_rules, list):
            raise ShadowConfigError(
                "policy YAML must have a top-level `rules:` list (or be a list itself)"
            )
    elif isinstance(data, list):
        raw_rules = data
    else:
        raise ShadowConfigError(
            f"policy input must be a mapping or list; got {type(data).__name__}"
        )

    out: list[PolicyRule] = []
    for i, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            raise ShadowConfigError(f"policy rule #{i} must be a mapping")
        kind = raw.get("kind")
        if kind not in _POLICY_KINDS:
            raise ShadowConfigError(
                f"policy rule #{i} has unknown kind {kind!r}.\n"
                f"hint: supported kinds are {sorted(_POLICY_KINDS)}"
            )
        scope = raw.get("scope", "trace")
        if scope not in ("trace", "session"):
            raise ShadowConfigError(
                f"policy rule #{i} has invalid scope {scope!r}; " f"must be 'trace' or 'session'."
            )
        out.append(
            PolicyRule(
                id=str(raw.get("id") or f"rule-{i}"),
                kind=kind,
                params=dict(raw.get("params") or {}),
                severity=str(raw.get("severity") or "error"),
                description=str(raw.get("description") or ""),
                scope=scope,
            )
        )
    return out


def check_policy(records: list[dict[str, Any]], rules: list[PolicyRule]) -> list[PolicyViolation]:
    """Run every rule against one trace; return the flat violation list.

    Rules with ``scope="session"`` are applied independently within each
    user-initiated session (see :func:`_compute_session_of_pair`), so a
    single well-ordered ticket cannot mask subsequent violations.
    """
    violations: list[PolicyViolation] = []
    tool_calls = _extract_tool_call_sequence(records)
    responses = [r for r in records if r.get("kind") == "chat_response"]

    needs_sessions = any(rule.scope == "session" for rule in rules)
    session_of_pair = _compute_session_of_pair(records) if needs_sessions else []

    for rule in rules:
        if rule.scope == "session":
            violations.extend(_check_rule_per_session(rule, tool_calls, responses, session_of_pair))
        else:
            violations.extend(_check_single_rule(rule, records, tool_calls, responses))
    return violations


def _extract_tool_call_sequence(
    records: list[dict[str, Any]],
) -> list[tuple[int, str, dict[str, Any]]]:
    """Return [(pair_index, tool_name, args), ...] in call order."""
    out: list[tuple[int, str, dict[str, Any]]] = []
    pair_idx = -1
    for rec in records:
        if rec.get("kind") != "chat_response":
            continue
        pair_idx += 1
        payload = rec.get("payload") or {}
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append((pair_idx, str(block.get("name") or ""), dict(block.get("input") or {})))
    return out


def _is_session_start(request_payload: dict[str, Any]) -> bool:
    """True when a ``chat_request`` begins a new user-initiated session.

    Detection walks ``payload.messages`` backward, skipping ``system``
    entries, and checks whether the most recent remaining role is
    ``"user"``. Tool-result follow-ups (``role == "tool"``) and assistant
    continuations (``role == "assistant"``) are correctly treated as
    mid-session turns. Requests whose ``messages`` array is absent or
    empty are conservatively treated as session starts — the typical
    shape for importers (A2A, MCP) that flatten every task into a
    single-message request.
    """
    messages = request_payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return True
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            continue
        return role == "user"
    return True


# Only `tool_use` marks a mid-turn response: the agent pauses to run a
# tool and will resume. Every other stop_reason — end_turn, max_tokens,
# stop_sequence, content_filter, refusal, error, and any novel value a
# foreign importer might introduce — signals the agent is done speaking,
# which means the next request is necessarily a new session. Treating
# unknown stop_reasons as terminal is the robust default for foreign
# trace shapes; a non-tool_use value never legitimately continues a turn.
_CONTINUATION_STOP_REASON = "tool_use"


def _compute_session_of_pair(records: list[dict[str, Any]]) -> list[int]:
    """Map each response's pair_index → session index.

    Three signals can mark a session boundary. They're ranked by
    authority so adapters that have perfect knowledge can declare it
    without Shadow second-guessing them:

    1. **Explicit metadata markers (authoritative).** When the trace
       contains two or more ``metadata`` records, each new metadata
       record marks a new session. This is the signal framework
       adapters emit via :meth:`~shadow.sdk.Session.record_metadata`
       on events like ``CrewKickoffStartedEvent`` or an AG2
       ``initiate_chat``. Heuristic signals #2 and #3 are ignored in
       this mode — the adapter knows exactly where sessions split.
    2. **Request shape.** The request's most recent non-system message
       is from the user (see :func:`_is_session_start`). Covers SDK-
       instrumented traces where tool results are appended to
       ``messages`` mid-session.
    3. **Prior terminal stop reason.** The previous response ended
       with anything other than ``tool_use``. Recovers session
       boundaries in traces where ``messages`` were abbreviated,
       post-mutated, or imported from a foreign format that didn't
       preserve them.

    Records without request/response pairing (orphan responses at the
    start, pure metadata-only traces) are handled defensively. Returns
    a list of length ``len(responses)``.
    """
    metadata_count = sum(1 for r in records if r.get("kind") == "metadata")
    use_explicit_markers = metadata_count >= 2

    session_of_pair: list[int] = []
    session_idx = -1
    pending_session: int | None = None
    prior_stop_terminal = True  # before any response, the "prior" state
    # is "agent not mid-turn" - the first request starts session 0.
    seen_first_metadata = False
    for rec in records:
        kind = rec.get("kind")
        if kind == "metadata":
            if use_explicit_markers:
                # Every metadata record after the first starts a new
                # session. The first one opens session 0 just by
                # existing.
                if seen_first_metadata:
                    session_idx += 1
                else:
                    seen_first_metadata = True
                    if session_idx < 0:
                        session_idx = 0
            continue
        if kind == "chat_request":
            payload = rec.get("payload") or {}
            if use_explicit_markers:
                # Metadata records drove the session_idx; just attach
                # the current request to the active session (bootstrap
                # session 0 if the first marker hasn't appeared yet).
                if session_idx < 0:
                    session_idx = 0
                pending_session = session_idx
            else:
                if _is_session_start(payload) or prior_stop_terminal:
                    session_idx += 1
                pending_session = session_idx if session_idx >= 0 else 0
                if session_idx < 0:
                    session_idx = 0
            # Consume the terminal marker - only the first request after
            # a terminal response gets the boundary; subsequent requests
            # inside the same session don't.
            prior_stop_terminal = False
        elif kind == "chat_response":
            if pending_session is None:
                session_idx = max(session_idx, 0)
                pending_session = session_idx
            session_of_pair.append(pending_session)
            pending_session = None
            stop = (rec.get("payload") or {}).get("stop_reason", "")
            prior_stop_terminal = stop != _CONTINUATION_STOP_REASON
    return session_of_pair


def _check_rule_per_session(
    rule: PolicyRule,
    tool_calls: list[tuple[int, str, dict[str, Any]]],
    responses: list[dict[str, Any]],
    session_of_pair: list[int],
) -> list[PolicyViolation]:
    """Apply ``rule`` once per session; combine the per-session violations.

    Sessions with no tool calls *and* no responses are skipped (nothing
    to check). Sessions with only one of the two are still evaluated so
    rules like ``no_call`` (tool-call only) and ``required_stop_reason``
    (response only) both work correctly.
    """
    if not session_of_pair:
        # No session info — safest fallback is to treat the entire trace as
        # one session so behavior matches the trace-scoped path.
        return _check_single_rule(rule, [], tool_calls, responses)

    num_sessions = max(session_of_pair) + 1
    out: list[PolicyViolation] = []
    for s_idx in range(num_sessions):
        s_tool_calls = [tc for tc in tool_calls if session_of_pair[tc[0]] == s_idx]
        s_responses = [r for i, r in enumerate(responses) if session_of_pair[i] == s_idx]
        if not s_tool_calls and not s_responses:
            continue
        out.extend(_check_single_rule(rule, [], s_tool_calls, s_responses))
    return out


def _check_single_rule(
    rule: PolicyRule,
    records: list[dict[str, Any]],
    tool_calls: list[tuple[int, str, dict[str, Any]]],
    responses: list[dict[str, Any]],
) -> list[PolicyViolation]:
    ps = rule.params
    if rule.kind == "must_call_before":
        first = ps.get("first")
        then = ps.get("then")
        if not isinstance(first, str) or not isinstance(then, str):
            return [_whole_trace_violation(rule, "missing `first`/`then` params")]
        first_idx = next((i for i, (_, n, _) in enumerate(tool_calls) if n == first), -1)
        then_idx = next((i for i, (_, n, _) in enumerate(tool_calls) if n == then), -1)
        if then_idx == -1:
            return []  # rule doesn't apply if `then` never called
        if first_idx == -1 or first_idx >= then_idx:
            pair = tool_calls[then_idx][0]
            return [
                PolicyViolation(
                    rule_id=rule.id,
                    kind=rule.kind,
                    severity=rule.severity,
                    pair_index=pair,
                    detail=(
                        f"`{first}` must be called before `{then}` "
                        f"(was: {first_idx=}, {then_idx=})"
                    ),
                )
            ]
        return []

    if rule.kind == "must_call_once":
        name = ps.get("tool")
        if not isinstance(name, str):
            return [_whole_trace_violation(rule, "missing `tool` param")]
        count = sum(1 for _, n, _ in tool_calls if n == name)
        if count != 1:
            return [
                _whole_trace_violation(
                    rule, f"tool `{name}` called {count} times; expected exactly 1"
                )
            ]
        return []

    if rule.kind == "no_call":
        name = ps.get("tool")
        if not isinstance(name, str):
            return [_whole_trace_violation(rule, "missing `tool` param")]
        return [
            PolicyViolation(
                rule_id=rule.id,
                kind=rule.kind,
                severity=rule.severity,
                pair_index=pair,
                detail=f"forbidden tool `{name}` called",
            )
            for pair, n, _ in tool_calls
            if n == name
        ]

    if rule.kind == "max_turns":
        limit = ps.get("limit")
        if not isinstance(limit, int):
            return [_whole_trace_violation(rule, "missing integer `limit` param")]
        if len(responses) > limit:
            return [
                _whole_trace_violation(rule, f"trace has {len(responses)} turns; max is {limit}")
            ]
        return []

    if rule.kind == "required_stop_reason":
        allowed = ps.get("allowed")
        if not isinstance(allowed, list) or not responses:
            return [] if responses else [_whole_trace_violation(rule, "trace has no responses")]
        last = (responses[-1].get("payload") or {}).get("stop_reason")
        if last not in allowed:
            return [
                _whole_trace_violation(
                    rule,
                    f"final stop_reason {last!r} not in allowed set {allowed}",
                )
            ]
        return []

    if rule.kind == "max_total_tokens":
        limit = ps.get("limit")
        if not isinstance(limit, int):
            return [_whole_trace_violation(rule, "missing integer `limit` param")]
        total = 0
        for resp in responses:
            usage = (resp.get("payload") or {}).get("usage") or {}
            total += int(usage.get("input_tokens") or 0)
            total += int(usage.get("output_tokens") or 0)
            total += int(usage.get("thinking_tokens") or 0)
        if total > limit:
            return [_whole_trace_violation(rule, f"total token usage {total} exceeds max {limit}")]
        return []

    if rule.kind == "must_include_text":
        needle = ps.get("text")
        if not isinstance(needle, str) or not responses:
            return (
                [_whole_trace_violation(rule, "missing `text` param or empty trace")]
                if not isinstance(needle, str)
                else []
            )
        haystack = _gather_response_text(responses)
        if needle not in haystack:
            return [
                _whole_trace_violation(rule, f"required text {needle!r} not found in any response")
            ]
        return []

    if rule.kind == "forbidden_text":
        needle = ps.get("text")
        if not isinstance(needle, str) or not responses:
            return (
                [_whole_trace_violation(rule, "missing `text` param")]
                if not isinstance(needle, str)
                else []
            )
        out: list[PolicyViolation] = []
        for i, resp in enumerate(responses):
            text = _gather_response_text([resp])
            if needle in text:
                out.append(
                    PolicyViolation(
                        rule_id=rule.id,
                        kind=rule.kind,
                        severity=rule.severity,
                        pair_index=i,
                        detail=f"forbidden text {needle!r} present in response",
                    )
                )
        return out

    # pragma: no cover — unreachable given load_policy validation
    return [_whole_trace_violation(rule, f"unhandled rule kind {rule.kind!r}")]


def _whole_trace_violation(rule: PolicyRule, detail: str) -> PolicyViolation:
    return PolicyViolation(
        rule_id=rule.id,
        kind=rule.kind,
        severity=rule.severity,
        pair_index=None,
        detail=detail,
    )


def _gather_response_text(responses: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for resp in responses:
        payload = resp.get("payload") or {}
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    out.append(text)
    return "\n".join(out)


def policy_diff(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    rules: list[PolicyRule],
) -> PolicyDiff:
    """Apply a policy overlay to both traces; classify regressions vs fixes."""
    base_v = check_policy(baseline_records, rules)
    cand_v = check_policy(candidate_records, rules)

    def _key(v: PolicyViolation) -> tuple[str, int | None, str]:
        return (v.rule_id, v.pair_index, v.detail)

    base_keys = {_key(v) for v in base_v}
    cand_keys = {_key(v) for v in cand_v}
    regressions = [v for v in cand_v if _key(v) not in base_keys]
    fixes = [v for v in base_v if _key(v) not in cand_keys]
    return PolicyDiff(
        baseline_violations=base_v,
        candidate_violations=cand_v,
        regressions=regressions,
        fixes=fixes,
    )


def render_policy_diff(diff: PolicyDiff) -> str:
    """Human-readable policy-diff summary."""
    lines = [
        f"Policy diff — {len(diff.baseline_violations)} baseline "
        f"violation(s), {len(diff.candidate_violations)} candidate violation(s)"
    ]
    if diff.regressions:
        lines.append(f"  regressions ({len(diff.regressions)}):")
        for v in diff.regressions:
            scope = f"turn #{v.pair_index}" if v.pair_index is not None else "trace"
            lines.append(f"    ✗ [{v.severity}] {v.rule_id} @ {scope}: {v.detail}")
    if diff.fixes:
        lines.append(f"  fixes ({len(diff.fixes)}):")
        for v in diff.fixes:
            scope = f"turn #{v.pair_index}" if v.pair_index is not None else "trace"
            lines.append(f"    ✓ [{v.severity}] {v.rule_id} @ {scope}: {v.detail}")
    if not diff.regressions and not diff.fixes:
        lines.append("  (no change in policy outcomes)")
    return "\n".join(lines)


__all__ = [
    "PolicyDiff",
    "PolicyRule",
    "PolicyViolation",
    "SessionDiff",
    "SpanDiff",
    "TokenDiff",
    "TokenDistSummary",
    "TokenPairDelta",
    "check_policy",
    "diff_by_session",
    "load_policy",
    "policy_diff",
    "render_policy_diff",
    "render_session_summary",
    "render_spans",
    "render_token_diff",
    "span_diff",
    "token_diff",
]
