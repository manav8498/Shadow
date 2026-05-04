"""Internal implementation of the `shadow.align` public surface.

Three of the five public functions (`align_traces`,
`first_divergence`, `top_k_divergences`) delegate to
`shadow._core.compute_diff_report` so the alignment matches what
the 9-axis differ already does — there's one canonical pairing
algorithm in the project, not two.

The other two (`trajectory_distance`, `tool_arg_delta`) don't
themselves call Rust, but importing `shadow.align` still loads
the parent `shadow` package which pulls in the Rust extension
via `shadow/__init__.py`. So the v0.1 align surface still
requires the Rust extension to be installed. The spec-literal
top-level `shadow_align` package (no parent dependency) is
deferred to v0.2.

Known limitations:
  * `first_divergence` and `align_traces` return None / zero
    turns when one side is empty (asymmetric pair count). The
    underlying Rust differ surfaces no divergence in this case;
    v0.2 will add an explicit "structural_drift_full" divergence
    kind for the empty-candidate case.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

ArgDeltaKind = Literal["added", "removed", "changed", "type_changed"]


@dataclass(frozen=True)
class AlignedTurn:
    """One paired (baseline_index, candidate_index) entry.

    Either index can be `None` when one side is missing — that's a
    structural-drift turn in alignment terms (insertion or deletion).
    """

    baseline_index: int | None
    candidate_index: int | None
    cost: float
    """0.0 = perfect match, 1.0 = pure-gap insertion/deletion;
    intermediate values for partial matches."""


@dataclass(frozen=True)
class Alignment:
    """The full alignment of two traces' chat turns. `turns` is the
    canonical paired sequence; `total_cost` is its sum (lower is
    better).
    """

    turns: list[AlignedTurn] = field(default_factory=list)
    total_cost: float = 0.0


@dataclass(frozen=True)
class Divergence:
    """One identified divergence between baseline and candidate.

    `kind` is a coarse classification — `structural_drift` (a turn
    inserted/dropped), `tool_swap` (different tool called),
    `arg_delta` (same tool, different args), `text_drift` (text
    response semantic shift), `safety` (refusal flip), etc.

    Mirror of `shadow._core.FirstDivergence` for callers who want
    a stable typed surface that doesn't change when the underlying
    Rust struct evolves.
    """

    baseline_turn: int
    candidate_turn: int
    kind: str
    primary_axis: str
    explanation: str
    confidence: float


@dataclass(frozen=True)
class ArgDelta:
    """One leaf-level change between two JSON values, keyed by a
    slash-separated path (`/key/0/sub`)."""

    path: str
    kind: ArgDeltaKind
    old: Any = None
    new: Any = None


# ---------------------------------------------------------------------------
# Public API — wrappers over `shadow._core.compute_diff_report`
# ---------------------------------------------------------------------------


def _records_or_raise(label: str, recs: Sequence[dict[str, Any]]) -> None:
    if not isinstance(recs, list):
        raise TypeError(f"{label} must be a list of records (dicts), got {type(recs).__name__}")


def first_divergence(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
) -> Divergence | None:
    """Return the FIRST point at which baseline and candidate
    meaningfully diverge in alignment order, or `None` when they
    agree end-to-end.

    Wraps the Rust differ's internal alignment via
    `shadow._core.compute_diff_report`. Equivalent to taking
    `report["first_divergence"]` and reshaping into the stable
    `Divergence` dataclass.

    Special case: when one side is empty and the other has chat
    pairs, the underlying Rust differ surfaces no divergence (the
    pair-count is zero, so its alignment loop never fires). This
    function detects that asymmetry up front and returns a
    `structural_drift_full` Divergence — closing the v0.1
    documented limitation.
    """
    _records_or_raise("baseline", baseline)
    _records_or_raise("candidate", candidate)

    base_pairs = _count_chat_pairs(baseline)
    cand_pairs = _count_chat_pairs(candidate)
    if base_pairs == 0 and cand_pairs == 0:
        return None
    if base_pairs == 0 or cand_pairs == 0:
        return Divergence(
            baseline_turn=0,
            candidate_turn=0,
            kind="structural_drift_full",
            primary_axis="trajectory",
            explanation=(
                f"asymmetric corpus: baseline has {base_pairs} chat pair(s), "
                f"candidate has {cand_pairs}"
            ),
            confidence=1.0,
        )

    from shadow import _core

    report = _core.compute_diff_report(list(baseline), list(candidate), None, None)
    fd = report.get("first_divergence")
    if fd is None:
        return None
    return _divergence_from_dict(fd)


def _count_chat_pairs(records: Sequence[dict[str, Any]]) -> int:
    """Count the number of chat_request -> chat_response pairs in a
    record list. Used by `first_divergence` / `align_traces` to
    detect the asymmetric-corpus case the underlying differ doesn't
    flag."""
    pairs = 0
    pending_req = False
    for r in records:
        kind = r.get("kind")
        if kind == "chat_request":
            pending_req = True
        elif kind == "chat_response" and pending_req:
            pairs += 1
            pending_req = False
    return pairs


def top_k_divergences(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    k: int = 5,
) -> list[Divergence]:
    """Return up to `k` ranked divergences, sorted by severity
    times downstream impact (the Rust differ's own `divergences`
    field). Empty list when traces agree."""
    _records_or_raise("baseline", baseline)
    _records_or_raise("candidate", candidate)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    from shadow import _core

    report = _core.compute_diff_report(list(baseline), list(candidate), None, None)
    divs = report.get("divergences") or []
    return [_divergence_from_dict(d) for d in divs[:k]]


def align_traces(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
) -> Alignment:
    """Pair every baseline chat turn to its best-match candidate
    turn (and vice versa).

    The Rust differ runs the same alignment internally; this
    surface exposes the per-turn pairing so external tools can
    implement their own per-turn analyses on top.

    v0.1 returns a coarse pairing reconstructed from the differ's
    `pair_count` + `first_divergence`. Future versions will expose
    the full alignment matrix from `_core` directly once the Rust
    `Alignment` struct is stabilised for PyO3 export.
    """
    _records_or_raise("baseline", baseline)
    _records_or_raise("candidate", candidate)
    from shadow import _core

    report = _core.compute_diff_report(list(baseline), list(candidate), None, None)
    pair_count = int(report.get("pair_count", 0))
    fd = report.get("first_divergence")
    turns: list[AlignedTurn] = []
    for i in range(pair_count):
        if fd is not None and i == int(fd.get("baseline_turn", -1)):
            turns.append(
                AlignedTurn(
                    baseline_index=i,
                    candidate_index=int(fd.get("candidate_turn", i)),
                    cost=1.0 - float(fd.get("confidence", 1.0)) * 0.0 + 0.5,
                )
            )
        else:
            turns.append(AlignedTurn(baseline_index=i, candidate_index=i, cost=0.0))
    total = sum(t.cost for t in turns)
    return Alignment(turns=turns, total_cost=total)


# ---------------------------------------------------------------------------
# Public API — pure-Python (no Rust dependency)
# ---------------------------------------------------------------------------


def trajectory_distance(
    baseline_tools: Sequence[str],
    candidate_tools: Sequence[str],
) -> float:
    """Levenshtein edit distance between two flat tool-name
    sequences, normalised to [0.0, 1.0] by the longer length.

    Useful when callers have only the tool-trajectory (e.g.
    `["search", "summarize", "issue_refund"]`) and not the full
    `.agentlog` records. Returns 0.0 when sequences are equal,
    1.0 when they share no tools at the same positions.
    """
    a = list(baseline_tools)
    b = list(candidate_tools)
    if not a and not b:
        return 0.0
    n = len(a)
    m = len(b)
    # Standard DP Levenshtein.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / max(n, m)


def tool_arg_delta(a: Any, b: Any, *, prefix: str = "") -> list[ArgDelta]:
    """Structural diff between two JSON values. Walks dicts, lists,
    and scalars; produces typed deltas keyed by slash-separated
    JSON-pointer-like paths.

    Examples:
        tool_arg_delta({"a": 1}, {"a": 2})
            -> [ArgDelta(path="/a", kind="changed", old=1, new=2)]
        tool_arg_delta({}, {"x": 0})
            -> [ArgDelta(path="/x", kind="added", new=0)]
        tool_arg_delta({"k": [1, 2]}, {"k": [1, 2, 3]})
            -> [ArgDelta(path="/k/2", kind="added", new=3)]
    """
    deltas: list[ArgDelta] = []
    _walk_arg_delta(a, b, prefix, deltas)
    return deltas


def _walk_arg_delta(a: Any, b: Any, path: str, out: list[ArgDelta]) -> None:
    if a is None and b is None:
        return
    if a is None:
        out.append(ArgDelta(path=path or "/", kind="added", new=b))
        return
    if b is None:
        out.append(ArgDelta(path=path or "/", kind="removed", old=a))
        return
    if type(a) is not type(b):
        out.append(ArgDelta(path=path or "/", kind="type_changed", old=a, new=b))
        return
    if isinstance(a, dict):
        keys = sorted(set(a) | set(b))
        for k in keys:
            sub = f"{path}/{k}"
            if k not in a:
                out.append(ArgDelta(path=sub, kind="added", new=b[k]))
            elif k not in b:
                out.append(ArgDelta(path=sub, kind="removed", old=a[k]))
            else:
                _walk_arg_delta(a[k], b[k], sub, out)
        return
    if isinstance(a, list):
        n_a, n_b = len(a), len(b)
        for i in range(min(n_a, n_b)):
            _walk_arg_delta(a[i], b[i], f"{path}/{i}", out)
        for i in range(n_a, n_b):
            out.append(ArgDelta(path=f"{path}/{i}", kind="added", new=b[i]))
        for i in range(n_b, n_a):
            out.append(ArgDelta(path=f"{path}/{i}", kind="removed", old=a[i]))
        return
    if a != b:
        out.append(ArgDelta(path=path or "/", kind="changed", old=a, new=b))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _divergence_from_dict(d: dict[str, Any]) -> Divergence:
    return Divergence(
        baseline_turn=int(d.get("baseline_turn", 0)),
        candidate_turn=int(d.get("candidate_turn", 0)),
        kind=str(d.get("kind", "unknown")),
        primary_axis=str(d.get("primary_axis", "unknown")),
        explanation=str(d.get("explanation", "")),
        confidence=float(d.get("confidence", 0.0)),
    )


__all__ = [
    "AlignedTurn",
    "Alignment",
    "ArgDelta",
    "ArgDeltaKind",
    "Divergence",
    "align_traces",
    "first_divergence",
    "tool_arg_delta",
    "top_k_divergences",
    "trajectory_distance",
]
