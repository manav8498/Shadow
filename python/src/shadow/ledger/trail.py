"""Pure graph-walk logic for ``shadow trail``.

Given a trace id and a list of ledger entries, walks backwards through
the (anchor → candidate) edge structure and emits one
:class:`TrailStep` per node visited. The result is a linear chain ready
for rendering.

No I/O, no Rich. Trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shadow.ledger.store import LedgerEntry

#: Default maximum trail depth. Five steps is enough to cover the most
#: common "what changed last week" investigation while keeping the
#: rendered panel within one terminal screen. Callers can override.
DEFAULT_DEPTH = 5


@dataclass(frozen=True)
class TrailStep:
    """One node in the walked chain.

    Each step records the trace id, its role at this point in the chain
    (``candidate`` when we walked into it via an anchor→candidate edge,
    ``anchor`` when we landed at a terminal node), and the originating
    entry's metadata so the renderer can show tier / driver / timestamp
    without re-reading the entry list.
    """

    trace_id: str
    role: str  # "candidate" | "anchor"
    kind: str  # entry.kind, e.g. "call" | "diff"
    tier: str | None
    driver_summary: str | None
    primary_axis: str | None
    timestamp: str
    # Only populated when this step represents a regressed candidate;
    # the anchor it points back to is the next step in the chain.
    parent_trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "role": self.role,
            "kind": self.kind,
            "tier": self.tier,
            "driver_summary": self.driver_summary,
            "primary_axis": self.primary_axis,
            "timestamp": self.timestamp,
            "parent_trace_id": self.parent_trace_id,
        }


@dataclass(frozen=True)
class TrailResult:
    """The full output of :func:`compute_trail`."""

    root_trace_id: str  # the trace id the user asked about
    steps: list[TrailStep] = field(default_factory=list)
    truncated_by_depth: bool = False  # True when depth limit cut the walk
    truncated_by_cycle: bool = False  # True when a cycle was detected

    @property
    def found(self) -> bool:
        """True when at least one step was discovered."""
        return bool(self.steps)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_trail(
    entries: list[LedgerEntry],
    *,
    trace_id: str,
    depth: int = DEFAULT_DEPTH,
) -> TrailResult:
    """Walk back from ``trace_id`` through the (anchor → candidate) edges.

    The walk advances one step at a time: at each iteration we look for
    the most-recent entry where the current trace id is the *candidate*,
    record it, and continue from that entry's anchor. The loop stops on
    any of:

        * ``depth`` steps emitted
        * a previously-visited trace id (cycle)
        * the current trace id appears only as an anchor in the ledger
          (terminal — no edge to follow back from)
        * the current trace id isn't in the ledger at all

    Returns a :class:`TrailResult` whose ``steps`` are ordered newest-
    first (the input id is the first step, its parent the second, etc.).
    """
    by_candidate: dict[str, list[LedgerEntry]] = {}
    by_anchor: dict[str, list[LedgerEntry]] = {}
    for e in entries:
        if e.candidate_id:
            by_candidate.setdefault(e.candidate_id, []).append(e)
        if e.anchor_id:
            by_anchor.setdefault(e.anchor_id, []).append(e)

    steps: list[TrailStep] = []
    visited: set[str] = set()
    current = trace_id
    truncated_by_depth = False
    truncated_by_cycle = False

    while current:
        if current in visited:
            truncated_by_cycle = True
            break
        visited.add(current)

        if len(steps) >= depth:
            truncated_by_depth = True
            break

        # Most-recent entry where `current` is the candidate, preferring
        # entries with a distinct anchor over self-comparisons. A user
        # who runs `shadow call X X` (sanity check) creates a self-loop
        # entry that we don't want to walk into; if the only entries are
        # self-loops we still emit the step but mark it as terminal.
        candidates = sorted(
            by_candidate.get(current, []),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        non_self = [e for e in candidates if e.anchor_id and e.anchor_id != current]
        chosen = non_self[0] if non_self else (candidates[0] if candidates else None)
        if chosen is not None:
            self_loop = chosen.anchor_id == current or not chosen.anchor_id
            steps.append(
                TrailStep(
                    trace_id=current,
                    role="candidate",
                    kind=chosen.kind,
                    tier=chosen.tier,
                    driver_summary=chosen.driver_summary,
                    primary_axis=chosen.primary_axis,
                    timestamp=chosen.timestamp,
                    parent_trace_id=None if self_loop else chosen.anchor_id,
                )
            )
            if self_loop:
                break
            current = chosen.anchor_id
            continue

        # No candidate edge — `current` only appears as an anchor.
        # Emit a terminal step labelled with whichever entry it last
        # served (most-recent-as-anchor) so the renderer can still show
        # the trace id with some context.
        anchors = sorted(
            by_anchor.get(current, []),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        if anchors:
            entry = anchors[0]
            steps.append(
                TrailStep(
                    trace_id=current,
                    role="anchor",
                    kind=entry.kind,
                    tier=None,  # the anchor itself wasn't called
                    driver_summary=None,
                    primary_axis=None,
                    timestamp=entry.timestamp,
                    parent_trace_id=None,
                )
            )
        # Either way, the chain ends here.
        break

    return TrailResult(
        root_trace_id=trace_id,
        steps=steps,
        truncated_by_depth=truncated_by_depth,
        truncated_by_cycle=truncated_by_cycle,
    )
