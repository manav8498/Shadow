#!/usr/bin/env python3
"""A non-Shadow tool calling `shadow.align` directly.

Demonstrates the Phase 6 standalone-library use case: an external
eval framework, observability platform, or test harness that wants
Shadow's trace alignment + divergence detection WITHOUT the rest of
the diagnose-pr / 9-axis / policy machinery.

Imports only `shadow.align`. No other Shadow surface is touched.

Run:
    python examples/standalone-align/compare_tool_trajectories.py
"""

from __future__ import annotations

import json

# The only Shadow import — the standalone alignment library.
from shadow.align import (
    Divergence,
    first_divergence,
    tool_arg_delta,
    top_k_divergences,
    trajectory_distance,
)


def demo_trajectory_distance() -> None:
    """An eval framework with only tool-call sequences (no full
    `.agentlog` records) can still measure how much the trajectory
    drifted between two agent runs."""
    print("=" * 60)
    print("1. trajectory_distance — Levenshtein on tool-call sequences")
    print("=" * 60)

    baseline = ["search_docs", "summarize", "draft_email", "send"]
    candidate_minor = ["search_docs", "summarize", "draft_email"]
    candidate_major = ["search_docs", "issue_refund"]

    print(f"  baseline:        {baseline}")
    print(f"  candidate (minor edit):    {candidate_minor}")
    print(f"    distance: {trajectory_distance(baseline, candidate_minor):.3f}")
    print(f"  candidate (different agent): {candidate_major}")
    print(f"    distance: {trajectory_distance(baseline, candidate_major):.3f}")
    print()


def demo_tool_arg_delta() -> None:
    """A tool-call schema-change detector can use this to spot
    subtle argument-shape differences between two recorded calls.
    """
    print("=" * 60)
    print("2. tool_arg_delta — structural diff of two JSON values")
    print("=" * 60)

    old_call = {
        "tool": "issue_refund",
        "args": {
            "order_id": "1234",
            "amount": 45.0,
            "reason": "duplicate charge",
        },
    }
    new_call = {
        "tool": "issue_refund",
        "args": {
            "order_id": "1234",
            "amount": 45.00,
            "reason": "duplicate-charge",  # hyphen vs space
            "approver": "ops-team",  # NEW field
        },
    }
    deltas = tool_arg_delta(old_call, new_call)
    print(f"  detected {len(deltas)} delta(s):")
    for d in deltas:
        print(f"    [{d.kind:13}] {d.path}: {d.old!r} → {d.new!r}")
    print()


def demo_first_divergence() -> None:
    """An observability platform receives two traces and wants to
    pinpoint the FIRST point at which the candidate diverged from
    the baseline. shadow.align.first_divergence answers this in
    one call."""
    print("=" * 60)
    print("3. first_divergence + top_k_divergences — full trace diff")
    print("=" * 60)

    # Build two minimal traces using shadow.sdk for the demo. The
    # `shadow.sdk.Session` import IS a Shadow dependency, but only
    # for constructing test data — a real external tool would feed
    # in records it parsed itself from any source.
    from importlib import resources

    from shadow import _core
    import shadow.quickstart_data as q

    fixtures = resources.files(q) / "fixtures"
    base = _core.parse_agentlog(fixtures.joinpath("baseline.agentlog").read_bytes())
    cand = _core.parse_agentlog(fixtures.joinpath("candidate.agentlog").read_bytes())

    fd: Divergence | None = first_divergence(base, cand)
    if fd is None:
        print("  traces agree end-to-end")
    else:
        print(f"  first divergence:")
        print(f"    baseline_turn:  {fd.baseline_turn}")
        print(f"    candidate_turn: {fd.candidate_turn}")
        print(f"    kind:           {fd.kind}")
        print(f"    primary_axis:   {fd.primary_axis}")
        print(f"    confidence:     {fd.confidence:.2f}")
        print(f"    explanation:    {fd.explanation}")

    print()
    top = top_k_divergences(base, cand, k=3)
    print(f"  top {len(top)} divergence(s):")
    for i, d in enumerate(top, 1):
        print(f"    {i}. [{d.primary_axis}] {d.kind}: {d.explanation}")
    print()


def main() -> None:
    print()
    print("Shadow's standalone alignment library — used WITHOUT diagnose-pr.")
    print("Three primitives, one import: `from shadow.align import ...`.")
    print()
    demo_trajectory_distance()
    demo_tool_arg_delta()
    demo_first_divergence()
    print("Done. None of `shadow.diagnose_pr`, `shadow.causal`, or `shadow.cli`")
    print("was imported — pure alignment primitives only.")


if __name__ == "__main__":
    main()
