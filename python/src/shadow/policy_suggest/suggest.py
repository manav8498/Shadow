"""Mine recorded traces for plausible ``must_call_before`` policies.

Algorithm
---------
1. Partition records by ``meta.scenario_id`` (or treat as one bucket
   if no scenario IDs are present).
2. For each scenario, build the ordered list of tool calls.
3. Across all scenarios, count for every ordered pair (A, B):
     n_AB = number of scenarios where A appeared before B
     n_BA = number of scenarios where B appeared before A
     n_both = scenarios where both A and B appear
4. A pair (A, B) is a candidate ``must_call_before(A, B)`` rule when:
     n_AB / n_both >= min_consistency (default 1.0 — always)
     n_both >= min_support (default 3 — observed often enough)
     A != B

The output is a list of :class:`PolicySuggestion` rows with a
``confidence`` score, the candidate rule kind+params, and a one-line
``rationale`` for the human reviewer.

Why narrow
----------
"Always-before-the-other" is the only ordering invariant we can read
directly from the trace. Other policy kinds (``forbidden_text``,
``required_stop_reason``, dollar limits, tenant-specific exclusions)
need production semantics that aren't in the trace data. We
deliberately do NOT attempt to invent those — the human stays in the
loop.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicySuggestion:
    """A candidate policy rule the trace data supports.

    Attributes
    ----------
    rule_id
        Auto-generated ID (e.g. ``"verify_user-before-transfer_funds"``).
        Stable across runs of the same trace.
    kind
        Policy rule kind (currently only ``"must_call_before"``).
    params
        Parameters for the rule, ready to drop into a YAML file.
    confidence
        ``n_AB / n_both`` — fraction of scenarios where A preceded B.
    n_both
        How many scenarios contained both tools (the support).
    n_ordered
        How many of those scenarios had A before B.
    rationale
        Human-readable one-liner the reviewer can scan quickly.
    """

    rule_id: str
    kind: str
    params: dict[str, Any]
    confidence: float
    n_both: int
    n_ordered: int
    rationale: str


def _tool_call_sequence(records: list[dict[str, Any]]) -> list[str]:
    """Return the ordered list of tool names called in this scenario."""
    out: list[str] = []
    for rec in records:
        if rec.get("kind") != "chat_response":
            continue
        payload = rec.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name")
                if isinstance(name, str) and name:
                    out.append(name)
    return out


def _partition_by_scenario(
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group records by ``meta.scenario_id`` into a list of scenarios.

    Records without a scenario_id all go into a single bucket
    (treating the whole trace as one scenario).
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    has_any_scenario = False
    for rec in records:
        meta = rec.get("meta") or {}
        sid = meta.get("scenario_id") if isinstance(meta, dict) else None
        if isinstance(sid, str) and sid:
            buckets[sid].append(rec)
            has_any_scenario = True
        else:
            buckets["__default__"].append(rec)
    if not has_any_scenario:
        # Whole trace as one scenario.
        return [records]
    return list(buckets.values())


def suggest_policies(
    records: list[dict[str, Any]],
    *,
    min_consistency: float = 1.0,
    min_support: int = 3,
) -> list[PolicySuggestion]:
    """Return ranked ``must_call_before`` suggestions for the trace.

    Parameters
    ----------
    records
        Parsed agentlog records.
    min_consistency
        Required fraction of scenarios where A precedes B given both
        appear. Default 1.0 (must always precede). 0.9 yields more
        but lower-confidence suggestions.
    min_support
        Minimum number of scenarios that must contain both tools for a
        suggestion to be emitted. Default 3.

    Returns
    -------
    Suggestions sorted by support (n_both) descending, then by
    confidence descending.
    """
    scenarios = _partition_by_scenario(records)
    sequences = [_tool_call_sequence(s) for s in scenarios]

    # Track for each unordered pair {A, B} the directional counts.
    n_a_before_b: dict[tuple[str, str], int] = defaultdict(int)
    n_both: dict[tuple[str, str], int] = defaultdict(int)

    for seq in sequences:
        # Use first-occurrence index per tool for "who came first".
        first_idx: dict[str, int] = {}
        for i, name in enumerate(seq):
            if name not in first_idx:
                first_idx[name] = i

        tools_in_scenario = list(first_idx.keys())
        for a in tools_in_scenario:
            for b in tools_in_scenario:
                if a == b:
                    continue
                # Canonical order (sorted) for the "both appeared" bucket.
                lo, hi = (a, b) if a < b else (b, a)
                key_pair: tuple[str, str] = (lo, hi)
                if (a, b) <= (b, a):
                    # Only count "both appear" once per unordered pair.
                    n_both[key_pair] += 1
                # Directional count: A appeared before B in this scenario?
                if first_idx[a] < first_idx[b]:
                    n_a_before_b[(a, b)] += 1

    # Build suggestions for each (a, b) with sufficient support.
    suggestions: list[PolicySuggestion] = []
    for (a, b), n_ord in n_a_before_b.items():
        lo, hi = (a, b) if a < b else (b, a)
        key_pair = (lo, hi)
        n_both_count = n_both[key_pair]
        if n_both_count < min_support:
            continue
        confidence = n_ord / n_both_count if n_both_count > 0 else 0.0
        if confidence < min_consistency:
            continue
        rule_id = f"{a}-before-{b}"
        suggestions.append(
            PolicySuggestion(
                rule_id=rule_id,
                kind="must_call_before",
                params={"first": a, "then": b},
                confidence=confidence,
                n_both=n_both_count,
                n_ordered=n_ord,
                rationale=(
                    f"`{a}` precedes `{b}` in {n_ord}/{n_both_count} scenarios "
                    f"({confidence:.0%}); never observed in the other order."
                ),
            )
        )

    # Sort by support then confidence, both descending.
    suggestions.sort(key=lambda s: (-s.n_both, -s.confidence))
    return suggestions
