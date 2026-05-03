"""Verdict logic and dangerous-tool detection for `shadow diagnose-pr`.

Implements the spec §3.5 verdict mapping:

  ship  = no affected traces and no new policy violations
  probe = affected traces exist but no severe axis and no dangerous violation
          (v1: same as hold, distinguished in Week 3 by CI excluding zero)
  hold  = affected traces with no dangerous violation and no severe axis
  stop  = severe axis regression OR dangerous policy violation

v1 dangerous-tool detection: severity ∈ {error, critical} AND
  (rule has `tags: [dangerous]` OR rule's tool name matches a
   keyword in `_DANGEROUS_KEYWORDS`).

The keyword list is intentionally conservative — false positives
push to STOP which is recoverable; false negatives let dangerous
operations slip past which is not. Tags are the preferred path;
keywords are a stopgap until rule authors adopt tags.
"""

from __future__ import annotations

from typing import Any

from shadow.diagnose_pr.models import Verdict

_DANGEROUS_KEYWORDS = frozenset(
    {
        "refund",
        "pay",
        "transfer",
        "wire",
        "delete",
        "drop",
        "shutdown",
        "revoke",
        "grant",
        "escalate",
        "issue_refund",
    }
)
_DANGEROUS_SEVERITIES = frozenset({"error", "critical"})


def classify_verdict(
    *,
    affected: int,
    total: int,
    has_dangerous_violation: bool,
    has_severe_axis: bool,
) -> Verdict:
    """Map the three classification signals to a v0.1 verdict.

    Order matters: `stop` short-circuits everything; `ship` requires
    no affected AND no dangerous violation; everything else is
    `hold` in v1 (Week 3 will distinguish probe vs. hold by causal
    CI excluding zero).

    `total` is currently unused in v1 — kept in the signature so
    Week 3's blast-radius-aware verdict tuning has a place to land
    without a signature change.
    """
    del total  # reserved for v2 blast-radius tuning
    if has_dangerous_violation or has_severe_axis:
        return "stop"
    if affected == 0:
        return "ship"
    return "hold"


def is_dangerous_violation(rule: dict[str, Any]) -> bool:
    """v1 dangerous-tool detection. See module docstring for the
    rationale; test_diagnose_pr_verdict.py pins each branch."""
    severity = str(rule.get("severity", "")).lower()
    if severity not in _DANGEROUS_SEVERITIES:
        return False
    tags = rule.get("tags") or []
    if isinstance(tags, list) and "dangerous" in tags:
        return True
    params = rule.get("params") or {}
    candidates = (
        str(params.get("tool", "")),
        str(params.get("then", "")),
        str(params.get("first", "")),
    )
    for name in candidates:
        if not name:
            continue
        lower = name.lower()
        for kw in _DANGEROUS_KEYWORDS:
            if kw in lower:
                return True
    return False


__all__ = ["classify_verdict", "is_dangerous_violation"]
