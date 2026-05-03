"""Policy adapter for `shadow diagnose-pr`.

Wraps `shadow.hierarchical.{load_policy, policy_diff}` into a small
typed surface — `evaluate_policy(path, baseline, candidate) ->
PolicyResult` — so the CLI doesn't depend on hierarchical's full
public surface (which is large).

Naming note: the underlying module is `shadow.hierarchical` rather
than `shadow.policy` for historical reasons; renaming the
underlying module is tracked as a follow-up (design spec §11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from shadow.errors import ShadowConfigError


@dataclass(frozen=True)
class PolicyResult:
    """Outcome of evaluating a policy overlay against a baseline +
    candidate trace pair.

    `regressions` is a list of `PolicyViolation.to_dict()` dicts —
    typed via dict[str, Any] so the result is JSON-serialisable
    without further conversion.
    """

    new_violations: int
    worst_rule: str | None
    regressions: list[dict[str, Any]] = field(default_factory=list)
    fixes: list[dict[str, Any]] = field(default_factory=list)


_SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


def evaluate_policy(
    policy_path: Path | None,
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> PolicyResult:
    """Apply a policy overlay (if provided) and return the
    candidate's regressions vs. the baseline.

    A `None` policy_path short-circuits to an empty result so the
    caller can always invoke this and get a uniform shape.
    """
    if policy_path is None:
        return PolicyResult(new_violations=0, worst_rule=None)

    if not policy_path.is_file():
        raise ShadowConfigError(f"policy file not found: {policy_path}")

    try:
        text = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ShadowConfigError(f"could not read {policy_path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ShadowConfigError(f"could not parse {policy_path}: {exc}") from exc

    from shadow.hierarchical import load_policy, policy_diff

    rules = load_policy(data)
    diff = policy_diff(baseline_records, candidate_records, rules)

    regressions = [v.to_dict() for v in diff.regressions]
    fixes = [v.to_dict() for v in diff.fixes]
    worst_rule = _pick_worst_rule(diff.regressions)

    return PolicyResult(
        new_violations=len(regressions),
        worst_rule=worst_rule,
        regressions=regressions,
        fixes=fixes,
    )


def _pick_worst_rule(violations: list[Any]) -> str | None:
    """Pick the highest-severity violated rule id; ties break on
    insertion order."""
    if not violations:
        return None
    best = violations[0]
    for v in violations[1:]:
        if _SEVERITY_ORDER.get(v.severity, 0) > _SEVERITY_ORDER.get(best.severity, 0):
            best = v
    return str(best.rule_id)


__all__ = ["PolicyResult", "evaluate_policy"]
