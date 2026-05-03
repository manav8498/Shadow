# Phase 1 Week 2: `shadow diagnose-pr` 9-axis classification + policy + verdict

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire real per-trace 9-axis behavioral diff + policy violation detection + ship/probe/hold/stop verdict logic into `shadow diagnose-pr`. After this week, a candidate that breaks a `must_call_before` policy returns HOLD/STOP and the PR comment names the violated rule.

**Architecture:** Add three focused modules — `diffing`, `policy`, `risk` — each composing existing internals (`shadow._core.compute_diff_report`, `shadow.hierarchical.{load_policy, check_policy, policy_diff}`). The CLI gets a new `--candidate-traces` flag for paired-trace diffing (replay-driven candidate generation lands in Week 3). The v0.1 schema is unchanged — Week 2 fills in fields that were `None`/`0`/`[]` in the skeleton.

**Tech Stack:** Pure Python (no Rust changes). PyO3 binding `_core.compute_diff_report` already exists; `shadow.hierarchical` already exists.

---

## File structure

### Created

| File | Responsibility |
|---|---|
| `python/src/shadow/diagnose_pr/diffing.py` | `diff_pair`, `is_affected`, `worst_axis_for` — wraps the 9-axis Rust differ for per-trace use |
| `python/src/shadow/diagnose_pr/policy.py` | `evaluate_policy` — adapter to `shadow.hierarchical.policy_diff` |
| `python/src/shadow/diagnose_pr/risk.py` | `classify_verdict`, `is_dangerous_violation` — verdict mapping |
| `python/tests/test_diagnose_pr_diffing.py` | per-pair diff + affected classifier |
| `python/tests/test_diagnose_pr_policy.py` | policy diff adapter |
| `python/tests/test_diagnose_pr_verdict.py` | verdict matrix |
| `python/tests/test_diagnose_pr_e2e_hold.py` | end-to-end: known policy violation → HOLD verdict |

### Modified

| File | Change |
|---|---|
| `python/src/shadow/diagnose_pr/report.py` | `build_report` accepts a list of `TraceDiagnosis` directly (no longer just an "affected_trace_ids" set), routes through `risk.classify_verdict` |
| `python/src/shadow/diagnose_pr/models.py` | No schema changes; just pin `flags` to also support `"high_blast_radius"` for >50% verdicts |
| `python/src/shadow/cli/app.py` | Add `--candidate-traces`, `--policy`; wire diffing/policy/risk into the per-trace loop |

---

## Task 1: `diffing.py` — per-trace 9-axis diff wrapper

**Files:**
- Create: `python/src/shadow/diagnose_pr/diffing.py`
- Create: `python/tests/test_diagnose_pr_diffing.py`

- [ ] **1.1: Failing tests**

```python
"""Tests for shadow.diagnose_pr.diffing — wraps the 9-axis Rust
differ for per-trace use inside diagnose-pr."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest


def _quickstart_pair() -> tuple[list[dict], list[dict]]:
    import shadow.quickstart_data as q
    from shadow import _core

    root = resources.files(q) / "fixtures"
    base = _core.parse_agentlog(root.joinpath("baseline.agentlog").read_bytes())
    cand = _core.parse_agentlog(root.joinpath("candidate.agentlog").read_bytes())
    return base, cand


def test_diff_pair_returns_dict_with_rows_and_severity_per_axis() -> None:
    from shadow.diagnose_pr.diffing import diff_pair

    base, cand = _quickstart_pair()
    report = diff_pair(base, cand)
    assert "rows" in report
    assert isinstance(report["rows"], list)
    severities = {row["severity"] for row in report["rows"]}
    assert severities <= {"none", "minor", "moderate", "severe"}


def test_is_affected_true_when_any_axis_severity_moderate_or_above() -> None:
    from shadow.diagnose_pr.diffing import is_affected

    rep = {"rows": [{"axis": "trajectory", "severity": "moderate"}]}
    assert is_affected(rep) is True
    rep = {"rows": [{"axis": "trajectory", "severity": "severe"}]}
    assert is_affected(rep) is True


def test_is_affected_false_when_all_axes_minor_or_none() -> None:
    from shadow.diagnose_pr.diffing import is_affected

    rep = {
        "rows": [
            {"axis": "trajectory", "severity": "none"},
            {"axis": "verbosity", "severity": "minor"},
        ]
    }
    assert is_affected(rep) is False


def test_is_affected_true_when_first_divergence_present_and_nontrivial() -> None:
    """Even a 'minor' on every axis is affected if the report includes
    a first_divergence — that's a flag from the differ that something
    structurally interesting happened."""
    from shadow.diagnose_pr.diffing import is_affected

    rep = {
        "rows": [{"axis": "trajectory", "severity": "minor"}],
        "first_divergence": {"pair_index": 1, "axis": "trajectory", "detail": "x"},
    }
    assert is_affected(rep) is True


def test_worst_axis_returns_axis_with_highest_severity() -> None:
    from shadow.diagnose_pr.diffing import worst_axis_for

    rep = {
        "rows": [
            {"axis": "verbosity", "severity": "minor"},
            {"axis": "trajectory", "severity": "severe"},
            {"axis": "safety", "severity": "moderate"},
        ]
    }
    assert worst_axis_for(rep) == "trajectory"


def test_worst_axis_returns_none_when_all_axes_are_none() -> None:
    from shadow.diagnose_pr.diffing import worst_axis_for

    rep = {"rows": [{"axis": "trajectory", "severity": "none"}]}
    assert worst_axis_for(rep) is None


def test_diff_pair_on_real_demo_fixtures_returns_nine_axes() -> None:
    from shadow.diagnose_pr.diffing import diff_pair

    base, cand = _quickstart_pair()
    rep = diff_pair(base, cand)
    axes = {row["axis"] for row in rep["rows"]}
    expected = {
        "semantic", "trajectory", "safety", "verbosity",
        "latency", "cost", "reasoning", "judge", "conformance",
    }
    # Judge may be missing if no judge configured; allow that.
    assert (axes & expected) >= (expected - {"judge"})
```

- [ ] **1.2: Run tests, see ImportError**

```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_diffing.py -q
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **1.3: Implement diffing**

```python
"""Per-trace 9-axis diff for `shadow diagnose-pr`.

Wraps `shadow._core.compute_diff_report` (the Rust 9-axis differ)
for one-pair use, plus two classifiers — `is_affected` and
`worst_axis_for` — that read the report's `severity` and
`first_divergence` fields.

The classifier thresholds live here, not in the Rust differ, so
they're easy to tune without recompiling.
"""

from __future__ import annotations

from typing import Any

from shadow import _core

_AFFECTED_SEVERITIES = {"moderate", "severe"}
_SEVERITY_ORDER = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def diff_pair(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    pricing: dict[str, tuple[float, float]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute a 9-axis diff report for one (baseline, candidate)
    record-list pair. Thin wrapper over the Rust differ so the
    diagnose-pr surface doesn't depend on `_core` directly."""
    return _core.compute_diff_report(baseline, candidate, pricing, seed)


def is_affected(report: dict[str, Any]) -> bool:
    """Classify whether a candidate trace meaningfully changed
    behavior relative to its baseline.

    Affected if either:
      * any axis has severity moderate/severe, OR
      * the differ flagged a `first_divergence` (a structural pin
        that something interesting happened, even if the per-axis
        severities are still minor).
    """
    rows = report.get("rows") or []
    for row in rows:
        if row.get("severity") in _AFFECTED_SEVERITIES:
            return True
    fd = report.get("first_divergence")
    if fd and isinstance(fd, dict):
        return True
    return False


def worst_axis_for(report: dict[str, Any]) -> str | None:
    """Return the axis name with the highest severity in this report,
    or None if all axes are at severity `none`. Ties are broken by
    insertion order (the differ already orders axes deterministically)."""
    rows = report.get("rows") or []
    best_rank = 0
    best_axis: str | None = None
    for row in rows:
        sev = row.get("severity", "none")
        rank = _SEVERITY_ORDER.get(sev, 0)
        if rank > best_rank:
            best_rank = rank
            best_axis = row.get("axis")
    return best_axis


__all__ = ["diff_pair", "is_affected", "worst_axis_for"]
```

- [ ] **1.4: Run tests, see green**

```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_diffing.py -q
```
Expected: `7 passed`.

- [ ] **1.5: Commit**

```
feat(diagnose-pr): per-trace 9-axis diff wrapper

diff_pair(baseline, candidate) wraps the Rust differ for
diagnose-pr's per-trace loop. is_affected(report) classifies
moderate/severe severity OR non-null first_divergence as affected.
worst_axis_for(report) ranks axes by severity. Thresholds live
here so they're easy to tune without a Rust rebuild.
```

---

## Task 2: `policy.py` — adapter for shadow.hierarchical

**Files:**
- Create: `python/src/shadow/diagnose_pr/policy.py`
- Create: `python/tests/test_diagnose_pr_policy.py`

- [ ] **2.1: Failing tests**

```python
"""Tests for shadow.diagnose_pr.policy — thin adapter to
shadow.hierarchical for diagnose-pr's per-pair use."""

from __future__ import annotations

from pathlib import Path

import pytest


_SAMPLE_POLICY = """
apiVersion: shadow.dev/v1alpha1
rules:
  - id: confirm-before-refund
    kind: must_call_before
    params:
      first: confirm_refund_amount
      then: issue_refund
    severity: error
"""


def _make_pair_with_violation(tmp_path: Path) -> tuple[list[dict], list[dict]]:
    """Build a baseline that confirms-before-refund and a candidate
    that doesn't. Used to assert the policy adapter detects the
    regression."""
    from shadow import _core
    from shadow.sdk import Session

    base_path = tmp_path / "baseline.agentlog"
    cand_path = tmp_path / "candidate.agentlog"

    # Baseline: confirm then refund (compliant)
    with Session(output_path=base_path, tags={}) as s:
        s.record_chat(
            request={"model": "x", "messages": [{"role": "user", "content": "refund"}], "params": {}},
            response={
                "model": "x",
                "content": [
                    {"type": "tool_use", "name": "confirm_refund_amount", "input": {}, "id": "1"}
                ],
                "stop_reason": "tool_use",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
        s.record_chat(
            request={"model": "x", "messages": [{"role": "user", "content": "ok"}], "params": {}},
            response={
                "model": "x",
                "content": [
                    {"type": "tool_use", "name": "issue_refund", "input": {}, "id": "2"}
                ],
                "stop_reason": "tool_use",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    # Candidate: refund directly (violation)
    with Session(output_path=cand_path, tags={}) as s:
        s.record_chat(
            request={"model": "x", "messages": [{"role": "user", "content": "refund"}], "params": {}},
            response={
                "model": "x",
                "content": [
                    {"type": "tool_use", "name": "issue_refund", "input": {}, "id": "1"}
                ],
                "stop_reason": "tool_use",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    return _core.parse_agentlog(base_path.read_bytes()), _core.parse_agentlog(cand_path.read_bytes())


def test_evaluate_policy_returns_zero_when_no_policy_path(tmp_path: Path) -> None:
    from shadow.diagnose_pr.policy import evaluate_policy

    base, cand = _make_pair_with_violation(tmp_path)
    res = evaluate_policy(None, base, cand)
    assert res.new_violations == 0
    assert res.worst_rule is None
    assert res.regressions == []


def test_evaluate_policy_detects_regression_from_yaml_file(tmp_path: Path) -> None:
    from shadow.diagnose_pr.policy import evaluate_policy

    p = tmp_path / "policy.yaml"
    p.write_text(_SAMPLE_POLICY)
    base, cand = _make_pair_with_violation(tmp_path)
    res = evaluate_policy(p, base, cand)
    assert res.new_violations >= 1
    assert res.worst_rule == "confirm-before-refund"
    assert any(v["rule_id"] == "confirm-before-refund" for v in res.regressions)


def test_evaluate_policy_compliant_candidate_has_no_regressions(tmp_path: Path) -> None:
    from shadow.diagnose_pr.policy import evaluate_policy

    p = tmp_path / "policy.yaml"
    p.write_text(_SAMPLE_POLICY)
    # Baseline already compliant, use it as both sides.
    base, _ = _make_pair_with_violation(tmp_path)
    res = evaluate_policy(p, base, base)
    assert res.new_violations == 0
    assert res.worst_rule is None
```

- [ ] **2.2: Run tests, see ImportError**

```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_policy.py -q
```

- [ ] **2.3: Implement policy**

```python
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
    return best.rule_id


__all__ = ["PolicyResult", "evaluate_policy"]
```

- [ ] **2.4: Run tests, see green**

- [ ] **2.5: Commit**

---

## Task 3: `risk.py` — verdict + dangerous-tool detection

**Files:**
- Create: `python/src/shadow/diagnose_pr/risk.py`
- Create: `python/tests/test_diagnose_pr_verdict.py`

- [ ] **3.1: Failing tests**

```python
"""Tests for shadow.diagnose_pr.risk — verdict logic + dangerous-tool
detection. These pin the spec §3.5 verdict mapping."""

from __future__ import annotations

from shadow.diagnose_pr.risk import (
    is_dangerous_violation,
    classify_verdict,
)


def test_zero_affected_is_ship() -> None:
    assert classify_verdict(
        affected=0, total=10, has_dangerous_violation=False, has_severe_axis=False
    ) == "ship"


def test_zero_affected_with_dangerous_violation_still_stop() -> None:
    """A dangerous policy violation alone is enough for STOP, even
    if the 9-axis diff hasn't (yet) classified any trace as
    affected. This catches the case where the violation is in the
    candidate but the differ found no other axes moving."""
    assert classify_verdict(
        affected=0, total=10, has_dangerous_violation=True, has_severe_axis=False
    ) == "stop"


def test_affected_traces_with_severe_axis_is_stop() -> None:
    assert classify_verdict(
        affected=5, total=10, has_dangerous_violation=False, has_severe_axis=True
    ) == "stop"


def test_affected_traces_no_severe_no_dangerous_is_hold() -> None:
    assert classify_verdict(
        affected=5, total=10, has_dangerous_violation=False, has_severe_axis=False
    ) == "hold"


def test_dangerous_violation_short_circuits_to_stop() -> None:
    assert classify_verdict(
        affected=5, total=10, has_dangerous_violation=True, has_severe_axis=False
    ) == "stop"


def test_dangerous_keyword_in_tool_name() -> None:
    """v1 keyword fallback: refund / issue_refund / wire_transfer
    are all flagged as dangerous when the rule is severity error."""
    rule = {"params": {"tool": "issue_refund"}, "severity": "error", "tags": []}
    assert is_dangerous_violation(rule) is True


def test_explicit_tags_dangerous_marks_dangerous() -> None:
    rule = {"params": {"tool": "harmless_tool"}, "severity": "error", "tags": ["dangerous"]}
    assert is_dangerous_violation(rule) is True


def test_low_severity_violation_not_dangerous_even_with_dangerous_tool() -> None:
    rule = {"params": {"tool": "issue_refund"}, "severity": "info", "tags": []}
    assert is_dangerous_violation(rule) is False


def test_must_call_before_uses_then_field_for_keyword_match() -> None:
    """In must_call_before rules, the dangerous tool is `then`, not
    `tool`. We have to look at both."""
    rule = {
        "kind": "must_call_before",
        "params": {"first": "confirm", "then": "issue_refund"},
        "severity": "error",
        "tags": [],
    }
    assert is_dangerous_violation(rule) is True
```

- [ ] **3.2: Run tests, see ImportError**

- [ ] **3.3: Implement risk**

```python
"""Verdict logic and dangerous-tool detection for `shadow diagnose-pr`.

Implements the spec §3.5 verdict mapping:

  ship  = no affected traces and no new policy violations
  probe = affected traces exist but no severe axis and no dangerous violation
  hold  = affected traces with no dangerous violation and no severe axis
          (v1: same as probe, distinguished in Week 3 by CI excluding zero)
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
        "refund", "pay", "transfer", "wire", "delete", "drop",
        "shutdown", "revoke", "grant", "escalate", "issue_refund",
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
    CI excluding zero)."""
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
```

- [ ] **3.4: Run tests, see green**

- [ ] **3.5: Commit**

---

## Task 4: Refactor `report.py` to consume real classifications

**Files:**
- Modify: `python/src/shadow/diagnose_pr/report.py`
- Modify: `python/tests/test_diagnose_pr_report.py`

The Week 1 `build_report` took an `affected_trace_ids: set[str]` and trivially branched ship/probe. Week 2 hands it a list of `TraceDiagnosis` directly + a `PolicyResult` + a `has_severe_axis` flag, and routes through `risk.classify_verdict`.

- [ ] **4.1: Update tests + signature**

Update `build_report` signature:

```python
def build_report(
    *,
    traces: list[LoadedTrace],
    deltas: list[ConfigDelta],
    diagnoses: list[TraceDiagnosis] | None = None,
    affected_trace_ids: set[str] | None = None,    # legacy Week-1 path
    new_policy_violations: int = 0,
    worst_policy_rule: str | None = None,
    has_dangerous_violation: bool = False,
    has_severe_axis: bool = False,
) -> DiagnosePrReport: ...
```

Both code paths must stay green: existing Week-1 callers passing `affected_trace_ids=` keep working.

- [ ] **4.2: Wire risk.classify_verdict + run all report tests**

- [ ] **4.3: Commit**

---

## Task 5: CLI wiring — `--candidate-traces`, `--policy`

**Files:**
- Modify: `python/src/shadow/cli/app.py`
- Modify: `python/tests/test_diagnose_pr_cli.py`

The Week 1 CLI took `--traces DIR` (baseline only) and skipped per-trace diff. Week 2 adds:

```
--candidate-traces DIR    # paired-by-filename candidate traces
--policy FILE             # YAML policy overlay
```

When `--candidate-traces` is given:
1. Load both baseline and candidate trace lists.
2. Pair by filename (e.g. `traces/a.agentlog` ↔ `cand_traces/a.agentlog`).
3. For each pair: run `diff_pair`, classify `is_affected`, apply policy, build `TraceDiagnosis`.
4. Pass to `build_report` with the real classifications.

When `--candidate-traces` is omitted (Week 1 path): keep the skeleton behavior — every trace stays unaffected, verdict is `ship`. Week 3 will add replay-driven candidate generation.

- [ ] **5.1: Update CLI tests with policy + candidate-traces**

- [ ] **5.2: Implement CLI wiring**

- [ ] **5.3: All CLI + e2e tests green**

- [ ] **5.4: Commit**

---

## Task 6: End-to-end HOLD/STOP scenario test

**Files:**
- Create: `python/tests/test_diagnose_pr_e2e_hold.py`

Synthesise the canonical refund-confirmation scenario from the
design spec. Confirm:

  * verdict is HOLD or STOP (any positive classification is valid;
    Week 3 will distinguish them via causal CI).
  * `worst_policy_rule == "confirm-before-refund"`.
  * `affected_traces > 0`.
  * PR comment names the rule.

- [ ] **6.1: Write the test**
- [ ] **6.2: Verify it passes**
- [ ] **6.3: Commit**

---

## Task 7: Lint + full suite gate

- [ ] **7.1: ruff check + format clean**
- [ ] **7.2: full pytest suite green (1741 + new tests)**

---

*End of Week 2 plan.*
