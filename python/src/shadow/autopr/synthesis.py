"""Deterministic synthesizer: regression evidence -> Shadow policy YAML.

The function ``synthesize_policy`` walks the structured fields of a
``DiffReport`` (the output of ``shadow._core.compute_diff_report``) and
emits a list of policy rule dicts in Shadow's existing 12-kind policy
language. Every emitted rule is guaranteed to:

    1. Round-trip through :func:`shadow.hierarchical.load_policy` without
       error (the synthesizer never invents new ``kind`` values).
    2. Carry a stable ``id`` derived from rule contents so re-running the
       synthesizer on the same evidence is idempotent.
    3. Embed the statistical evidence that triggered it in the
       ``description`` field so PR reviewers can see the why.

What this module **does not** do:

    * No LLM calls. No network. Pure functions of the inputs.
    * No new policy kinds. Only the 9 currently-validated kinds with
      simple required-param contracts (the v2.0 grounding rules and the
      ``ltl_formula`` raw-LTL kind have richer contracts that don't lend
      themselves to mechanical synthesis from diff signals).
    * No claim that the generated policy *exhaustively* catches the
      regression. The synthesizer ships rules it can justify from the
      evidence; ungrounded inferences are dropped.

The companion :func:`verify_policy` runs the generated policy through
:func:`shadow.hierarchical.policy_diff` and reports whether (a) the
baseline produces zero violations and (b) the candidate produces at
least one violation. Rules that fail this counterfactual gate should
be discarded by the caller.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

# Re-import inside functions to keep the module import-light at startup.
# The Rust extension and the policy module are both available everywhere
# Shadow runs, so imports never fail in practice.

# Diff signals we know how to translate. Listed by the *axis* + the
# explanation pattern that uniquely identifies the regression class.
# Keeping the table here makes it easy to see at a glance which slice of
# divergence space the synthesizer is willing to commit to.
_DROPPED_TOOL_RE = re.compile(r"dropped tool call\(s\)?:\s*`([^`(]+)(?:\([^`]*\))?`")
_ADDED_TOOL_RE = re.compile(r"added tool call\(s\)?:\s*`([^`(]+)(?:\([^`]*\))?`")
_INSERTED_TOOL_RE = re.compile(r"inserted an extra `([^`(]+)(?:\([^`]*\))?`")
_DUPLICATE_TOOL_RE = re.compile(r"called `([^`(]+)(?:\([^`]*\))?`\s+\d+\s+time")
_STOP_REASON_RE = re.compile(r"stop_reason changed:\s*`([^`]+)`\s*[→>-]+\s*`([^`]+)`")
# `tool set changed: removed `X(args)`, added `Y(args)`` — fires for both
# tool swaps (X != Y) and schema renames (X == Y). The synthesiser only
# pins the swap case; schema renames go through verify_policy() and get
# dropped because forbidding the renamed tool also rejects baseline.
_TOOL_SET_CHANGED_RE = re.compile(
    r"tool set changed:\s*removed\s+`([^`(]+)(?:\([^`]*\))?`,?\s*added\s+`([^`(]+)(?:\([^`]*\))?`"
)


@dataclass
class SynthesizedPolicy:
    """A policy synthesized from a regression, plus diagnostic notes.

    ``rules`` is a list of dicts in the wire format that
    :func:`shadow.hierarchical.load_policy` accepts (i.e. the same shape
    you'd write by hand in YAML). ``diagnostics`` records every decision
    the synthesizer made so reviewers can audit why a particular rule
    was or was not generated.
    """

    rules: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    api_version: str = "shadow.dev/v1alpha1"

    def to_dict(self) -> dict[str, Any]:
        """Wire format consumed by :func:`load_policy` and YAML dumpers."""
        return {"apiVersion": self.api_version, "rules": list(self.rules)}

    def to_yaml(self) -> str:
        """Stable YAML for committing to a repo."""
        # Local import: pyyaml is already a Shadow runtime dep, but
        # importing it lazily keeps the public surface side-effect-free
        # at module import time.
        import yaml

        return yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,
            default_flow_style=False,
        )


def synthesize_policy(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    name: str = "regression",
    seed: int = 42,
) -> SynthesizedPolicy:
    """Synthesize a policy that catches the regression in ``candidate`` vs
    ``baseline``.

    Parameters
    ----------
    baseline_records, candidate_records:
        Lists of ``.agentlog`` records as returned by
        ``shadow._core.parse_agentlog`` (or written by
        ``shadow.sdk.Session``). The synthesizer inspects only the diff
        report computed from these; the raw record list is also passed
        through to verification.
    name:
        Slug used as a prefix for rule ids. Use a short stable string —
        e.g. ``"refund-flow-2026-04-29"``.
    seed:
        Bootstrap seed for reproducibility of the underlying diff. Doesn't
        affect rule generation (the synthesizer never samples), but is
        forwarded to the Rust differ so reports are identical between
        runs.

    Returns
    -------
    SynthesizedPolicy
        The generated rules plus diagnostic notes. May be empty (no
        rules) if the diff didn't surface any actionable regression
        signals — that's a feature, not a bug; pinning a "regression"
        that the synthesizer can't classify would risk locking in
        irrelevant noise.
    """
    from shadow import _core

    report = _core.compute_diff_report(baseline_records, candidate_records, None, seed)

    out = SynthesizedPolicy()
    seen_ids: set[str] = set()

    # ---- 1. Per-divergence rules ------------------------------------
    # Walk the structured divergence list. Each FirstDivergence carries
    # a `kind` (style/decision/structural), a `primary_axis`, and a
    # human-readable `explanation` whose phrasing is stable enough to
    # parse with simple regexes (Rust side at crates/shadow-core/src/
    # diff/alignment.rs constructs them with the same templates).
    for dv in report.get("divergences") or []:
        rule = _rule_for_divergence(dv, name=name)
        if rule is None:
            continue
        if rule["id"] in seen_ids:
            out.diagnostics.append(
                f"skipped duplicate rule {rule['id']!r} from divergence "
                f"on axis {dv.get('primary_axis')!r}"
            )
            continue
        seen_ids.add(rule["id"])
        out.rules.append(rule)

    # ---- 2. Axis-level rules (whole-trace signals) ------------------
    # Some regressions don't surface as a per-turn divergence — a
    # severe cost or verbosity shift across the whole trace, for
    # example. These get whole-trace rules.
    for row in report.get("rows") or []:
        rule = _rule_for_axis(row, name=name, baseline_records=baseline_records)
        if rule is None:
            continue
        if rule["id"] in seen_ids:
            continue
        seen_ids.add(rule["id"])
        out.rules.append(rule)

    if not out.rules:
        out.diagnostics.append(
            "no rules synthesised — the diff didn't surface a "
            "regression in a class the synthesizer can pin "
            "(structural drift, refusal flip, axis-level cost/token spike)."
        )
    return out


def verify_policy(
    policy: SynthesizedPolicy,
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Counterfactual gate: does the policy actually catch the regression?

    Loads the generated policy and runs it against both traces:

    * Baseline must produce **zero** violations. If it doesn't, the
      synthesizer over-fit and the rule rejects baseline behaviour the
      candidate also has — that's the oracle-locking failure mode the
      research community has been calling out (see STING, AgentAssay).
    * Candidate must produce **at least one** violation. If it doesn't,
      the rule is misfit to the regression — drop it.

    Returns
    -------
    (verified, reasons):
        ``verified`` is True iff both checks pass. ``reasons`` is a list
        of human-readable lines explaining the outcome — included even
        on success so callers can audit what was checked.
    """
    from shadow.hierarchical import load_policy, policy_diff

    if not policy.rules:
        return False, ["policy is empty — nothing to verify"]

    # load_policy raises on malformed wire format; wrap so callers get a
    # graceful False rather than an exception.
    try:
        rules = load_policy(policy.to_dict())
    except Exception as e:
        return False, [f"policy failed to parse via load_policy(): {e}"]

    diff = policy_diff(baseline_records, candidate_records, rules)

    reasons: list[str] = []
    ok = True

    if diff.baseline_violations:
        ok = False
        reasons.append(
            f"baseline shows {len(diff.baseline_violations)} violation(s) — "
            f"the rule is over-fit and rejects good behaviour: "
            f"{[v.rule_id for v in diff.baseline_violations[:3]]}"
        )
    else:
        reasons.append(
            "baseline shows 0 violations (good — rule doesn't reject baseline behaviour)"
        )

    if not diff.candidate_violations:
        ok = False
        reasons.append(
            "candidate shows 0 violations — the synthesised rule "
            "doesn't actually catch the regression in the candidate trace."
        )
    else:
        reasons.append(
            f"candidate shows {len(diff.candidate_violations)} violation(s) "
            "(good — rule fires on the regression as intended)"
        )

    return ok, reasons


# ---------------------------------------------------------------------------
# Internal helpers — divergence & axis -> rule
# ---------------------------------------------------------------------------


def _stable_id(name: str, kind: str, params: dict[str, Any]) -> str:
    """Stable, idempotent rule id from (name, kind, params).

    Re-running the synthesizer on the same evidence must produce the
    same id so downstream tooling (PR opener, dedup) can recognise an
    already-shipped rule.
    """
    canonical = json.dumps({"k": kind, "p": params}, sort_keys=True)
    short = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower() or "policy"
    return f"{safe_name}-{kind.replace('_', '-')}-{short}"


def _rule_for_divergence(dv: dict[str, Any], *, name: str) -> dict[str, Any] | None:
    """Translate one ``FirstDivergence`` into a policy rule, if we can.

    Returns ``None`` for divergences whose shape we deliberately don't
    pin (style drift below the noise floor, structural patterns that
    don't map cleanly to one of the 9 simple-contract kinds, etc.).
    Dropping is the safe behaviour: the synthesizer should never emit a
    rule it can't justify.
    """
    explanation = str(dv.get("explanation", ""))
    kind_class = str(dv.get("kind", ""))
    confidence = float(dv.get("confidence", 0.0))

    # Threshold tuning: real-world divergences on small fixtures land
    # in the 0.3-0.4 range even when the regression is structurally
    # clear. Setting the floor at 0.3 keeps obvious noise out (random
    # near-zero confidences) while admitting genuine signals on small
    # traces. The verify_policy() counterfactual gate is the second
    # net underneath: any rule the synthesiser ships still has to pin
    # the regression in practice, otherwise the caller drops it.
    if confidence < 0.3:
        return None

    # ---- Structural drift -----------------------------------------
    if kind_class == "structural_drift":
        # Pattern A: candidate dropped a tool. The baseline relied on
        # this tool being called; emit `must_call_once` to require it.
        m = _DROPPED_TOOL_RE.search(explanation)
        if m:
            tool = m.group(1).strip()
            params = {"tool": tool}
            kind = "must_call_once"
            return {
                "id": _stable_id(name, kind, params),
                "kind": kind,
                "params": params,
                "severity": "error",
                "description": (
                    f"Auto-synthesised from regression: candidate dropped tool "
                    f"call to `{tool}`. Pinning this rule requires the "
                    f"candidate to invoke `{tool}` at least once. "
                    f"(divergence confidence {confidence:.2f}, axis "
                    f"{dv.get('primary_axis', '?')})"
                ),
            }

        # Pattern B: candidate added a tool that the baseline never
        # called. Forbid that tool — the canonical "lock the bad
        # behaviour out" rule.
        m = _ADDED_TOOL_RE.search(explanation) or _INSERTED_TOOL_RE.search(explanation)
        if m:
            tool = m.group(1).strip()
            params = {"tool": tool}
            kind = "no_call"
            return {
                "id": _stable_id(name, kind, params),
                "kind": kind,
                "params": params,
                "severity": "error",
                "description": (
                    f"Auto-synthesised from regression: candidate began "
                    f"calling `{tool}` which the baseline never used. "
                    f"Pinning this rule forbids `{tool}` for the same "
                    f"input class. (divergence confidence {confidence:.2f}, "
                    f"axis {dv.get('primary_axis', '?')})"
                ),
            }

        # Pattern C: duplicate tool call. The baseline called it once;
        # the candidate called it more than once. Pin to "exactly once."
        m = _DUPLICATE_TOOL_RE.search(explanation)
        if m:
            tool = m.group(1).strip()
            params = {"tool": tool}
            kind = "must_call_once"
            return {
                "id": _stable_id(name, kind, params),
                "kind": kind,
                "params": params,
                "severity": "error",
                "description": (
                    f"Auto-synthesised from regression: candidate called "
                    f"`{tool}` more than once where the baseline called "
                    f"it once. Pinning this rule requires `{tool}` to be "
                    f"invoked at-most-once per session. (divergence "
                    f"confidence {confidence:.2f})"
                ),
            }

        # Pattern D: tool set changed (one tool swapped for another).
        # When the names differ this is a real swap and we forbid the
        # new tool. When they match (schema rename: same name, different
        # args) we deliberately don't pin — verify_policy() would drop
        # any rule that also rejects baseline behaviour, and forbidding
        # the renamed tool falls into that trap. Skip and let the
        # caller see "no rules" via diagnostics.
        m = _TOOL_SET_CHANGED_RE.search(explanation)
        if m:
            removed = m.group(1).strip()
            added = m.group(2).strip()
            if removed != added:
                params = {"tool": added}
                kind = "no_call"
                return {
                    "id": _stable_id(name, kind, params),
                    "kind": kind,
                    "params": params,
                    "severity": "error",
                    "description": (
                        f"Auto-synthesised from regression: tool set "
                        f"changed — candidate began calling `{added}` "
                        f"instead of baseline's `{removed}`. Pinning "
                        f"forbids `{added}` for the same input class. "
                        f"(divergence confidence {confidence:.2f}, axis "
                        f"{dv.get('primary_axis', '?')})"
                    ),
                }
            # Same-name schema rename: nothing safe to pin without
            # arg-level inspection, which is out of scope for the
            # MVP synthesiser.
            return None

    # ---- Decision drift on safety: stop_reason flipped -----------
    if kind_class == "decision_drift" and dv.get("primary_axis") == "safety":
        m = _STOP_REASON_RE.search(explanation)
        if m:
            baseline_reason = m.group(1).strip()
            candidate_reason = m.group(2).strip()
            # Restrict allowed reasons to the baseline's. If the candidate
            # introduced `content_filter` or `policy_blocked`, this rule
            # rejects the regression on the next PR.
            params = {"allowed": [baseline_reason]}
            kind = "required_stop_reason"
            return {
                "id": _stable_id(name, kind, params),
                "kind": kind,
                "params": params,
                "severity": "error",
                "description": (
                    f"Auto-synthesised from regression: stop_reason flipped "
                    f"from `{baseline_reason}` to `{candidate_reason}`. "
                    f"Pinning this rule requires every chat response to "
                    f"end with `{baseline_reason}`. (divergence confidence "
                    f"{confidence:.2f})"
                ),
            }

    # No matching pattern. Don't invent a rule.
    return None


# ---------------------------------------------------------------------------
# Axis-level signals
# ---------------------------------------------------------------------------


_TOTAL_TOKEN_HEADROOM = 1.20  # 20% over baseline median total tokens


def _rule_for_axis(
    row: dict[str, Any],
    *,
    name: str,
    baseline_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Whole-trace rule for severe axis-level shifts.

    Only fires for ``severe`` axes with a positive delta direction we
    can pin to a hard limit. Negative deltas (improvement) and minor /
    moderate severities are not pinned: the synthesiser is conservative
    by design.
    """
    severity = str(row.get("severity", "none"))
    axis = str(row.get("axis", ""))
    delta = float(row.get("delta", 0.0))
    if severity != "severe":
        return None

    # ---- max_total_tokens: cost / verbosity severe + positive ------
    if axis in {"cost", "verbosity"} and delta > 0:
        baseline_total = _baseline_total_tokens(baseline_records)
        if baseline_total <= 0:
            return None
        limit = int(baseline_total * _TOTAL_TOKEN_HEADROOM)
        params = {"limit": limit}
        kind = "max_total_tokens"
        return {
            "id": _stable_id(name, kind, params),
            "kind": kind,
            "params": params,
            "severity": "error",
            "description": (
                f"Auto-synthesised from regression: {axis} axis shifted "
                f"severely (delta {delta:+.3f}). Pinning total token "
                f"budget at {limit} (baseline observed total "
                f"{baseline_total} + 20% headroom)."
            ),
        }
    return None


def _baseline_total_tokens(records: list[dict[str, Any]]) -> int:
    """Sum input + output + thinking tokens across baseline responses.

    Mirrors the existing token-extraction in ``shadow.hierarchical``,
    kept here as a tiny helper so the synthesizer doesn't import from
    that module's internals.
    """
    total = 0
    for r in records:
        if r.get("kind") != "chat_response":
            continue
        usage = (r.get("payload") or {}).get("usage") or {}
        try:
            total += int(usage.get("input_tokens", 0) or 0)
            total += int(usage.get("output_tokens", 0) or 0)
            total += int(usage.get("thinking_tokens", 0) or 0)
        except (TypeError, ValueError):
            # Malformed usage block — skip silently. The token-budget
            # rule will simply not fire if no recorded usage exists.
            continue
    return total
