"""Pure classification logic for ``shadow heal``.

Takes a diff-report dict and produces a :class:`HealDecision` with a
tier, an ordered list of gate checks, and the evidence that drove each
check. No I/O, no Rich, no actions. Every output is a function of the
input report — re-running on the same dict is byte-identical.

The classifier is deliberately conservative: every gate is a one-way
refusal. The default tier is ``hold``; every upgrade to ``propose`` or
``heal`` requires the corresponding check to pass. A reviewer reading
the check list can audit *exactly* why the classifier landed where it
did, with no hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Below this pair count the diff is too thin to make any safety
#: claim. Mirrors the ``shadow call`` probe floor for consistency.
_MIN_PAIRS_FOR_HEAL = 5


class HealTier(str, Enum):
    """Three-tier classification for an auto-heal decision.

    Ordered from "no action" to "auto-acceptable" so callers can compare
    by :attr:`rank` when needed.
    """

    HOLD = "hold"
    PROPOSE = "propose"
    HEAL = "heal"

    @property
    def rank(self) -> int:
        return {HealTier.HOLD: 0, HealTier.PROPOSE: 1, HealTier.HEAL: 2}[self]


@dataclass(frozen=True)
class HealCheck:
    """One gate the classifier evaluated.

    ``passed`` is the boolean outcome; ``detail`` is a single-sentence
    human-readable explanation that surfaces in the rendered panel and
    in the ledger entry.
    """

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class HealDecision:
    """The full output of :func:`classify`.

    ``checks`` is ordered: hard-refusal gates first, then refinement
    gates that distinguish ``heal`` from ``propose``. Every check the
    classifier *evaluated* appears in the list; checks that weren't
    reached because an earlier gate refused are absent so the audit
    trail accurately reflects the decision path.
    """

    tier: HealTier
    anchor_id: str
    candidate_id: str
    pair_count: int
    checks: list[HealCheck] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "anchor_id": self.anchor_id,
            "candidate_id": self.candidate_id,
            "pair_count": self.pair_count,
            "checks": [c.to_dict() for c in self.checks],
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def classify(report: dict[str, Any]) -> HealDecision:
    """Classify a diff report into a heal tier.

    The order of gates matters and is documented in :mod:`shadow.heal`:

        1. ``pair_count >= 5`` — below the probe floor, no claim is safe
        2. No ``structural_drift`` divergences
        3. No ``decision_drift`` on the safety axis (refusal flips)
        4. No severe axis whose CI excludes zero (real signal above noise)
        5. Refinement: only ``style_drift`` or no divergences ⇒ ``heal``;
           otherwise ⇒ ``propose``

    Returns a :class:`HealDecision` whose ``checks`` list documents the
    decision path in order. The function is pure and re-running on the
    same input produces byte-identical output.
    """
    pair_count = int(report.get("pair_count", 0) or 0)
    rows = list(report.get("rows") or [])
    divergences = list(report.get("divergences") or [])
    anchor_id = _short_id(report.get("baseline_trace_id", ""))
    candidate_id = _short_id(report.get("candidate_trace_id", ""))

    checks: list[HealCheck] = []

    # ---- Gate 1: pair-count floor --------------------------------
    if pair_count < _MIN_PAIRS_FOR_HEAL:
        checks.append(
            HealCheck(
                name="pair_count_floor",
                passed=False,
                detail=(
                    f"only {pair_count} response pair(s) — heal requires at "
                    f"least {_MIN_PAIRS_FOR_HEAL} for a safe call"
                ),
            )
        )
        return HealDecision(
            tier=HealTier.HOLD,
            anchor_id=anchor_id,
            candidate_id=candidate_id,
            pair_count=pair_count,
            checks=checks,
            rationale=(
                f"Below the probe floor ({pair_count} < {_MIN_PAIRS_FOR_HEAL}). "
                "Record more pairs before any heal decision is meaningful."
            ),
        )
    checks.append(
        HealCheck(
            name="pair_count_floor",
            passed=True,
            detail=f"{pair_count} response pair(s) — above the heal floor",
        )
    )

    # ---- Gate 2: no structural drift ----------------------------
    structural = [d for d in divergences if d.get("kind") == "structural_drift"]
    if structural:
        first = structural[0]
        checks.append(
            HealCheck(
                name="no_structural_drift",
                passed=False,
                detail=(
                    f"structural drift at turn {first.get('baseline_turn', '?')}: "
                    f"{first.get('explanation', '?')}"
                ),
            )
        )
        return HealDecision(
            tier=HealTier.HOLD,
            anchor_id=anchor_id,
            candidate_id=candidate_id,
            pair_count=pair_count,
            checks=checks,
            rationale=(
                "Structural drift means the tool sequence changed — that is "
                "behaviour, not implementation. Heal cannot adapt to a "
                "behaviour change; this needs human review."
            ),
        )
    checks.append(
        HealCheck(
            name="no_structural_drift",
            passed=True,
            detail="no structural divergences detected",
        )
    )

    # ---- Gate 3: no decision drift on safety axis ----------------
    safety_decision = [
        d
        for d in divergences
        if d.get("kind") == "decision_drift" and d.get("primary_axis") == "safety"
    ]
    if safety_decision:
        first = safety_decision[0]
        checks.append(
            HealCheck(
                name="no_safety_decision_drift",
                passed=False,
                detail=(
                    f"safety decision drift at turn {first.get('baseline_turn', '?')}: "
                    f"{first.get('explanation', '?')}"
                ),
            )
        )
        return HealDecision(
            tier=HealTier.HOLD,
            anchor_id=anchor_id,
            candidate_id=candidate_id,
            pair_count=pair_count,
            checks=checks,
            rationale=(
                "Safety-axis decision drift (refusal flip, stop_reason change) is "
                "behavioural and never auto-healable. Surface to a reviewer."
            ),
        )
    checks.append(
        HealCheck(
            name="no_safety_decision_drift",
            passed=True,
            detail="no safety-axis decision drift",
        )
    )

    # ---- Gate 4: no severe axis whose CI excludes zero -----------
    severe_with_signal = []
    for r in rows:
        if r.get("severity") != "severe":
            continue
        ci_low = float(r.get("ci95_low", 0.0) or 0.0)
        ci_high = float(r.get("ci95_high", 0.0) or 0.0)
        # CI excludes zero when both endpoints have the same sign.
        if (ci_low > 0.0 and ci_high > 0.0) or (ci_low < 0.0 and ci_high < 0.0):
            severe_with_signal.append((str(r.get("axis", "?")), ci_low, ci_high))

    if severe_with_signal:
        names = ", ".join(name for name, _, _ in severe_with_signal[:3])
        checks.append(
            HealCheck(
                name="no_severe_with_signal",
                passed=False,
                detail=(
                    f"severe axes with CI excluding zero: {names}"
                    + (
                        f" (+{len(severe_with_signal) - 3} more)"
                        if len(severe_with_signal) > 3
                        else ""
                    )
                ),
            )
        )
        return HealDecision(
            tier=HealTier.HOLD,
            anchor_id=anchor_id,
            candidate_id=candidate_id,
            pair_count=pair_count,
            checks=checks,
            rationale=(
                "At least one axis is severe with a confidence interval that "
                "excludes zero — the regression is statistically real. Auto-heal "
                "would mask a real bug; refused."
            ),
        )
    checks.append(
        HealCheck(
            name="no_severe_with_signal",
            passed=True,
            detail=("all severe axes (if any) have CIs crossing zero — within noise"),
        )
    )

    # ---- Refinement: heal vs propose ----------------------------
    non_style = [d for d in divergences if d.get("kind") != "style_drift"]
    if not non_style:
        checks.append(
            HealCheck(
                name="cosmetic_only",
                passed=True,
                detail=(
                    "only cosmetic (style) divergences detected"
                    if divergences
                    else "no divergences detected"
                ),
            )
        )
        return HealDecision(
            tier=HealTier.HEAL,
            anchor_id=anchor_id,
            candidate_id=candidate_id,
            pair_count=pair_count,
            checks=checks,
            rationale=(
                "Every hard-refusal gate passed and the only divergences (if any) "
                "are cosmetic style drift. The change is provably implementation-"
                "only by the math; an auto-mode would accept the candidate as the "
                "new anchor."
            ),
        )

    # Non-cosmetic divergences exist but no hard gate refused — propose.
    other_kinds = sorted({str(d.get("kind", "?")) for d in non_style})
    checks.append(
        HealCheck(
            name="cosmetic_only",
            passed=False,
            detail=(
                f"non-cosmetic divergences present: {', '.join(other_kinds)} "
                "— heal not safe, propose only"
            ),
        )
    )
    return HealDecision(
        tier=HealTier.PROPOSE,
        anchor_id=anchor_id,
        candidate_id=candidate_id,
        pair_count=pair_count,
        checks=checks,
        rationale=(
            "Hard-refusal gates passed (no structural or safety drift, no severe "
            "signal above noise) but non-cosmetic divergences remain. An auto-mode "
            "would save the candidate as a named variant for the user to approve "
            "explicitly — it would not auto-accept."
        ),
    )


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _short_id(content_id: str) -> str:
    """First eight hex chars of a SHA-256 content id, or empty string."""
    if not content_id:
        return ""
    if content_id.startswith("sha256:"):
        content_id = content_id[len("sha256:") :]
    return content_id[:8]
