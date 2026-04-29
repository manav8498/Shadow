"""Pure decision logic for ``shadow call``.

Takes a diff report dict and produces a :class:`CallResult` containing
the tier (ship / hold / probe / stop), the dominant driver, the worst
axes, and a short ordered list of suggested next commands.

No I/O, no Rich, no surprises. Every output is a function of the input
report dict — re-running on the same dict is byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Tier(str, Enum):
    """The four-way ship-readiness call.

    Ordered from green to red so callers can compare ranks numerically
    via :attr:`rank` without pattern matching.
    """

    SHIP = "ship"
    HOLD = "hold"
    PROBE = "probe"
    STOP = "stop"

    @property
    def rank(self) -> int:
        return {
            Tier.SHIP: 0,
            Tier.HOLD: 1,
            Tier.PROBE: 2,
            Tier.STOP: 3,
        }[self]


class Confidence(str, Enum):
    """How comfortable the underlying CI is.

    ``firm`` — the CI excludes zero and the effect is well above noise.
    ``fair`` — the CI excludes zero but the effect is near the noise floor.
    ``faint`` — the CI crosses zero; the signal is directional.
    """

    FIRM = "firm"
    FAIR = "fair"
    FAINT = "faint"


@dataclass
class AxisLine:
    """One axis row that contributed to the call.

    Carries enough numeric detail for the renderer to lay out a clean
    table without re-deriving anything from the raw report.
    """

    axis: str
    delta: float
    ci_low: float
    ci_high: float
    severity: str  # "none" | "minor" | "moderate" | "severe"
    confidence: Confidence
    n: int


@dataclass
class Driver:
    """The dominant cause shown to the reviewer.

    The driver is derived from the diff report's structured divergence
    list when one is available, falling back to the worst axis when no
    divergence has been classified.
    """

    summary: str  # one-line human-readable description
    turn: int | None  # baseline turn index, when applicable
    primary_axis: str  # axis name the driver surfaces on
    confidence: Confidence
    detail: str = ""  # optional second line for renderer

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "turn": self.turn,
            "primary_axis": self.primary_axis,
            "confidence": self.confidence.value,
            "detail": self.detail,
        }


@dataclass
class CallResult:
    """The full output of :func:`compute_call`."""

    tier: Tier
    pair_count: int
    anchor_id: str  # short trace id of the baseline (8-char prefix)
    candidate_id: str  # short trace id of the candidate
    driver: Driver | None
    worst_axes: list[AxisLine] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def exit_code(self) -> int:
        """Conventional exit code for the call.

        ``ship``  → 0 (CI passes)
        ``hold``  → 0 (review required but does not block)
        ``probe`` → 0 (informational; ``--strict`` callers can override)
        ``stop``  → 1 (CI blocks the merge)
        """
        return 1 if self.tier is Tier.STOP else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "pair_count": self.pair_count,
            "anchor_id": self.anchor_id,
            "candidate_id": self.candidate_id,
            "driver": self.driver.to_dict() if self.driver else None,
            "worst_axes": [
                {
                    "axis": a.axis,
                    "delta": a.delta,
                    "ci_low": a.ci_low,
                    "ci_high": a.ci_high,
                    "severity": a.severity,
                    "confidence": a.confidence.value,
                    "n": a.n,
                }
                for a in self.worst_axes
            ],
            "suggestions": list(self.suggestions),
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# Tunables — kept at module scope so they're easy to surface in config later.
# Conservative defaults: the call layer prefers PROBE over STOP when the
# evidence is thin, never the other way around.
# ---------------------------------------------------------------------------

#: Below this pair count the data is too thin for a definitive call,
#: regardless of how loud the per-axis severity reads.
_MIN_PAIRS_FOR_CALL = 5

#: An effect is "comfortably above noise" when |delta| is at least this
#: multiple of the CI half-width. Used to grade ``firm`` vs ``fair``.
_FIRM_DELTA_TO_HALFWIDTH_RATIO = 2.0


def compute_call(report: dict[str, Any]) -> CallResult:
    """Reduce a diff report to a single decisive call.

    Parameters
    ----------
    report:
        The dict returned by :func:`shadow._core.compute_diff_report`.

    Returns
    -------
    CallResult
        A populated :class:`CallResult`. Always non-empty; the worst-case
        return is ``Tier.PROBE`` with a "no data" reason when the report
        is empty or malformed.
    """
    pair_count = int(report.get("pair_count", 0) or 0)
    rows = list(report.get("rows") or [])
    divergences = list(report.get("divergences") or [])

    anchor_id = _short_id(report.get("baseline_trace_id", ""))
    candidate_id = _short_id(report.get("candidate_trace_id", ""))

    tier, reasons = _decide_tier(pair_count, rows)
    worst_axes = _summarise_worst_axes(rows, limit=3)
    driver = _extract_driver(divergences, worst_axes)
    suggestions = _suggestions_for(tier, driver, pair_count)

    return CallResult(
        tier=tier,
        pair_count=pair_count,
        anchor_id=anchor_id,
        candidate_id=candidate_id,
        driver=driver,
        worst_axes=worst_axes,
        suggestions=suggestions,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Tier rules
# ---------------------------------------------------------------------------


def _decide_tier(pair_count: int, rows: list[dict[str, Any]]) -> tuple[Tier, list[str]]:
    """Compute the call tier and the reasons string the renderer surfaces.

    Rules, in priority order:

        1. ``probe`` when ``pair_count`` is below the floor.
        2. ``stop``  when any axis is severe.
        3. ``hold``  when any axis is moderate.
        4. ``ship``  otherwise.

    The first rule wins; subsequent rules are evaluated only if the
    earlier ones don't fire. This keeps the decision deterministic and
    auditable from the reasons list alone.
    """
    reasons: list[str] = []

    if pair_count < _MIN_PAIRS_FOR_CALL:
        reasons.append(
            f"only {pair_count} response pair(s) — record at least "
            f"{_MIN_PAIRS_FOR_CALL} for a definitive call"
        )
        return Tier.PROBE, reasons

    severe = [r for r in rows if r.get("severity") == "severe"]
    if severe:
        names = ", ".join(str(r.get("axis", "?")) for r in severe[:3])
        more = "" if len(severe) <= 3 else f" (+{len(severe) - 3} more)"
        reasons.append(f"{len(severe)} axis severe: {names}{more}")
        return Tier.STOP, reasons

    moderate = [r for r in rows if r.get("severity") == "moderate"]
    if moderate:
        names = ", ".join(str(r.get("axis", "?")) for r in moderate[:3])
        more = "" if len(moderate) <= 3 else f" (+{len(moderate) - 3} more)"
        reasons.append(f"{len(moderate)} axis moderate: {names}{more}")
        return Tier.HOLD, reasons

    reasons.append("no axis crossed the moderate threshold")
    return Tier.SHIP, reasons


# ---------------------------------------------------------------------------
# Confidence grading
# ---------------------------------------------------------------------------


def _classify_confidence(delta: float, ci_low: float, ci_high: float, n: int) -> Confidence:
    """Map (delta, CI bounds, sample size) to one of the three F-tiers.

    The rules are deliberately conservative: when the CI crosses zero,
    confidence is always ``faint`` regardless of the point estimate.
    """
    if n <= 0:
        return Confidence.FAINT
    crosses_zero = ci_low <= 0.0 <= ci_high
    if crosses_zero:
        return Confidence.FAINT
    half_width = max(abs(ci_high - ci_low) / 2.0, 1e-12)
    ratio = abs(delta) / half_width
    if ratio >= _FIRM_DELTA_TO_HALFWIDTH_RATIO:
        return Confidence.FIRM
    return Confidence.FAIR


# ---------------------------------------------------------------------------
# Axis summary
# ---------------------------------------------------------------------------


_SEVERITY_RANK = {"severe": 3, "moderate": 2, "minor": 1, "none": 0}


def _summarise_worst_axes(rows: list[dict[str, Any]], *, limit: int) -> list[AxisLine]:
    """Pick the most-regressed axes and convert each to an :class:`AxisLine`.

    Sort key: severity rank (severe first), then |delta| descending so
    larger movements outrank smaller ones at the same severity. Ties
    break on axis name for stability.
    """
    scored = []
    for r in rows:
        sev = str(r.get("severity", "none"))
        if sev == "none":
            continue
        delta = float(r.get("delta", 0.0) or 0.0)
        ci_low = float(r.get("ci95_low", 0.0) or 0.0)
        ci_high = float(r.get("ci95_high", 0.0) or 0.0)
        n = int(r.get("n", 0) or 0)
        line = AxisLine(
            axis=str(r.get("axis", "?")),
            delta=delta,
            ci_low=ci_low,
            ci_high=ci_high,
            severity=sev,
            confidence=_classify_confidence(delta, ci_low, ci_high, n),
            n=n,
        )
        scored.append((_SEVERITY_RANK.get(sev, 0), abs(delta), line.axis, line))

    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [line for _sev, _abs, _name, line in scored[:limit]]


# ---------------------------------------------------------------------------
# Driver extraction
# ---------------------------------------------------------------------------


def _extract_driver(
    divergences: list[dict[str, Any]],
    worst_axes: list[AxisLine],
) -> Driver | None:
    """Pick the dominant driver to display.

    Order of preference:

        1. The first ``structural_drift`` divergence if any — these are
           almost always the real cause of a behavioural regression.
        2. The first ``decision_drift`` divergence if any — usually a
           refusal flip or arg-value change worth surfacing.
        3. The worst axis row, when no divergence was classified.
        4. ``None`` when the report has no movement at all.
    """
    structural = [d for d in divergences if d.get("kind") == "structural_drift"]
    decision = [d for d in divergences if d.get("kind") == "decision_drift"]

    pick = None
    if structural:
        pick = structural[0]
    elif decision:
        pick = decision[0]

    if pick is not None:
        confidence = _confidence_from_divergence(pick)
        return Driver(
            summary=_driver_summary_from_divergence(pick),
            turn=int(pick.get("baseline_turn", 0) or 0),
            primary_axis=str(pick.get("primary_axis", "?")),
            confidence=confidence,
            detail=str(pick.get("explanation", "")),
        )

    if worst_axes:
        a = worst_axes[0]
        return Driver(
            summary=f"{a.axis} moved by Δ {a.delta:+.3f} ({a.severity})",
            turn=None,
            primary_axis=a.axis,
            confidence=a.confidence,
            detail=(
                f"{a.severity} severity over {a.n} pair(s); "
                f"95% CI [{a.ci_low:+.3f}, {a.ci_high:+.3f}]"
            ),
        )

    return None


def _driver_summary_from_divergence(dv: dict[str, Any]) -> str:
    """One short headline for the driver block."""
    kind = str(dv.get("kind", ""))
    turn = int(dv.get("baseline_turn", 0) or 0)
    if kind == "structural_drift":
        return f"structural change at turn {turn}"
    if kind == "decision_drift":
        return f"decision change at turn {turn}"
    if kind == "style_drift":
        return f"style change at turn {turn}"
    return f"change at turn {turn}"


def _confidence_from_divergence(dv: dict[str, Any]) -> Confidence:
    """Best-effort confidence label for a divergence record.

    The Rust differ stores a 0..1 confidence score on each divergence.
    Map that to one of the three F-tiers using the same noise floor
    the rendering layer uses for axes.
    """
    score = float(dv.get("confidence", 0.0) or 0.0)
    if score >= 0.6:
        return Confidence.FIRM
    if score >= 0.35:
        return Confidence.FAIR
    return Confidence.FAINT


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------


def _suggestions_for(
    tier: Tier,
    driver: Driver | None,
    pair_count: int,
) -> list[str]:
    """Two or three next-step commands the reviewer can copy-paste.

    Returned strings are bare shell commands without the leading ``$``
    so the renderer can prefix them consistently.
    """
    out: list[str] = []
    if tier is Tier.STOP:
        out.append(
            "shadow autopr <anchor>.agentlog <candidate>.agentlog -o tests/regressions/r.yaml"
        )
        out.append("shadow diff --judge auto <anchor>.agentlog <candidate>.agentlog")
    elif tier is Tier.HOLD:
        out.append("shadow diff --judge auto <anchor>.agentlog <candidate>.agentlog")
        out.append("shadow autopr --no-verify <anchor>.agentlog <candidate>.agentlog")
    elif tier is Tier.PROBE:
        if pair_count == 0:
            out.append("shadow record -o anchor.agentlog -- python your_agent.py")
        else:
            out.append(f"record at least {_MIN_PAIRS_FOR_CALL - pair_count} more pair(s)")
        out.append("shadow diff --token-diff <anchor>.agentlog <candidate>.agentlog")
    return out


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
