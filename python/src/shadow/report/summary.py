"""Plain-English deterministic summary of a DiffReport.

Turns the structured nine-axis table into a short paragraph a
reviewer can read in one breath — the "what this means" line that
goes between raw metrics and the next click.

Design choices:

1. **Deterministic templating, no LLM.** A summary that cites concrete
   numbers needs to be reproducible. Any LLM round-trip risks rewriting
   the numbers; a template doesn't. `shadow diff --explain` is the
   opt-in LLM-sourced narrative layer on top.

2. **Axis priority.** Summaries lead with structural signals that
   matter to a PR reviewer: trajectory, safety, conformance, first-
   divergence. Scalar axes (verbosity / latency / cost) come after,
   with their deltas only if they hit `moderate` or `severe`.

3. **Low-n honesty.** With `pair_count < 5`, the summary opens with a
   one-line caveat rather than burying it. Matches the `LowPower`
   flag the Rust differ already emits.

4. **Word budget.** Aim for ~60 words; cap at ~120. Longer summaries
   just get skipped by readers.
"""

from __future__ import annotations

from typing import Any

# axes-to-label mapping, phrased for sentences (not column headers).
_AXIS_NARRATIVE = {
    "semantic": "semantic similarity",
    "trajectory": "tool-call trajectory",
    "safety": "refusal rate",
    "verbosity": "response length",
    "latency": "end-to-end latency",
    "cost": "per-call cost",
    "reasoning": "reasoning depth",
    "judge": "judge score",
    "conformance": "format conformance",
}

_STRUCTURAL_AXES = ("trajectory", "safety", "conformance", "semantic")
_SCALAR_AXES = ("verbosity", "latency", "cost", "reasoning", "judge")


def _sev_rank(sev: str) -> int:
    return {"none": 0, "minor": 1, "moderate": 2, "severe": 3}.get(sev, 0)


def _row_for(report: dict[str, Any], axis: str) -> dict[str, Any] | None:
    for row in report.get("rows", []) or []:
        if row.get("axis") == axis:
            assert isinstance(row, dict)
            return row
    return None


def _fmt_delta(row: dict[str, Any]) -> str:
    """Format a delta as a human-readable string. Adds units per axis."""
    delta = float(row.get("delta", 0.0))
    axis = row.get("axis", "")
    if axis == "latency":
        return f"{delta:+.0f} ms"
    if axis == "verbosity":
        return f"{delta:+.0f} tokens"
    if axis == "cost":
        return f"${delta:+.4f}"
    if axis in ("reasoning",):
        return f"{delta:+.0f}"
    return f"{delta:+.2f}"


def _low_n_caveat(report: dict[str, Any]) -> str:
    n = int(report.get("pair_count", 0))
    if n == 0:
        return "No paired responses — diff is empty."
    if n < 5:
        return (
            f"Only {n} paired response{'s' if n != 1 else ''} — severities below "
            "are directional, not definitive (record 10+ for stable CIs)."
        )
    return ""


def _structural_bullets(report: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for axis in _STRUCTURAL_AXES:
        row = _row_for(report, axis)
        if row is None:
            continue
        sev = row.get("severity", "none")
        if _sev_rank(sev) < 2:  # below `moderate`, skip
            continue
        label = _AXIS_NARRATIVE[axis]
        delta = _fmt_delta(row)
        out.append(f"{label} moved {delta} ({sev})")
    return out


def _scalar_bullets(report: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for axis in _SCALAR_AXES:
        row = _row_for(report, axis)
        if row is None:
            continue
        sev = row.get("severity", "none")
        if _sev_rank(sev) < 2:
            continue
        label = _AXIS_NARRATIVE[axis]
        delta = _fmt_delta(row)
        out.append(f"{label} {delta} ({sev})")
    return out


def _first_divergence_line(report: dict[str, Any]) -> str:
    fd = report.get("first_divergence")
    if not fd:
        return ""
    kind = fd.get("kind", "")
    bt = fd.get("baseline_turn", 0)
    ct = fd.get("candidate_turn", 0)
    explanation = (fd.get("explanation") or "").strip()
    if explanation:
        # Explanation is already terse; trust it.
        return f"First divergence: turn #{bt}/#{ct} ({kind.replace('_', ' ')}) — " f"{explanation}"
    return f"First divergence at turn #{bt}/#{ct} ({kind.replace('_', ' ')})."


def _top_drill_down_line(report: dict[str, Any]) -> str:
    drill = report.get("drill_down") or []
    if not drill:
        return ""
    top = drill[0]
    score = float(top.get("regression_score", 0.0))
    if score < 1.0:
        return ""  # below 1.0 is essentially noise; don't highlight
    return (
        f"Worst pair: turn #{top.get('pair_index', 0)} "
        f"(dominant: {top.get('dominant_axis', '?')}, score {score:.1f})."
    )


def _recommendation_line(report: dict[str, Any]) -> str:
    recs = report.get("recommendations") or []
    if not recs:
        return ""
    errors = [r for r in recs if r.get("severity") == "error"]
    if errors:
        top = errors[0]
        return f"First fix: {top.get('message', '').rstrip('.')}"
    if recs:
        top = recs[0]
        return f"Suggested check: {top.get('message', '').rstrip('.')}"
    return ""


def summarise_report(report: dict[str, Any]) -> str:
    """Return a plain-English summary paragraph or empty string."""
    lines: list[str] = []
    caveat = _low_n_caveat(report)
    if caveat:
        lines.append(caveat)

    structural = _structural_bullets(report)
    scalar = _scalar_bullets(report)
    headline = structural + scalar

    if not headline and not caveat:
        # All axes at minor / none. Say so explicitly rather than staying silent.
        worst = max(
            (_sev_rank(r.get("severity", "none")) for r in report.get("rows", []) or []),
            default=0,
        )
        if worst == 0:
            lines.append("All axes within noise floor — no regression detected.")
            return "\n".join(lines)

    if headline:
        joined = "; ".join(headline)
        # Capitalise first letter without smashing existing casing.
        joined = joined[:1].upper() + joined[1:]
        lines.append(joined + ".")

    fd_line = _first_divergence_line(report)
    if fd_line:
        lines.append(fd_line)

    drill_line = _top_drill_down_line(report)
    if drill_line:
        lines.append(drill_line)

    rec_line = _recommendation_line(report)
    if rec_line:
        lines.append(rec_line)

    return "\n".join(lines)


__all__ = ["summarise_report"]
