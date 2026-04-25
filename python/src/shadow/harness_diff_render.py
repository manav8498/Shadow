"""Renderers for harness-event diff output.

:func:`shadow.v02_records.harness_event_diff` returns
:class:`HarnessEventDelta` objects with count delta + first-occurrence
pair index per ``(category, name)``. This module turns that into
two reviewer-friendly surfaces:

- :func:`render_terminal` — `rich`-friendly compact output for the
  CLI (``shadow diff --harness-diff``).
- :func:`render_markdown` — table for GitHub PR comments
  (``shadow report --format github-pr``).

Both renderers prioritise the regression cases (positive count
deltas) and fold "fixes" (negative deltas) into a separate section so
reviewers see what got worse first. Severity ordering matches the
nine-axis report convention: ``error`` > ``warning`` > ``info``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shadow.v02_records import HarnessEventDelta


_SEVERITY_RANK = {"error": 3, "warning": 2, "info": 1, "fatal": 4}


def _rank_for_sort(d: HarnessEventDelta) -> tuple[int, int, int]:
    """Sort key: regressions first (positive count delta), then by
    severity desc, then by absolute count delta desc."""
    is_regression = 0 if d.count_delta > 0 else 1
    sev = -_SEVERITY_RANK.get(d.severity, 0)
    abs_delta = -abs(d.count_delta)
    return (is_regression, sev, abs_delta)


def render_terminal(deltas: list[HarnessEventDelta]) -> str:
    """Compact human-readable rendering of harness event deltas.

    Empty list returns a one-line "no harness events" notice so
    callers can include the renderer's output unconditionally.
    """
    if not deltas:
        return "harness events: no events in either trace."
    sorted_deltas = sorted(deltas, key=_rank_for_sort)
    regressions = [d for d in sorted_deltas if d.count_delta > 0]
    fixes = [d for d in sorted_deltas if d.count_delta < 0]
    # Severity shifts at unchanged count: real regressions/fixes that
    # the count-delta view alone misses.
    severity_regressions = [
        d
        for d in sorted_deltas
        if d.count_delta == 0 and d.severity_shift == "regression"
    ]
    severity_fixes = [
        d for d in sorted_deltas if d.count_delta == 0 and d.severity_shift == "fix"
    ]
    unchanged = [
        d for d in sorted_deltas if d.count_delta == 0 and d.severity_shift is None
    ]

    lines: list[str] = []
    lines.append(
        f"harness events: {len(regressions)} regression(s), "
        f"{len(fixes)} fix(es), {len(severity_regressions)} severity regression(s), "
        f"{len(severity_fixes)} severity fix(es), {len(unchanged)} unchanged"
    )
    if regressions:
        lines.append("")
        lines.append("regressions (candidate has more):")
        for d in regressions:
            sev_marker = _terminal_severity_marker(d.severity)
            first = (
                f" first at pair {d.first_occurrence_candidate}"
                if d.first_occurrence_candidate is not None
                else ""
            )
            lines.append(
                f"  {sev_marker} {d.category}.{d.name}: "
                f"{d.baseline_count} → {d.candidate_count} "
                f"({_signed(d.count_delta)}){first}"
            )
    if severity_regressions:
        lines.append("")
        lines.append("severity regressions (count unchanged, severity worse):")
        for d in severity_regressions:
            sev_marker = _terminal_severity_marker(d.candidate_severity or d.severity)
            lines.append(
                f"  {sev_marker} {d.category}.{d.name}: "
                f"{d.baseline_severity} → {d.candidate_severity} "
                f"(count {d.candidate_count})"
            )
    if fixes:
        lines.append("")
        lines.append("fixes (candidate has fewer):")
        for d in fixes:
            lines.append(
                f"  ✓ {d.category}.{d.name}: "
                f"{d.baseline_count} → {d.candidate_count} "
                f"({_signed(d.count_delta)})"
            )
    if severity_fixes:
        lines.append("")
        lines.append("severity fixes (count unchanged, severity better):")
        for d in severity_fixes:
            lines.append(
                f"  ✓ {d.category}.{d.name}: "
                f"{d.baseline_severity} → {d.candidate_severity} "
                f"(count {d.candidate_count})"
            )
    return "\n".join(lines)


def render_markdown(deltas: list[HarnessEventDelta]) -> str:
    """Markdown for GitHub PR comments.

    Two tables: regressions (sorted by severity then absolute delta),
    then fixes. Empty list returns a one-liner so CI can pipe the
    output unconditionally.
    """
    if not deltas:
        return "_No harness events in either trace._"
    sorted_deltas = sorted(deltas, key=_rank_for_sort)
    regressions = [d for d in sorted_deltas if d.count_delta > 0]
    fixes = [d for d in sorted_deltas if d.count_delta < 0]

    sections: list[str] = []
    if regressions:
        sections.append("**Harness-event regressions** (candidate has more):")
        rows = [
            "| event | severity | baseline | candidate | Δ | first at |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for d in regressions:
            first = (
                f"pair {d.first_occurrence_candidate}"
                if d.first_occurrence_candidate is not None
                else "—"
            )
            sev = _markdown_severity_marker(d.severity)
            rows.append(
                f"| `{d.category}.{d.name}` | {sev} | {d.baseline_count} | "
                f"{d.candidate_count} | {_signed(d.count_delta)} | {first} |"
            )
        sections.append("\n".join(rows))
    if fixes:
        sections.append("\n**Harness-event fixes** (candidate has fewer):")
        rows = [
            "| event | baseline | candidate | Δ |",
            "|---|---:|---:|---:|",
        ]
        for d in fixes:
            rows.append(
                f"| `{d.category}.{d.name}` | {d.baseline_count} | "
                f"{d.candidate_count} | {_signed(d.count_delta)} |"
            )
        sections.append("\n".join(rows))
    if not sections:
        return "_No harness-event count changes between baseline and candidate._"
    return "\n\n".join(sections)


def _signed(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)


def _terminal_severity_marker(severity: str) -> str:
    return {
        "fatal": "💥",
        "error": "🔴",
        "warning": "🟠",
        "info": "🟡",
    }.get(severity, "·")


def _markdown_severity_marker(severity: str) -> str:
    return {
        "fatal": "🔴 fatal",
        "error": "🔴 error",
        "warning": "🟠 warning",
        "info": "🟡 info",
    }.get(severity, severity)


__all__ = ["render_markdown", "render_terminal"]
