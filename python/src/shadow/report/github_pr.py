"""GitHub PR-comment flavoured markdown with production-risk language.

The basic markdown report from :func:`shadow.report.markdown.render_markdown`
shows the nine-axis table with deltas and severities. That's useful
for engineers but reads abstractly to engineering managers and
compliance reviewers ("trajectory regression on axis 2" doesn't
translate to "this PR newly allows production SQL deletion").

This renderer wraps the standard markdown report with a
**production-risk header** that:

- Flags new severe regressions in plain language (which axis, which
  rules), so the headline is "Critical: 2 new policy violations in
  this PR" rather than "axis 3 severity: severe".
- Calls out specific recommendation actions ("Restore X", "Remove
  duplicate Y") at the top, ahead of the axis table.
- Uses risk-tier bullets (CRITICAL / ERROR / WARN / INFO) so any
  reader can decide if the PR is mergeable without reading the math.

The lower part of the report (axis table, sample counts) is preserved
unchanged so engineers retain the per-axis detail.
"""

from __future__ import annotations

from typing import Any

from shadow.report.markdown import render_markdown

# Severity tier labels for the risk-summary header. Order matters —
# we surface the most urgent tier first.
_TIER_ORDER = ("critical", "error", "warning", "info")
_TIER_EMOJI = {
    "critical": "🛑",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
}
_TIER_LABEL = {
    "critical": "CRITICAL",
    "error": "ERROR",
    "warning": "WARNING",
    "info": "INFO",
}

# Severity-to-tier mapping for axis rows. Shadow's per-axis severity
# uses different vocabulary; map onto the four tiers used by the
# risk header.
_AXIS_SEVERITY_TO_TIER = {
    "severe": "critical",
    "major": "error",
    "moderate": "warning",
    "minor": "info",
    "none": "info",
}


def _tier_for_axis(severity: str) -> str:
    return _AXIS_SEVERITY_TO_TIER.get(str(severity).lower(), "warning")


def _risk_summary_header(report: dict[str, Any]) -> str:
    """Produce the production-risk summary at the top of the PR comment.

    The summary aggregates per-axis severity rows + recommendation
    severities into a tiered count, then prints one line per non-empty
    tier with the most actionable phrasing first.
    """
    # Group by tier. Each entry: (kind, message). kind is 'axis' or
    # 'recommendation'; message is the human-readable phrasing.
    by_tier: dict[str, list[tuple[str, str]]] = {tier: [] for tier in _TIER_ORDER}

    for row in report.get("rows", []) or []:
        sev = str(row.get("severity") or "")
        if sev.lower() == "none":
            continue
        tier = _tier_for_axis(sev)
        axis = str(row.get("axis") or "?")
        delta = row.get("delta")
        delta_str = f" ({delta:+.4f})" if isinstance(delta, int | float) else ""
        by_tier[tier].append(("axis", f"`{axis}` regressed{delta_str}"))

    for rec in report.get("recommendations", []) or []:
        rec_sev = str(rec.get("severity") or "warning").lower()
        tier = rec_sev if rec_sev in _TIER_ORDER else "warning"
        action = str(rec.get("action") or "review").lower()
        message = str(rec.get("message") or rec.get("rationale") or "see report")
        by_tier[tier].append(("rec", f"{action}: {message}"))

    has_any = any(by_tier[t] for t in _TIER_ORDER)
    if not has_any:
        return (
            "## ✅ Behavioral diff: no regressions\n\n"
            "All nine axes within tolerance. No new policy violations. "
            "No actionable recommendations.\n\n"
            "Detail below.\n"
        )

    # Build the header. Worst tier first.
    worst_tier = next(t for t in _TIER_ORDER if by_tier[t])
    header = (
        f"## {_TIER_EMOJI[worst_tier]} Behavioral diff: "
        f"{_TIER_LABEL[worst_tier]} regressions detected\n\n"
    )

    # Risk summary: one line per non-empty tier, with bullet items.
    summary_lines: list[str] = []
    for tier in _TIER_ORDER:
        items = by_tier[tier]
        if not items:
            continue
        summary_lines.append(f"### {_TIER_EMOJI[tier]} {_TIER_LABEL[tier]} ({len(items)})")
        for kind, msg in items[:5]:  # cap at 5 per tier so the header stays readable
            prefix = "Axis" if kind == "axis" else "Action"
            summary_lines.append(f"- **{prefix}:** {msg}")
        if len(items) > 5:
            summary_lines.append(f"- _... and {len(items) - 5} more in the table below._")
        summary_lines.append("")

    return header + "\n".join(summary_lines)


def render_github_pr(report: dict[str, Any]) -> str:
    """Produce a PR-friendly markdown comment with production-risk header.

    Three sections, top to bottom:

    1. **Risk summary** — tiered (CRITICAL / ERROR / WARNING / INFO)
       count of new regressions and recommendations, with plain-
       language phrasing. The first line tells a non-engineer
       reviewer whether the PR is mergeable.
    2. **Per-axis table** — the standard markdown report (deltas,
       severities, CIs) for engineers who want the details.
    3. **Sample counts** — collapsible `<details>` section, plus
       a "Generated by Shadow" footer link.
    """
    risk_header = _risk_summary_header(report)
    base = render_markdown(report)
    details_lines = ["<details>", "<summary>Per-axis sample counts</summary>", ""]
    details_lines.append("| axis | n |")
    details_lines.append("|------|---:|")
    for row in report.get("rows", []):
        details_lines.append(f"| {row.get('axis', '')} | {row.get('n', 0)} |")
    details_lines.append("</details>")
    details_lines.append("")
    details_lines.append("_Generated by [Shadow](https://github.com/manav8498/Shadow)._")
    return risk_header + "\n" + base + "\n" + "\n".join(details_lines) + "\n"
