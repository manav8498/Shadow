"""Markdown rendering for DiffReport dicts."""

from __future__ import annotations

from typing import Any


def render_markdown(report: dict[str, Any]) -> str:
    """Render a DiffReport dict as a PR-friendly markdown table."""
    pair_count = report.get("pair_count", 0)
    lines = [
        f"# Shadow diff — {pair_count} response pair{'s' if pair_count != 1 else ''}",
        "",
        f"**Baseline:** `{_short(report.get('baseline_trace_id', ''))}`  ",
        f"**Candidate:** `{_short(report.get('candidate_trace_id', ''))}`  ",
        "",
        "| axis | baseline | candidate | delta | 95% CI | severity | flags | n |",
        "|------|---------:|----------:|------:|:-------|:---------|:------|---:|",
    ]
    worst = "none"
    for row in report.get("rows", []):
        sev = row.get("severity", "none")
        if _sev_rank(sev) > _sev_rank(worst):
            worst = sev
        flags = row.get("flags", []) or []
        flags_str = ", ".join(f"`{f}`" for f in flags) if flags else ""
        lines.append(
            f"| {row.get('axis', '')} "
            f"| {row.get('baseline_median', 0.0):.3f} "
            f"| {row.get('candidate_median', 0.0):.3f} "
            f"| {row.get('delta', 0.0):+.3f} "
            f"| [{row.get('ci95_low', 0.0):+.2f}, {row.get('ci95_high', 0.0):+.2f}] "
            f"| {_sev_label(sev)} "
            f"| {flags_str} "
            f"| {row.get('n', 0)} |"
        )
    lines.append("")
    lines.append(f"**Worst severity:** {_sev_label(worst)}")
    # Prefer the top-K `divergences` list when present; fall back to the
    # scalar `first_divergence` field for backward compat with callers
    # that were built before top-K landed.
    divergences = report.get("divergences") or []
    if not divergences and report.get("first_divergence"):
        divergences = [report["first_divergence"]]
    if divergences:
        lines.append("")
        total = len(divergences)
        header = "### Top divergences" if total > 1 else "### First divergence"
        lines.append(header)
        lines.append("")
        # Show top 3 inline. Remaining go into a collapsible section.
        inline_count = min(3, total)
        for idx, dv in enumerate(divergences[:inline_count]):
            lines.append(_render_divergence_markdown(dv, idx + 1, total))
        if total > inline_count:
            lines.append("")
            lines.append("<details>")
            lines.append(
                f"<summary>+{total - inline_count} more (ranks "
                f"{inline_count + 1}-{total})</summary>\n"
            )
            for idx, dv in enumerate(divergences[inline_count:], start=inline_count):
                lines.append(_render_divergence_markdown(dv, idx + 1, total))
            lines.append("</details>")
    recommendations = report.get("recommendations") or []
    if recommendations:
        lines.append("")
        lines.append("### Recommendations")
        lines.append("")
        for rec in recommendations:
            lines.append(_render_recommendation_markdown(rec))
    return "\n".join(lines) + "\n"


_REC_SEVERITY_ICON = {
    "error": "🔴",
    "warning": "🟡",
    "info": "🔵",
}


def _render_recommendation_markdown(rec: dict[str, Any]) -> str:
    """One recommendation as a markdown bullet with severity icon."""
    sev = rec.get("severity", "info")
    action = rec.get("action", "").upper()
    message = rec.get("message", "")
    rationale = rec.get("rationale", "")
    icon = _REC_SEVERITY_ICON.get(sev, "")
    parts = [f"- {icon} **`{sev}`** — **{action}** — {message}"]
    if rationale:
        parts.append(f"  - _{rationale}_")
    return "\n".join(parts)


def _render_divergence_markdown(dv: dict[str, Any], rank: int, total: int) -> str:
    """One divergence as a markdown block with a rank header."""
    kind = dv.get("kind", "")
    axis = dv.get("primary_axis", "")
    bt = dv.get("baseline_turn", 0)
    ct = dv.get("candidate_turn", 0)
    conf = dv.get("confidence", 0.0)
    exp = dv.get("explanation", "")
    rank_prefix = f"**#{rank}" + (f" of {total}**" if total > 1 else "**")
    return (
        f"\n{rank_prefix} &nbsp; **Turn** baseline `#{bt}` ↔ candidate `#{ct}` "
        f"&nbsp; · &nbsp; **Kind** `{kind}` &nbsp; · &nbsp; **Axis** `{axis}` "
        f"&nbsp; · &nbsp; **Confidence** {conf * 100:.0f}%\n\n> {exp}"
    )


_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
_EMOJI = {"none": "🟢", "minor": "🟡", "moderate": "🟠", "severe": "🔴"}


def _sev_rank(sev: str) -> int:
    return _RANK.get(sev, 0)


def _sev_label(sev: str) -> str:
    return f"{_EMOJI.get(sev, '')} {sev}".strip()


def _short(trace_id: str) -> str:
    if len(trace_id) > 16:
        return f"{trace_id[:12]}…{trace_id[-4:]}"
    return trace_id
