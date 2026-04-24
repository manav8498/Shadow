"""Markdown rendering for DiffReport dicts."""

from __future__ import annotations

from typing import Any


def render_markdown(report: dict[str, Any]) -> str:
    """Render a DiffReport dict as a PR-friendly markdown table."""
    from shadow.report.summary import summarise_report

    pair_count = int(report.get("pair_count", 0))
    lines: list[str] = [
        f"# Shadow diff — {pair_count} response pair{'s' if pair_count != 1 else ''}",
        "",
        f"**Baseline:** `{_short(report.get('baseline_trace_id', ''))}`  ",
        f"**Candidate:** `{_short(report.get('candidate_trace_id', ''))}`  ",
        "",
    ]

    # What-this-means paragraph first — this is what a PR reviewer
    # reads before clicking into the axis table.
    summary = summarise_report(report)
    if summary:
        lines.append("### What this means")
        lines.append("")
        for s_line in summary.splitlines():
            lines.append(f"> {s_line}")
        lines.append("")

    # Low-n banner. Matches the terminal renderer and the `LowPower`
    # flag the Rust differ emits per-row.
    if 0 < pair_count < 5:
        lines.append(
            f"> ⚠  Only {pair_count} paired response(s) — severities below are "
            "directional, not definitive. Record 10+ turns for stable CIs."
        )
        lines.append("")

    lines.append("| axis | baseline | candidate | delta | 95% CI | severity | flags | n |")
    lines.append("|------|---------:|----------:|------:|:-------|:---------|:------|---:|")
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

    # Session-cost attribution block, if the caller attached one.
    # The CLI doesn't currently embed the attribution in the DiffReport
    # dict (it renders to terminal separately), but the markdown
    # renderer accepts an optional `cost_attribution` key for PR-
    # comment integrations that want the full story inline.
    cost_attr = report.get("cost_attribution")
    if cost_attr and abs(float(cost_attr.get("total_delta_usd", 0.0))) > 1e-9:
        from shadow.cost_attribution import (
            CostAttributionReport,
            SessionAttribution,
        )
        from shadow.cost_attribution import (
            render_markdown as render_cost_md,
        )

        # Accept either a live CostAttributionReport or the serialised
        # dict form (what `to_dict()` produces).
        if isinstance(cost_attr, dict):
            per_session = [SessionAttribution(**s) for s in cost_attr["per_session"]]
            cost_report = CostAttributionReport(
                per_session=per_session,
                total_baseline_usd=cost_attr["total_baseline_usd"],
                total_candidate_usd=cost_attr["total_candidate_usd"],
                total_delta_usd=cost_attr["total_delta_usd"],
                total_model_swap_usd=cost_attr["total_model_swap_usd"],
                total_token_movement_usd=cost_attr["total_token_movement_usd"],
                total_mix_residual_usd=cost_attr["total_mix_residual_usd"],
                attribution_is_noisy=cost_attr["attribution_is_noisy"],
            )
        else:
            cost_report = cost_attr  # already a CostAttributionReport
        md = render_cost_md(cost_report)
        if md:
            lines.append("")
            lines.append(md)

    drill_down = report.get("drill_down") or []
    if drill_down:
        lines.append("")
        lines.append("### Top regressive pairs")
        lines.append("")
        inline = drill_down[:3]
        extras = drill_down[3:]
        for row in inline:
            lines.append(_render_drill_down_row_markdown(row))
        if extras:
            lines.append("")
            lines.append(f"<details><summary>+ {len(extras)} more regressive pair(s)</summary>\n")
            for row in extras:
                lines.append(_render_drill_down_row_markdown(row))
            lines.append("</details>")
    return "\n".join(lines) + "\n"


def _render_drill_down_row_markdown(row: dict[str, Any]) -> str:
    """One drill-down pair as a markdown sub-list."""
    idx = row.get("pair_index", 0)
    axis = row.get("dominant_axis", "")
    score = float(row.get("regression_score", 0.0))
    parts = [
        f"- **pair `#{idx}`** &nbsp;·&nbsp; dominant: `{axis}` "
        f"&nbsp;·&nbsp; score `{score:.2f}`"
    ]
    scores = sorted(
        row.get("axis_scores", []),
        key=lambda s: -float(s.get("normalized_delta", 0.0)),
    )
    for s in scores[:2]:
        if float(s.get("normalized_delta", 0.0)) < 0.05:
            break
        bv = float(s.get("baseline_value", 0.0))
        cv = float(s.get("candidate_value", 0.0))
        delta = float(s.get("delta", 0.0))
        norm = float(s.get("normalized_delta", 0.0))
        parts.append(
            f"  - `{s.get('axis', '')}`: {bv:.2f} → {cv:.2f} "
            f"&nbsp;(delta `{delta:+.2f}`, norm `{norm:.2f}`)"
        )
    return "\n".join(parts)


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
