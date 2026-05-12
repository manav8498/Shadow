"""Markdown rendering for DiffReport dicts."""

from __future__ import annotations

from typing import Any

from shadow.report.labels import axis_label


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

    # Long-form TF-IDF hint — see terminal.py for the rationale.
    if _should_hint_embeddings_md(report):
        lines.append(
            "> 💡 Long-form responses with default TF-IDF semantic distance "
            "can over-alarm on legitimate paraphrase. Re-run with `--semantic` "
            "(requires `shadow-diff[embeddings]`) for paraphrase-robust scoring."
        )
        lines.append("")

    rows_list = list(report.get("rows", []))
    # `low_power` fires on every row when n is small (typical in real
    # traces with under ten paired responses). Showing it on every row
    # makes the table look broken; collapse the universal case to a
    # single banner above the table.
    flag_universal = (
        "low_power"
        if rows_list and all("low_power" in (r.get("flags") or []) for r in rows_list)
        else None
    )
    lines.append("| signal | baseline | candidate | change | 95% CI | severity | n |")
    lines.append("|--------|---------:|----------:|-------:|:-------|:---------|---:|")
    worst = "none"
    for row in rows_list:
        sev = row.get("severity", "none")
        if _sev_rank(sev) > _sev_rank(worst):
            worst = sev
        # Drop the universal `low_power` flag and any non-actionable
        # flags from per-row display; keep specific flags that change
        # the reading (e.g. `ci_crosses_zero`, `no_pricing`).
        flags = [f for f in (row.get("flags") or []) if f != flag_universal]
        flag_suffix = f" _({', '.join(flags)})_" if flags else ""
        lines.append(
            f"| {axis_label(str(row.get('axis', '')))} "
            f"| {row.get('baseline_median', 0.0):.3f} "
            f"| {row.get('candidate_median', 0.0):.3f} "
            f"| {row.get('delta', 0.0):+.3f} "
            f"| [{row.get('ci95_low', 0.0):+.2f}, {row.get('ci95_high', 0.0):+.2f}] "
            f"| {_sev_label(sev)}{flag_suffix} "
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
    axis = axis_label(str(row.get("dominant_axis", "")))
    score = float(row.get("regression_score", 0.0))
    parts = [
        f"- **pair `#{idx}`** &nbsp;·&nbsp; biggest mover: {axis} "
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
            f"  - {axis_label(str(s.get('axis', '')))}: {bv:.2f} → {cv:.2f} "
            f"&nbsp;(Δ `{delta:+.2f}`, norm `{norm:.2f}`)"
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
    axis = axis_label(str(dv.get("primary_axis", "")))
    bt = dv.get("baseline_turn", 0)
    ct = dv.get("candidate_turn", 0)
    conf = dv.get("confidence", 0.0)
    exp = dv.get("explanation", "")
    rank_prefix = f"**#{rank}" + (f" of {total}**" if total > 1 else "**")
    return (
        f"\n{rank_prefix} &nbsp; **Turn** baseline `#{bt}` ↔ candidate `#{ct}` "
        f"&nbsp; · &nbsp; **Kind** `{kind}` &nbsp; · &nbsp; **Signal** {axis} "
        f"&nbsp; · &nbsp; **Confidence** {conf * 100:.0f}%\n\n> {exp}"
    )


_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
_EMOJI = {"none": "🟢", "minor": "🟡", "moderate": "🟠", "severe": "🔴"}
_LONG_FORM_TOKEN_THRESHOLD = 200


def _should_hint_embeddings_md(report: dict[str, Any]) -> bool:
    """Markdown-side mirror of `terminal._should_hint_embeddings`. See
    that function for the rationale.
    """
    if report.get("semantic_backend") == "embeddings":
        return False
    rows = report.get("rows") or []
    semantic = next((r for r in rows if r.get("axis") == "semantic"), None)
    if semantic is None or _sev_rank(str(semantic.get("severity", "none"))) < 2:
        return False
    length = next((r for r in rows if r.get("axis") == "response_length"), None)
    if length is None:
        return False
    base_med = float(length.get("baseline_median", 0.0) or 0.0)
    cand_med = float(length.get("candidate_median", 0.0) or 0.0)
    return max(base_med, cand_med) >= _LONG_FORM_TOKEN_THRESHOLD


def _sev_rank(sev: str) -> int:
    return _RANK.get(sev, 0)


def _sev_label(sev: str) -> str:
    return f"{_EMOJI.get(sev, '')} {sev}".strip()


def _short(trace_id: str) -> str:
    if len(trace_id) > 16:
        return f"{trace_id[:12]}…{trace_id[-4:]}"
    return trace_id
