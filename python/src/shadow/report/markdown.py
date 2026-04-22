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
        "| axis | baseline | candidate | delta | 95% CI | severity | n |",
        "|------|---------:|----------:|------:|:-------|:---------|---:|",
    ]
    worst = "none"
    for row in report.get("rows", []):
        sev = row.get("severity", "none")
        if _sev_rank(sev) > _sev_rank(worst):
            worst = sev
        lines.append(
            f"| {row.get('axis', '')} "
            f"| {row.get('baseline_median', 0.0):.3f} "
            f"| {row.get('candidate_median', 0.0):.3f} "
            f"| {row.get('delta', 0.0):+.3f} "
            f"| [{row.get('ci95_low', 0.0):+.2f}, {row.get('ci95_high', 0.0):+.2f}] "
            f"| {_sev_label(sev)} "
            f"| {row.get('n', 0)} |"
        )
    lines.append("")
    lines.append(f"**Worst severity:** {_sev_label(worst)}")
    return "\n".join(lines) + "\n"


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
