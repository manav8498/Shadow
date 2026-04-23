"""Terminal rendering for DiffReport dicts using rich."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table


def render_terminal(report: dict[str, Any], console: Console | None = None) -> None:
    """Pretty-print a DiffReport dict to a Rich console (stderr by default)."""
    con = console or Console()
    con.print(f"[bold]Shadow diff[/bold] — {report.get('pair_count', 0)} response pair(s)")
    con.print(f"baseline : {report.get('baseline_trace_id', '')}")
    con.print(f"candidate: {report.get('candidate_trace_id', '')}")
    con.print()
    table = Table(show_lines=False, pad_edge=False)
    for col, justify in (
        ("axis", "left"),
        ("baseline", "right"),
        ("candidate", "right"),
        ("delta", "right"),
        ("95% CI", "right"),
        ("severity", "left"),
        ("flags", "left"),
        ("n", "right"),
    ):
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    worst = "none"
    for row in report.get("rows", []):
        sev = row.get("severity", "none")
        if _sev_rank(sev) > _sev_rank(worst):
            worst = sev
        flags = row.get("flags", []) or []
        flags_str = ",".join(flags) if flags else ""
        table.add_row(
            str(row.get("axis", "")),
            f"{row.get('baseline_median', 0.0):.3f}",
            f"{row.get('candidate_median', 0.0):.3f}",
            f"{row.get('delta', 0.0):+.3f}",
            f"[{row.get('ci95_low', 0.0):+.2f}, {row.get('ci95_high', 0.0):+.2f}]",
            f"[{_sev_style(sev)}]{sev}[/]",
            f"[dim]{flags_str}[/]" if flags_str else "",
            str(row.get("n", 0)),
        )
    con.print(table)
    con.print(f"\nworst severity: [{_sev_style(worst)}]{worst}[/]")
    # Prefer the top-K `divergences` list when present; fall back to the
    # scalar `first_divergence` field for backward compat.
    divergences = report.get("divergences") or []
    if not divergences and report.get("first_divergence"):
        divergences = [report["first_divergence"]]
    if divergences:
        total = len(divergences)
        header_word = "top divergences" if total > 1 else "first divergence"
        con.print()
        con.print(
            f"[bold]{header_word}[/] "
            f"({min(3, total)} shown" + (f" of {total}" if total > 3 else "") + "):"
        )
        for idx, dv in enumerate(divergences[:3], start=1):
            _print_divergence_terminal(con, dv, idx, total)
        if total > 3:
            con.print(
                f"  [dim]+ {total - 3} more divergence(s) "
                f"(ranks 4-{total}) — see JSON output[/]"
            )
    recommendations = report.get("recommendations") or []
    if recommendations:
        con.print()
        con.print(f"[bold]recommendations[/] ({len(recommendations)}):")
        for rec in recommendations:
            _print_recommendation_terminal(con, rec)


_REC_STYLE = {
    "error": "bold red",
    "warning": "yellow",
    "info": "cyan",
}


def _print_recommendation_terminal(con: Console, rec: dict[str, Any]) -> None:
    sev = rec.get("severity", "info")
    action = rec.get("action", "").upper()
    message = rec.get("message", "")
    rationale = rec.get("rationale", "")
    style = _REC_STYLE.get(sev, "default")
    con.print(f"  [{style}]{sev:<7}[/] [bold]{action:<7}[/] {message}")
    if rationale:
        con.print(f"             [dim italic]{rationale}[/]")


def _print_divergence_terminal(con: Console, dv: dict[str, Any], rank: int, total: int) -> None:
    kind = dv.get("kind", "")
    axis = dv.get("primary_axis", "")
    bt = dv.get("baseline_turn", 0)
    ct = dv.get("candidate_turn", 0)
    conf = dv.get("confidence", 0.0)
    exp = dv.get("explanation", "")
    style = {
        "style_drift": "dim",
        "decision_drift": "yellow",
        "structural_drift": "bold red",
    }.get(kind, "default")
    rank_label = f"#{rank}" if total > 1 else ""
    prefix = f"[bold]{rank_label}[/] " if rank_label else ""
    con.print(f"  {prefix}baseline turn [cyan]#{bt}[/] ↔ candidate turn [cyan]#{ct}[/]")
    con.print(
        f"    kind: [{style}]{kind}[/]  ·  axis: [magenta]{axis}[/]  "
        f"·  confidence: {conf * 100:.0f}%"
    )
    con.print(f"    [italic]{exp}[/]")


_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def _sev_rank(sev: str) -> int:
    return _RANK.get(sev, 0)


def _sev_style(sev: str) -> str:
    return {
        "none": "green",
        "minor": "yellow",
        "moderate": "bold yellow",
        "severe": "bold red",
    }.get(sev, "default")
