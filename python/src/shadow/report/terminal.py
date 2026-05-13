"""Terminal rendering for DiffReport dicts using rich."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from shadow.report.labels import axis_label


def render_terminal(report: dict[str, Any], console: Console | None = None) -> None:
    """Pretty-print a DiffReport dict to a Rich console (stderr by default)."""
    con = console or Console()
    pair_count = int(report.get("pair_count", 0))
    con.print(f"[bold]Shadow diff[/bold] — {pair_count} response pair(s)")
    con.print(f"baseline : {report.get('baseline_trace_id', '')}")
    con.print(f"candidate: {report.get('candidate_trace_id', '')}")
    # Low-n banner: bootstrap CIs at n < 5 are unreliable, so signal
    # this before the user reads the severity column. Matches the
    # `LowPower` flag the Rust differ emits per-row.
    # Tiered statistical-power banner (v3.2.4). Replaces the prior
    # binary n<5 banner with three levels (low / moderate / adequate)
    # and a concrete recommended sample size for the next tier.
    if pair_count > 0:
        from shadow.report.statistical_power import classify_power, power_blurb

        tier = classify_power(pair_count)
        if tier in {"low", "moderate"}:
            style = "yellow" if tier == "low" else "bold yellow"
            con.print(f"[{style}]⚠  {power_blurb(pair_count)}[/]")
        elif tier == "adequate":
            con.print(f"[dim]{power_blurb(pair_count)}[/]")
    # Long-form TF-IDF hint: BM25/TF-IDF semantic distance over-alarms
    # on long-form outputs where vocabularies legitimately diverge
    # (e.g. GPT-4.1 vs GPT-5 deep-research reports, multi-paragraph
    # summarisations). When the default semantic backend is in use AND
    # the response_meaning axis flagged moderate/severe AND the median
    # response length suggests long-form, recommend embeddings.
    if _should_hint_embeddings(report):
        con.print(
            "[yellow]hint[/]: long-form responses with default TF-IDF "
            "semantic distance can over-alarm on legitimate paraphrase. "
            "Re-run with `--semantic` (requires `shadow-diff[embeddings]`) "
            "for paraphrase-robust scoring."
        )
    con.print()
    rows_list = list(report.get("rows", []))
    # Suppress the universal `low_power` flag (already covered by the
    # banner above) so the "flags" column doesn't read as redundant
    # warnings on every row.
    flag_universal = (
        "low_power"
        if rows_list and all("low_power" in (r.get("flags") or []) for r in rows_list)
        else None
    )
    table = Table(show_lines=False, pad_edge=False)
    for col, justify in (
        ("signal", "left"),
        ("baseline", "right"),
        ("candidate", "right"),
        ("change", "right"),
        ("95% CI", "right"),
        ("severity", "left"),
        ("flags", "left"),
        ("n", "right"),
    ):
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    worst = "none"
    for row in rows_list:
        sev = row.get("severity", "none")
        if _sev_rank(sev) > _sev_rank(worst):
            worst = sev
        flags = [f for f in (row.get("flags") or []) if f != flag_universal]
        flags_str = ",".join(flags) if flags else ""
        table.add_row(
            axis_label(str(row.get("axis", ""))),
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

    drill_down = report.get("drill_down") or []
    if drill_down:
        con.print()
        shown = min(3, len(drill_down))
        header = "top regressive pairs"
        con.print(
            f"[bold]{header}[/] "
            f"({shown} shown" + (f" of {len(drill_down)}" if len(drill_down) > 3 else "") + "):"
        )
        for row in drill_down[:3]:
            _print_drill_down_row_terminal(con, row)
        if len(drill_down) > 3:
            con.print(f"  [dim]+ {len(drill_down) - 3} more pair(s) — see JSON output[/]")


def _print_drill_down_row_terminal(con: Console, row: dict[str, Any]) -> None:
    idx = row.get("pair_index", 0)
    axis = axis_label(str(row.get("dominant_axis", "")))
    score = float(row.get("regression_score", 0.0))
    con.print(
        f"  pair [cyan]#{idx}[/]  ·  biggest mover: [magenta]{axis}[/]  "
        f"·  score: [bold]{score:.2f}[/]"
    )
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
        con.print(
            f"    [dim]{axis_label(str(s.get('axis', ''))):<18}[/] "
            f"{bv:.2f} → {cv:.2f}  "
            f"([italic]Δ {delta:+.2f}, norm {norm:.2f}[/])"
        )


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
    axis = axis_label(str(dv.get("primary_axis", "")))
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
        f"    kind: [{style}]{kind}[/]  ·  signal: [magenta]{axis}[/]  "
        f"·  confidence: {conf * 100:.0f}%"
    )
    con.print(f"    [italic]{exp}[/]")


_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}

# Median response length above which TF-IDF semantic distance starts
# over-alarming on legitimate paraphrase (a GPT-4.1 vs GPT-5 deep-research
# report is the canonical example). Picked empirically from the external
# reviewer's reproduction. Embeddings stay accurate above this threshold.
_LONG_FORM_TOKEN_THRESHOLD = 200


def _should_hint_embeddings(report: dict[str, Any]) -> bool:
    """True when the default TF-IDF semantic backend is in use AND the
    semantic axis flagged moderate/severe AND median response length
    suggests long-form output. In that combination, TF-IDF vocabulary
    divergence is a known false-positive driver and embeddings give a
    materially better answer.
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


def _sev_style(sev: str) -> str:
    return {
        "none": "green",
        "minor": "yellow",
        "moderate": "bold yellow",
        "severe": "bold red",
    }.get(sev, "default")
