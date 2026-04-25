"""Terminal and markdown renderers for bisect attribution output.

Why a dedicated renderer
------------------------

The raw JSON `run_bisect` returns is exhaustive but reads as more
certain than the underlying statistics actually justify. Without
sandboxed counterfactual replay, attribution is **correlational, not
causally proven** — LASSO + stability selection narrows the field, but
two correlated deltas can still split a single axis's true cause and
look equally "significant."

To stay honest, the renderer:

1. Leads every output with a single-line caveat that reads
   "estimated cause, not proven; confirm with `shadow replay`."
2. Prefixes attribution percentages with ``est. ``.
3. Replaces the bare ``✓`` checkmark with an explicit qualifier
   (``stable, CI excludes 0`` or ``screening only`` etc.).
4. Calls bracketed CI bounds out as 95% bootstrap intervals so a
   reader knows what the brackets mean.

This is the same data the JSON exposes; the renderer just stops
shaping it like a courtroom verdict.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_HEADER = (
    "Bisect attribution (estimated, correlational). "
    "Confirm with `shadow replay --backend <provider>` for causal proof."
)


def render_attribution_terminal(result: dict[str, Any], *, max_rows_per_axis: int = 5) -> str:
    """Format a `run_bisect` result for human reading.

    Honest language: every percentage is prefixed ``est.``, the
    significance flag spells out what makes the row stable, and the
    output opens with a one-line caveat about correlational vs.
    causal-with-proof.
    """
    lines: list[str] = [_DEFAULT_HEADER, ""]
    attributions = result.get("attributions", {})
    if not attributions:
        lines.append("(no attributions — run with --backend or --candidate-traces)")
        return "\n".join(lines)

    for axis in sorted(attributions):
        rows = attributions[axis] or []
        if not rows:
            continue
        lines.append(f"{axis}:")
        for row in rows[:max_rows_per_axis]:
            lines.append(_format_row(row))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_attribution_markdown(result: dict[str, Any], *, max_rows_per_axis: int = 5) -> str:
    """Same shape as the terminal renderer but emits a Markdown table.

    Used by `shadow report --format markdown` and the GitHub Action's
    PR-comment renderer when bisect output is part of the comment.
    """
    out: list[str] = [
        "## Bisect attribution",
        "",
        f"_{_DEFAULT_HEADER}_",
        "",
    ]
    attributions = result.get("attributions", {})
    if not attributions:
        out.append("_No attributions — run with `--backend` or `--candidate-traces`._")
        return "\n".join(out) + "\n"
    for axis in sorted(attributions):
        rows = attributions[axis] or []
        if not rows:
            continue
        out.append(f"### {axis}")
        out.append("")
        out.append("| delta | est. share | 95% bootstrap CI | selection freq | qualifier |")
        out.append("|---|---:|---|---:|---|")
        for row in rows[:max_rows_per_axis]:
            label = str(row.get("label") or row.get("category") or "?")
            share = _format_share(row, markdown=True)
            ci = _format_ci(row)
            sel_freq = _format_selection_frequency(row)
            qualifier = _qualifier_label(row)
            out.append(f"| `{label}` | {share} | {ci} | {sel_freq} | {qualifier} |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---- row formatters ------------------------------------------------------


def _format_row(row: dict[str, Any]) -> str:
    label = str(row.get("label") or row.get("category") or "?")
    share = _format_share(row)
    ci = _format_ci(row)
    sel_freq = _format_selection_frequency(row)
    qualifier = _qualifier_label(row)
    return f"  {label:30s} {share:>10s}  {ci}  sel_freq={sel_freq}  {qualifier}"


def _format_share(row: dict[str, Any], *, markdown: bool = False) -> str:
    val = row.get("attribution") or row.get("share") or row.get("weight") or 0.0
    pct = _to_percent(val)
    if markdown:
        return f"est. {pct}"
    return f"est. {pct}"


def _format_ci(row: dict[str, Any]) -> str:
    """Bracketed 95% bootstrap percentile CI when present, else a placeholder.

    Renders as "95% CI [a%, b%]" so the reader knows what the brackets
    mean — not "[a, b]" sitting next to a percentage with no label.
    """
    low = row.get("ci_low")
    high = row.get("ci_high")
    if low is None or high is None:
        return "95% CI [n/a]"
    return f"95% CI [{_to_percent(low)}, {_to_percent(high)}]"


def _format_selection_frequency(row: dict[str, Any]) -> str:
    """How often the delta survived a bootstrap resample, on [0,1].

    Stability selection (Meinshausen-Bühlmann): a row that appears in
    most resamples is unlikely to be a sampling artefact.
    """
    val = row.get("selection_frequency", 0.0)
    return f"{float(val):.2f}"


def _qualifier_label(row: dict[str, Any]) -> str:
    """Spell out what the boolean ``significant`` flag actually means.

    A bare ✓ reads like proof. The two real conditions matter
    individually; we surface both, so a reader can tell apart a
    "barely-screened" row from a "stable across resamples and CI
    excludes zero" one.
    """
    significant = bool(row.get("significant"))
    sel_freq = float(row.get("selection_frequency", 0.0))
    low = row.get("ci_low")
    high = row.get("ci_high")

    ci_excludes_zero: bool | None = (
        None if low is None or high is None else float(low) > 0 or float(high) < 0
    )

    parts: list[str] = []
    if sel_freq >= 0.6:
        parts.append("stable")
    if ci_excludes_zero is True:
        parts.append("CI excludes 0")
    elif ci_excludes_zero is False:
        parts.append("CI crosses 0")

    if not parts:
        return "(weak signal)"
    if significant:
        return "(" + ", ".join(parts) + ")"
    # Selected by screening but not the conjunction — say so explicitly.
    parts.append("screening only")
    return "(" + ", ".join(parts) + ")"


def _to_percent(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "?"
    return f"{v * 100:.1f}%"


__all__ = [
    "render_attribution_markdown",
    "render_attribution_terminal",
]
