"""PR-comment markdown renderer for `shadow diagnose-pr`.

Plain English first, metrics second. Voice rules:

  * Verdict on the first line, in caps.
  * Affected-trace count is a *fraction* (84 / 1,247) not a percent
    — fractions ground the reader in real numbers.
  * Cause block leads with the delta id (so the reader sees
    `system_prompt.md:47` before the math).
  * "Why it matters" is the policy-violation translation — what
    actually happens in the failing traces.
  * "Suggested fix" is a hint, not a patch (v1).
  * Verify-fix command is the call to action.

The hidden HTML marker `<!-- shadow-diagnose-pr -->` is what the
GitHub Action's comment.py looks for to update an existing PR
comment in place rather than stack new ones.
"""

from __future__ import annotations

from shadow.diagnose_pr.models import CauseEstimate, DiagnosePrReport

_MARKER = "<!-- shadow-diagnose-pr -->"


def _fmt_count(n: int) -> str:
    return f"{n:,}"


def _fmt_signed(x: float) -> str:
    return f"{x:+.2f}"


def _verdict_blurb(verdict: str) -> str:
    return {
        "ship": "No behavior regression detected against the production-like trace sample.",
        "probe": "Behavior changed but the effect is uncertain (CI crosses zero).",
        "hold": "This PR changes agent behavior with measurable effect.",
        "stop": "This PR violates a critical policy and must not merge as-is.",
    }.get(verdict, "")


def _truncate_for_display(text: str, max_len: int = 160) -> str:
    """Truncate a long line for the PR comment so a 500-char prompt
    line doesn't push the cause block past the visible window."""
    text = text.rstrip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _render_cause(c: CauseEstimate) -> list[str]:
    """Render the dominant cause when the evidence supports crowning one.

    Two sub-modes:
      * Causal proof — bootstrap CI was computed and excludes zero
        (`confidence == 1.0` AND `ci_low/ci_high` are present). Use the
        confident wording "appears to be the main cause."
      * Single-delta inference — only one config delta exists, so
        attribution is identifying-by-construction (`confidence == 1.0`,
        no CI). Same confident wording is fine.
      * Otherwise — soften to "is the most likely candidate" so we don't
        overclaim against weak attribution evidence.

    When `file_path` + `line_no` are populated (prompt deltas with
    `--baseline-ref` blame), the headline becomes
    `prompts/refund.md:17 appears to be the main cause.` and a
    "Removed:" / "Added:" code block lands directly below — readers
    see the regression-causing instruction without leaving the PR
    page.
    """
    confident = c.confidence >= 1.0
    # Prefer file:line when available — that's the key detail for
    # prompt regressions.
    headline_id = (
        f"{c.file_path}:{c.line_no}"
        if c.file_path is not None and c.line_no is not None
        else c.delta_id
    )
    headline = (
        f"`{headline_id}` appears to be the main cause."
        if confident
        else f"`{headline_id}` is the most likely candidate."
    )
    lines = [
        "### Dominant cause",
        "",
        headline,
        "",
    ]
    if c.removed_text or c.added_text:
        lines.append("```diff")
        if c.removed_text is not None:
            lines.append(f"- {_truncate_for_display(c.removed_text)}")
        if c.added_text is not None:
            lines.append(f"+ {_truncate_for_display(c.added_text)}")
        lines.append("```")
        lines.append("")
    lines.append(f"- Axis: `{c.axis}`")
    lines.append(f"- ATE: `{_fmt_signed(c.ate)}`")
    if c.ci_low is not None and c.ci_high is not None:
        lines.append(f"- 95% CI: `[{c.ci_low:.2f}, {c.ci_high:.2f}]`")
    if c.e_value is not None:
        lines.append(f"- E-value: `{c.e_value:.1f}`")
    return lines


def _render_likely_candidates(causes: list[CauseEstimate]) -> list[str]:
    """Render a candidate-list section when no single cause can be crowned.

    Fired by `runner.py` when `pick_dominant()` returns None — every
    cause is tied at the top of the |ATE|*confidence ranking, so the
    honest framing is "any one of these N candidates could be it."
    The list is capped at 5; we sort by confidence desc, then delta id
    for determinism.
    """
    ranked = sorted(causes, key=lambda c: (-c.confidence, c.delta_id))[:5]
    lines = [
        "### Likely cause candidates",
        "",
        "Multiple changes in this PR plausibly explain the regression — no "
        "single one stands out from the others on the available evidence. "
        "Re-run with `--backend live` to attribute by intervention, or "
        "narrow the PR to one config change at a time.",
        "",
    ]
    for c in ranked:
        lines.append(f"- `{c.delta_id}` (axis: `{c.axis}`)")
    return lines


def render_pr_comment(report: DiagnosePrReport) -> str:
    """Render a full PR-comment markdown body for a diagnose-pr
    report. Output ends in a newline."""
    out: list[str] = [_MARKER, ""]
    out.append(f"## Shadow verdict: {report.verdict.upper()}")
    out.append("")

    blurb = _verdict_blurb(report.verdict)
    if blurb:
        out.append(blurb)
        out.append("")

    if report.total_traces > 0:
        out.append(
            f"This PR changes agent behavior on **{_fmt_count(report.affected_traces)}**"
            f" / **{_fmt_count(report.total_traces)}** production-like traces."
        )
        out.append("")

    if "low_power" in report.flags:
        out.append(
            "> :warning: **Low statistical power** — fewer than 30 traces in the sample. "
            "Treat the verdict as advisory; widen `--max-traces` for more confidence."
        )
        out.append("")

    if "synthetic_mock" in report.flags:
        out.append(
            "> :information_source: **Synthetic mock backend.** Cause magnitudes "
            "below come from a deterministic per-delta heuristic (not real LLM "
            "behavior). Re-run with `--backend live` for a grounded estimate."
        )
        out.append("")

    if report.verdict == "probe":
        out.append(
            "_Verdict is `probe` because affected traces exist but the causal effect "
            "is uncertain (low confidence / CI crosses zero). Investigate before merge._"
        )
        out.append("")

    if report.dominant_cause is not None:
        out.extend(_render_cause(report.dominant_cause))
        out.append("")
    elif report.top_causes:
        # No single cause crowned but candidates exist — list them.
        # This avoids the "behavior changed but the PR comment says
        # nothing about why" silent-failure mode of v0.1.
        out.extend(_render_likely_candidates(report.top_causes))
        out.append("")

    if report.worst_policy_rule is not None and report.new_policy_violations > 0:
        out.append("### Why it matters")
        out.append("")
        out.append(
            f"{report.new_policy_violations} traces violate the "
            f"`{report.worst_policy_rule}` policy rule."
        )
        out.append("")

    if report.suggested_fix is not None:
        out.append("### Suggested fix")
        out.append("")
        out.append(report.suggested_fix)
        out.append("")

    out.append("### Verify the fix")
    out.append("")
    out.append("```bash")
    out.append("shadow verify-fix --report .shadow/diagnose-pr/report.json")
    out.append("```")
    out.append("")

    return "\n".join(out)


__all__ = ["render_pr_comment"]
