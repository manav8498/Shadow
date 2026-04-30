"""GitHub PR-comment flavoured markdown — plain-English first.

The PR comment is the single most-read Shadow surface for someone who
hasn't yet read the docs. The previous renderer led with axis-jargon
("Axis: ``trajectory`` regressed +0.333") and buried the plain-English
recommendations underneath. That ordering optimised for engineers who
already know what each axis measures, and lost everyone else — a
backend developer whose PR was just blocked would see a wall of math
and not know what to fix.

This renderer flips the order. A reviewer reads, top to bottom:

1. **Verdict line** — one sentence: "blocked", "needs review", "looks
   safe", or "no regressions". Plain English. Decision-grade.
2. **What probably broke** — the recommendations the engine already
   produces ("Refusal rate is up severely. Check for stricter system
   instructions or tighter content policies."). One bullet per
   recommendation, ordered by severity.
3. **What changed at the turn level** — the top divergences in plain
   prose ("tool set changed: removed `search_files(query)`, added
   `search_files(limit, query)`"). A reviewer can open the named turn
   and see the actual diff.
4. **The numbers** — the nine-axis table with deltas, CIs, and sample
   counts — wrapped in a `<details>` fold. The fold opens on click for
   anyone who wants to verify; everyone else can ignore it.

Same data, different reading experience. The math stays available;
it's just no longer the headline.
"""

from __future__ import annotations

from typing import Any

from shadow.report.labels import axis_label

_TIER_ORDER = ("critical", "error", "warning", "info")
_TIER_EMOJI = {
    "critical": "🛑",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
}

_AXIS_SEVERITY_TO_TIER = {
    "severe": "critical",
    "major": "error",
    "moderate": "warning",
    "minor": "info",
    "none": "info",
}

_REC_SEVERITY_ICON = {
    "critical": "🛑",
    "error": "🛑",
    "warning": "⚠️",
    "info": "ℹ️",
}

_AXIS_SEVERITY_EMOJI = {
    "severe": "🔴",
    "moderate": "🟠",
    "minor": "🟡",
    "none": "🟢",
}

_SEV_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def _verdict_line(report: dict[str, Any]) -> tuple[str, str]:
    """Pick the headline emoji + sentence for the comment.

    Returns ``(emoji, sentence)``. The sentence reads as a recommendation
    a non-expert reviewer can act on without learning Shadow's vocabulary.
    """
    rows = report.get("rows", []) or []
    worst = "none"
    for row in rows:
        sev = str(row.get("severity") or "none")
        if _SEV_RANK.get(sev, 0) > _SEV_RANK.get(worst, 0):
            worst = sev

    if worst == "severe":
        return ("🛑", "Shadow recommends: hold this PR for review.")
    if worst == "moderate":
        return ("⚠️", "Shadow recommends: review before merging.")
    if worst == "minor":
        return ("ℹ️", "Shadow flagged minor changes — likely safe to merge after a quick look.")
    return ("✅", "Shadow found no behaviour regressions.")


def _recommendations_section(report: dict[str, Any]) -> list[str]:
    """Render the engine's recommendations as the headline of the comment.

    Each recommendation already speaks plain English ("Refusal rate is
    up severely…"). We just give them the prominence they deserve.
    """
    recs = report.get("recommendations") or []
    if not recs:
        return []

    # Order by severity: critical → error → warning → info, preserving
    # original order within a tier.
    def sev_key(rec: dict[str, Any]) -> int:
        sev = str(rec.get("severity") or "info").lower()
        # Lower number = higher priority.
        return {"critical": 0, "error": 1, "warning": 2, "info": 3}.get(sev, 4)

    ordered = sorted(recs, key=sev_key)

    lines = ["### What probably broke", ""]
    for rec in ordered:
        sev = str(rec.get("severity") or "info").lower()
        icon = _REC_SEVERITY_ICON.get(sev, "ℹ️")
        message = str(rec.get("message") or "").strip()
        rationale = str(rec.get("rationale") or "").strip()
        if message:
            lines.append(f"{icon} **{message}**")
        if rationale:
            lines.append(f"  {rationale}")
        lines.append("")
    return lines


def _divergences_section(report: dict[str, Any]) -> list[str]:
    """Render the top divergences as turn-level prose.

    Each divergence is already a plain-English sentence ("tool set
    changed: removed X, added Y"). We name the turn so a reviewer can
    open it directly.
    """
    divergences = report.get("divergences") or []
    if not divergences and report.get("first_divergence"):
        divergences = [report["first_divergence"]]
    if not divergences:
        return []

    lines = ["### What changed at the turn level", ""]
    for dv in divergences[:3]:
        bt = dv.get("baseline_turn", 0)
        ct = dv.get("candidate_turn", 0)
        explanation = str(dv.get("explanation") or "").strip()
        if explanation:
            lines.append(f"- **Turn {ct}** (vs baseline turn {bt}): {explanation}")
    if len(divergences) > 3:
        lines.append(f"- _… and {len(divergences) - 3} more — see numbers below._")
    lines.append("")
    return lines


def _axis_table_in_details(report: dict[str, Any]) -> list[str]:
    """The nine-axis numbers, wrapped in a `<details>` fold.

    Uses plain-English column headings (``response meaning``,
    ``tool calls``, …) instead of the internal axis names so the
    reader doesn't need to consult the spec to read the table.
    """
    rows = report.get("rows", []) or []
    if not rows:
        return []

    pair_count = int(report.get("pair_count", 0))

    lines: list[str] = []
    lines.append("<details>")
    lines.append(
        "<summary>Show the nine-axis details (numbers, confidence intervals, samples)</summary>"
    )
    lines.append("")
    if 0 < pair_count < 5:
        lines.append(
            f"> ⚠️  Based on {pair_count} paired response(s). Severities are "
            "directional. Record 10+ turns for stable confidence intervals."
        )
        lines.append("")

    # Detect whether `low_power` is universal — if every row has it,
    # mention it once in the banner above and drop the per-row noise.
    flag_universal = (
        "low_power" if rows and all("low_power" in (r.get("flags") or []) for r in rows) else None
    )

    lines.append("| signal | baseline | candidate | change | 95% CI | severity | n |")
    lines.append("|--------|---------:|----------:|-------:|:-------|:---------|---:|")
    for row in rows:
        sev = str(row.get("severity") or "none")
        emoji = _AXIS_SEVERITY_EMOJI.get(sev, "")
        flags = row.get("flags") or []
        # Drop the universal flag from per-row display; keep any
        # row-specific flags (e.g. `ci_crosses_zero`, `no_pricing`).
        per_row_flags = [f for f in flags if f != flag_universal]
        # Severity cell shows emoji + label; flags column would be
        # noise for a non-expert, so we drop it entirely from the
        # user-facing table. Anyone who needs the flag set can read
        # the JSON report.
        del per_row_flags  # collected only to show we considered it; not rendered

        lines.append(
            f"| {axis_label(str(row.get('axis', '')))} "
            f"| {row.get('baseline_median', 0.0):.3f} "
            f"| {row.get('candidate_median', 0.0):.3f} "
            f"| {row.get('delta', 0.0):+.3f} "
            f"| [{row.get('ci95_low', 0.0):+.2f}, {row.get('ci95_high', 0.0):+.2f}] "
            f"| {emoji} {sev} "
            f"| {row.get('n', 0)} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


def _drill_down_in_details(report: dict[str, Any]) -> list[str]:
    """Top regressive pairs, in a separate `<details>` fold.

    Reviewers who want to understand which specific turn pair drove
    the regression can open this. The default reader doesn't need it.
    """
    drill_down = report.get("drill_down") or []
    if not drill_down:
        return []

    lines = ["<details>"]
    lines.append(
        "<summary>Show top regressive turn pairs (which pair drove the regression)</summary>"
    )
    lines.append("")
    for row in drill_down[:5]:
        idx = row.get("pair_index", 0)
        axis = axis_label(str(row.get("dominant_axis", "")))
        score = float(row.get("regression_score", 0.0))
        lines.append(f"- **pair #{idx}** — biggest mover: {axis} (score {score:.2f})")
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
            sub_axis = axis_label(str(s.get("axis", "")))
            lines.append(f"  - {sub_axis}: {bv:.2f} → {cv:.2f} (Δ {delta:+.2f})")
    if len(drill_down) > 5:
        lines.append(f"- _… and {len(drill_down) - 5} more pair(s)._")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


def render_github_pr(report: dict[str, Any]) -> str:
    """Produce a PR-friendly markdown comment.

    Reading order, top to bottom:
      1. Verdict line (one sentence, decision-grade).
      2. "What probably broke" — plain-English recommendations.
      3. "What changed at the turn level" — prose descriptions of
         the top divergences with turn references.
      4. Nine-axis table (in `<details>`) — numbers for verification.
      5. Top regressive pairs (in `<details>`) — drill-down.

    Sections 2-5 each only render when their underlying data exists,
    so a "no regressions" report stays short.
    """
    emoji, sentence = _verdict_line(report)
    pair_count = int(report.get("pair_count", 0))

    lines: list[str] = []
    lines.append(f"## {emoji} {sentence}")
    lines.append("")
    if pair_count > 0:
        suffix = "s" if pair_count != 1 else ""
        lines.append(
            f"_Compared {pair_count} response pair{suffix} between baseline and candidate._"
        )
        lines.append("")

    lines.extend(_recommendations_section(report))
    lines.extend(_divergences_section(report))
    lines.extend(_axis_table_in_details(report))
    lines.extend(_drill_down_in_details(report))

    lines.append("---")
    lines.append("")
    lines.append("_Generated by [Shadow](https://github.com/manav8498/Shadow)._")
    lines.append("")
    return "\n".join(lines)
