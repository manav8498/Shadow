"""Rich rendering for ``shadow heal`` — classifier panel.

Mirrors the visual language of ``shadow call``: titled bordered box,
tier label up top, evidence underneath, action footer. The action
footer for this phase deliberately does not include any commands that
would *take* heal action — only commands that help the reviewer
inspect or pin the regression manually.
"""

from __future__ import annotations

from rich.box import HEAVY
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from shadow.heal.classify import HealDecision, HealTier

#: Tier label, glyph, and Rich colour.
_TIER_STYLE: dict[HealTier, tuple[str, str, str]] = {
    HealTier.HEAL: ("HEAL", "✓", "green"),
    HealTier.PROPOSE: ("PROPOSE", "◇", "yellow"),
    HealTier.HOLD: ("HOLD", "⛔", "red"),
}


def render_decision(decision: HealDecision, console: Console | None = None) -> None:
    """Print a :class:`HealDecision` as a Rich panel."""
    target = console or Console()
    target.print(_build_panel(decision))


def _build_panel(decision: HealDecision) -> Panel:
    label, glyph, colour = _TIER_STYLE[decision.tier]

    sections: list[RenderableType] = []

    # ---- Header --------------------------------------------------
    header = Text(no_wrap=True)
    header.append("HEAL    ", style="bold dim")
    header.append(f"{glyph}  {label}", style=f"bold {colour}")
    sections.append(header)
    sections.append(Text(""))

    # ---- Identity strip -----------------------------------------
    sections.append(_identity_block(decision))
    sections.append(Text(""))

    # ---- Checks --------------------------------------------------
    sections.append(_checks_block(decision))
    sections.append(Text(""))

    # ---- Rationale -----------------------------------------------
    if decision.rationale:
        sections.append(_rationale_block(decision))
        sections.append(Text(""))

    # ---- Next-step suggestions ----------------------------------
    sections.append(_next_steps_block(decision))

    return Panel(
        Group(*sections),
        border_style=colour,
        box=HEAVY,
        padding=(1, 2),
        title=Text(" shadow heal ", style=f"bold {colour}"),
        title_align="left",
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _identity_block(decision: HealDecision) -> Table:
    """Two-column trace-id strip, same layout as ``shadow call``."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("anchor", _id_text(decision.anchor_id))
    table.add_row("candidate", _id_text(decision.candidate_id))
    pair_text = Text(str(decision.pair_count))
    if decision.pair_count < 5:
        pair_text.append("   (low — heal not eligible)", style="dim italic")
    table.add_row("pairs", pair_text)
    return table


def _checks_block(decision: HealDecision) -> Group:
    """Ordered list of gate checks with pass/fail glyphs.

    Each check renders as ``✓ <name>`` (green) or ``✗ <name>`` (red)
    followed by the detail string. The order matches the classifier's
    decision path so the panel reads top-to-bottom as the reviewer
    needs.
    """
    title = Text("Checks", style="bold")
    rows: list[Text] = []
    for c in decision.checks:
        line = Text("  ")
        if c.passed:
            line.append("✓ ", style="bold green")
        else:
            line.append("✗ ", style="bold red")
        line.append(c.name.replace("_", " "), style="bold")
        line.append("    ")
        line.append(c.detail, style="dim")
        rows.append(line)
    return Group(title, *rows)


def _rationale_block(decision: HealDecision) -> Group:
    """One-paragraph plain-English explanation of the tier."""
    title = Text("Why", style="bold")
    body = Text(f"  {decision.rationale}")
    return Group(title, body)


def _next_steps_block(decision: HealDecision) -> Group:
    """Suggested next commands, derived from the tier.

    Phase 9.1 is observation-only: there is no ``shadow heal --apply``
    yet. The suggestions here help the reviewer act *manually* on the
    classifier's recommendation.
    """
    title = Text("What to do", style="bold")
    rows: list[Text] = []
    if decision.tier is HealTier.HEAL:
        rows.append(
            _command_line("shadow heal <a> <b> --log    log the audit decision to the ledger")
        )
        rows.append(
            _command_line("shadow call <a> <b>    re-confirm the diff before manual action")
        )
    elif decision.tier is HealTier.PROPOSE:
        rows.append(
            _command_line("shadow autopr <a> <b> -o tests/regressions/r.yaml    pin if intentional")
        )
        rows.append(_command_line("shadow trail <candidate>    walk back to the cause"))
    else:  # HOLD
        rows.append(_command_line("shadow trail <candidate>    walk back to the cause"))
        rows.append(_command_line("shadow autopr <a> <b>    pin the regression as a policy"))
    return Group(title, *rows)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _id_text(short_id: str) -> Text:
    if not short_id:
        return Text("(none)", style="dim italic")
    return Text(short_id, style="cyan")


def _command_line(text: str) -> Text:
    """Render one suggested-command row.

    Same shape as ``shadow call``'s and ``shadow ledger``'s footers so
    the reviewer's eye doesn't have to relearn anything.
    """
    parts = text.split("    ", 1)
    cmd = parts[0]
    note = parts[1] if len(parts) > 1 else ""
    line = Text("  $ ", style="dim")
    line.append(cmd, style="bold yellow")
    if note:
        line.append("    ")
        line.append(note, style="dim")
    return line
