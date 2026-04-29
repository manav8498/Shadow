"""Rich rendering for ``shadow call`` — the visual layer.

The decision logic lives in :mod:`shadow.call.decide` and is rendering-
free. This module turns a :class:`shadow.call.decide.CallResult` into a
terminal panel that's clean to read at a glance.

Design notes:

* Each tier has a colour and a glyph picked to read well in 80-column
  terminals on light and dark backgrounds. Glyphs degrade to ASCII
  on consoles without UTF-8 support.
* The panel is composed from Rich's :class:`Group` so each section
  (summary, driver, axes, next steps) sits in its own visual band.
  This keeps the output legible when piped through tools that strip
  panel borders.
* The renderer never mutates the call result. Callers can render the
  same :class:`CallResult` multiple times.
"""

from __future__ import annotations

from typing import Any

from rich.box import HEAVY
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from shadow.call.decide import CallResult, Confidence, Tier

# ---------------------------------------------------------------------------
# Visual palette
# ---------------------------------------------------------------------------

_TIER_STYLE: dict[Tier, tuple[str, str, str]] = {
    # tier  ->  (label, glyph, colour)
    Tier.SHIP: ("SHIP", "✓", "green"),
    Tier.HOLD: ("HOLD", "⚠", "yellow"),
    Tier.PROBE: ("PROBE", "◎", "cyan"),
    Tier.STOP: ("STOP", "⛔", "red"),
}

_CONFIDENCE_STYLE: dict[Confidence, str] = {
    Confidence.FIRM: "italic green",
    Confidence.FAIR: "italic yellow",
    Confidence.FAINT: "italic dim",
}


def render_call(call: CallResult, console: Console | None = None) -> None:
    """Print a :class:`CallResult` as a Rich panel to ``console``.

    Pass ``console=None`` to use a fresh stdout console — the default
    behaviour for the CLI command. Pass an explicit console (e.g. one
    captured in tests) to direct output elsewhere.
    """
    target = console or Console()
    target.print(_build_panel(call))


def _build_panel(call: CallResult) -> Panel:
    """Compose the full panel from individual section renderers."""
    label, glyph, colour = _TIER_STYLE[call.tier]

    sections: list[Any] = []

    # ---- Header line: tier glyph + label, big & bold -----------------
    header = Text(no_wrap=True)
    header.append("CALL    ", style="bold dim")
    header.append(f"{glyph}  {label}", style=f"bold {colour}")
    sections.append(header)
    sections.append(Text(""))

    # ---- Identity block: anchor / candidate / pair count ------------
    sections.append(_identity_block(call))
    sections.append(Text(""))

    # ---- Driver block (only when one was extracted) -----------------
    if call.driver is not None:
        sections.append(_driver_block(call))
        sections.append(Text(""))

    # ---- Worst axes table (only when there's any movement) ----------
    if call.worst_axes:
        sections.append(_worst_axes_table(call))
        sections.append(Text(""))

    # ---- Reasons (always present — bullet list) ---------------------
    if call.reasons:
        sections.append(_reasons_block(call))
        sections.append(Text(""))

    # ---- What to do (only when actionable) --------------------------
    if call.suggestions:
        sections.append(_suggestions_block(call))

    return Panel(
        Group(*sections),
        border_style=colour,
        box=HEAVY,
        padding=(1, 2),
        title=Text(" shadow call ", style=f"bold {colour}"),
        title_align="left",
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _identity_block(call: CallResult) -> Table:
    """Two-column key/value identity strip at the top of the panel."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("anchor", _trace_id_text(call.anchor_id))
    table.add_row("candidate", _trace_id_text(call.candidate_id))

    pair_text = Text(str(call.pair_count))
    if call.pair_count < 5:
        pair_text.append("   (low — call is directional)", style="dim italic")
    table.add_row("pairs", pair_text)
    return table


def _driver_block(call: CallResult) -> Group:
    """Driver section — surfaces the dominant cause of any regression."""
    assert call.driver is not None  # invariant on callsite
    driver = call.driver

    title = Text("Driver", style="bold")

    headline = Text("  ")
    headline.append(driver.summary, style="bold")

    body = Group(title, headline)
    if driver.detail:
        body = Group(
            body,
            Text(f"  {driver.detail}", style="dim"),
        )

    confidence_line = Text("  confidence  ", style="dim")
    confidence_line.append(driver.confidence.value, style=_CONFIDENCE_STYLE[driver.confidence])
    if driver.turn is not None:
        confidence_line.append(f"   turn  {driver.turn}", style="dim")

    return Group(body, confidence_line)


def _worst_axes_table(call: CallResult) -> Group:
    """Compact, monospace-aligned table of the top regressed axes."""
    title = Text("Worst axes", style="bold")

    table = Table(box=None, padding=(0, 2), show_header=False, expand=False)
    table.add_column("", style="dim", no_wrap=True)
    table.add_column("", justify="right", no_wrap=True)
    table.add_column("", justify="left", no_wrap=True)
    table.add_column("", no_wrap=True)
    table.add_column("", no_wrap=True)

    for line in call.worst_axes:
        sev_style = {
            "severe": "red",
            "moderate": "yellow",
            "minor": "dim",
            "none": "dim",
        }.get(line.severity, "dim")
        table.add_row(
            line.axis,
            _signed(line.delta, line.axis),
            f"[{line.ci_low:+.3f}, {line.ci_high:+.3f}]",
            Text(line.severity, style=sev_style),
            Text(line.confidence.value, style=_CONFIDENCE_STYLE[line.confidence]),
        )
    return Group(title, table)


def _reasons_block(call: CallResult) -> Group:
    """Bullet list of the human-readable rationale for the tier."""
    title = Text("Why", style="bold")
    body = Group(title, *[Text(f"  • {r}") for r in call.reasons])
    return body


def _suggestions_block(call: CallResult) -> Group:
    """Inline shell commands the reviewer can copy and run."""
    title = Text("What to do", style="bold")
    rows: list[Text] = []
    for cmd in call.suggestions:
        if cmd.startswith("shadow"):
            line = Text("  $ ", style="dim")
            line.append(cmd, style="bold yellow")
            rows.append(line)
        else:
            # Non-command instruction (e.g. "record more pairs").
            rows.append(Text(f"  • {cmd}", style="italic"))
    return Group(title, *rows)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _trace_id_text(short_id: str) -> Text:
    """Format a short trace id with the cyan data-accent style."""
    if not short_id:
        return Text("(none)", style="dim italic")
    return Text(short_id, style="cyan")


def _signed(value: float, axis: str) -> str:
    """Format a signed delta with the unit appropriate to the axis.

    Latency comes in milliseconds; verbosity in tokens; everything else
    is dimensionless. The unit suffix keeps the table readable without
    forcing the renderer to emit a separate column.
    """
    if axis == "latency":
        return f"{value:+.0f} ms"
    if axis == "verbosity":
        return f"{value:+.0f} tokens"
    if axis == "cost":
        return f"${value:+.4f}"
    return f"{value:+.3f}"
