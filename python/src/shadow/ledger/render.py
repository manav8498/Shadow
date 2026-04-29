"""Rich rendering for ``shadow ledger`` — the daily glance panel.

Turns a :class:`shadow.ledger.view.LedgerView` into a terminal panel
that mirrors the visual language of ``shadow call``: a titled bordered
box, a compact table, and a small action footer with copy-pasteable
shell commands.
"""

from __future__ import annotations

from rich.box import HEAVY
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from shadow.ledger.store import LedgerEntry
from shadow.ledger.view import LedgerView, relative_time

# Glyph + colour for the call-tier column. Same palette as `shadow call`
# so the eye learns one mapping across the workflow.
_CALL_STYLE: dict[str, tuple[str, str]] = {
    "ship": ("✓ ship", "green"),
    "hold": ("⚠ hold", "yellow"),
    "probe": ("◎ probe", "cyan"),
    "stop": ("⛔ stop", "red"),
}


def render_ledger(view: LedgerView, console: Console | None = None) -> None:
    """Print a :class:`LedgerView` as a Rich panel."""
    target = console or Console()
    target.print(_build_panel(view))


def _build_panel(view: LedgerView) -> Panel:
    """Compose the full panel from individual section renderers."""
    if not view.entries:
        return _empty_panel(view)

    sections: list[RenderableType] = [
        _header_line(view),
        Text(""),
        _entries_table(view),
        Text(""),
        _aggregate_block(view),
        Text(""),
        _actions_block(view),
    ]

    return Panel(
        Group(*sections),
        border_style="bold",
        box=HEAVY,
        padding=(1, 2),
        title=Text(" shadow ledger ", style="bold"),
        title_align="left",
    )


def _empty_panel(view: LedgerView) -> Panel:
    """Friendly empty-state panel.

    Shown the first time a user runs `shadow ledger` before opting into
    logging. Tells them the one command that turns this on.
    """
    body = Group(
        Text("No artifacts logged yet.", style="bold"),
        Text(""),
        Text(
            "Shadow's ledger is opt-in — operations don't write entries unless "
            "you ask. Two ways to start:",
            style="dim",
        ),
        Text(""),
        _command_line("shadow call <anchor> <candidate> --log"),
        _command_line("shadow log <report.json>"),
        Text(""),
        Text("Run `shadow ledger` again once you have entries.", style="dim italic"),
    )
    return Panel(
        body,
        border_style="dim",
        box=HEAVY,
        padding=(1, 2),
        title=Text(" shadow ledger ", style="bold"),
        title_align="left",
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _header_line(view: LedgerView) -> Text:
    """Top line: how many entries, optional date window."""
    n = len(view.entries)
    head = Text(no_wrap=True)
    head.append(f"{n}", style="bold")
    head.append(f" entr{'y' if n == 1 else 'ies'}", style="dim")
    if view.since is not None:
        head.append("   ·   ", style="dim")
        head.append(_humanize_window(view.since), style="dim")
    return head


def _entries_table(view: LedgerView) -> Table:
    """The compact recent-artifacts table."""
    table = Table(box=None, padding=(0, 2), show_header=True, expand=False, header_style="dim")
    table.add_column("anchor", style="cyan", no_wrap=True)
    table.add_column("when", style="dim", no_wrap=True)
    table.add_column("kind", style="dim", no_wrap=True)
    table.add_column("call", no_wrap=True)
    table.add_column("summary", no_wrap=False)

    for entry in view.entries:
        when = relative_time(entry.timestamp, now=view.now) if view.now else "—"
        call_text = _format_call_cell(entry)
        summary = _format_summary_cell(entry)
        anchor = entry.anchor_id or "—"
        table.add_row(anchor, when, entry.kind, call_text, summary)
    return table


def _format_call_cell(entry: LedgerEntry) -> Text:
    """Render the call-tier column for one row."""
    if entry.kind != "call" or entry.tier is None:
        return Text("—", style="dim")
    label, colour = _CALL_STYLE.get(entry.tier, (entry.tier, "white"))
    return Text(label, style=f"bold {colour}")


def _format_summary_cell(entry: LedgerEntry) -> Text:
    """Single-line summary derived from whatever fields the entry carries."""
    if entry.driver_summary:
        return Text(entry.driver_summary)
    if entry.worst_severity and entry.worst_severity != "none":
        sev_style = {
            "severe": "red",
            "moderate": "yellow",
            "minor": "dim",
        }.get(entry.worst_severity, "dim")
        primary = entry.primary_axis or "axis"
        return Text(f"{primary} {entry.worst_severity}", style=sev_style)
    return Text("—", style="dim")


def _aggregate_block(view: LedgerView) -> Group:
    """Pass-rate + most-concerning block under the table."""
    pr = view.pass_rate
    rate_line = Text("  ")
    rate_line.append("Anchor pass rate", style="bold")
    if pr.total > 0:
        rate_line.append("   ")
        rate_line.append(f"{pr.successes} of {pr.total} calls", style="dim")
        rate_line.append("   ")
        rate_line.append(pr.display_rate, style="bold")
        rate_line.append("   95% CI ", style="dim")
        rate_line.append(pr.display_ci, style="dim italic")
    else:
        rate_line.append("   ")
        rate_line.append("no calls logged yet", style="dim italic")

    concerning = view.most_concerning
    concern_line = Text("  ")
    concern_line.append("Most concerning", style="bold")
    if concerning is None:
        concern_line.append("   ")
        concern_line.append("nothing pending — all calls clean", style="dim italic")
    else:
        when = relative_time(concerning.timestamp, now=view.now) if view.now else "—"
        label, colour = _CALL_STYLE.get(concerning.tier or "", (concerning.tier or "?", "white"))
        concern_line.append("   ")
        concern_line.append(concerning.driver_summary or "regression", style="bold")
        concern_line.append("   ")
        concern_line.append(label, style=f"bold {colour}")
        concern_line.append(f"   {when}", style="dim")

    return Group(rate_line, concern_line)


def _actions_block(view: LedgerView) -> Group:
    """Two suggested next commands, derived from view state."""
    title = Text("What to do", style="bold")

    suggestions: list[str] = []
    if view.most_concerning is not None and view.most_concerning.anchor_id:
        suggestions.append(
            f"shadow trail {view.most_concerning.anchor_id}    walk back through attribution"
        )
    if not suggestions:
        suggestions.append("shadow call <anchor> <candidate> --log    log a new entry")
    suggestions.append("shadow brief --slack    broadcast the recent state")

    rows = [_command_line(s) for s in suggestions]
    return Group(title, *rows)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _command_line(text: str) -> Text:
    """Render a single suggested-command row.

    Shell-style commands get a ``$`` prefix and a yellow accent so they
    stand out in the panel; plain instructions get a bullet.
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


def _humanize_window(td) -> str:  # type: ignore[no-untyped-def]
    """Format a timedelta as ``"last 7 days"`` / ``"last 3 hours"``."""
    secs = int(td.total_seconds())
    if secs < 60:
        return f"last {secs} second{'s' if secs != 1 else ''}"
    if secs < 3600:
        n = secs // 60
        return f"last {n} minute{'s' if n != 1 else ''}"
    if secs < 86400:
        n = secs // 3600
        return f"last {n} hour{'s' if n != 1 else ''}"
    n = secs // 86400
    return f"last {n} day{'s' if n != 1 else ''}"
