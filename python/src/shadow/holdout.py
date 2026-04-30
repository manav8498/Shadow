"""Held-out trace id management.

A holdout is a trace id that has been deliberately flagged as
"acknowledged but not blocking" — typically a known flake or an
expected non-issue the team has already triaged. Each holdout carries
an owner, a reason, and a TTL after which the entry is considered
stale and warrants a fresh review.

Storage is a single JSON file at ``.shadow/holdout.json`` keyed by
trace id. Atomic write (temp + rename) so concurrent readers never
see a partially-written file.

The CRUD primitives in this module are pure functions of the file
state; they perform I/O but no networking and no clock-reading except
when explicitly passed a ``now`` argument.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

#: Bumped when the on-disk schema changes incompatibly. The current
#: layout is ``{"schema_version": 1, "entries": {trace_id: ...}}``.
SCHEMA_VERSION = 1

#: Default time-to-live for a holdout entry. Picked to match a typical
#: month-long review cadence; long enough to absorb a vacation, short
#: enough to keep the list honest.
DEFAULT_TTL_DAYS = 30

_DEFAULT_FILENAME = "holdout.json"
_TTL_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")


@dataclass(frozen=True)
class HoldoutEntry:
    """One held-out trace id with its triage metadata."""

    trace_id: str
    reason: str
    owner: str
    added_at: str  # ISO-8601 UTC, microsecond precision
    ttl_days: int = DEFAULT_TTL_DAYS

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "reason": self.reason,
            "owner": self.owner,
            "added_at": self.added_at,
            "ttl_days": self.ttl_days,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HoldoutEntry:
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            reason=str(payload.get("reason", "")),
            owner=str(payload.get("owner", "")),
            added_at=str(payload.get("added_at", "")),
            ttl_days=int(payload.get("ttl_days", DEFAULT_TTL_DAYS) or DEFAULT_TTL_DAYS),
        )

    def expires_at(self) -> datetime:
        """When this entry's review window runs out."""
        return _parse_iso(self.added_at) + timedelta(days=self.ttl_days)

    def is_stale(self, *, now: datetime) -> bool:
        """True when the review window has expired."""
        return now >= self.expires_at()

    def days_left(self, *, now: datetime) -> int:
        """How many days remain in the review window. Negative when stale."""
        delta = self.expires_at() - now
        return int(delta.total_seconds() // 86400)


@dataclass(frozen=True)
class Holdouts:
    """The full set of held-out trace ids, keyed by trace id."""

    entries: dict[str, HoldoutEntry] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": {tid: e.to_dict() for tid, e in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Holdouts:
        raw = payload.get("entries", {}) or {}
        if not isinstance(raw, dict):
            raw = {}
        entries: dict[str, HoldoutEntry] = {}
        for tid, body in raw.items():
            if isinstance(body, dict):
                entries[str(tid)] = HoldoutEntry.from_dict(body)
        return cls(
            entries=entries,
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )

    def stale_count(self, *, now: datetime) -> int:
        return sum(1 for e in self.entries.values() if e.is_stale(now=now))


# ---------------------------------------------------------------------------
# I/O — load / save
# ---------------------------------------------------------------------------


def default_path() -> Path:
    """Conventional holdout-file path under the cwd."""
    return Path.cwd() / ".shadow" / _DEFAULT_FILENAME


def load(*, path: Path | None = None) -> Holdouts:
    """Read the holdout file. Empty :class:`Holdouts` when the file
    is missing — the normal first-run case."""
    target = (path or default_path()).resolve()
    if not target.is_file():
        return Holdouts()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Don't crash a daily-glance command on a corrupt file. The
        # write path uses atomic rename so corruption only happens via
        # external tampering — surface as "no entries".
        return Holdouts()
    if not isinstance(payload, dict):
        return Holdouts()
    return Holdouts.from_dict(payload)


def save(holdouts: Holdouts, *, path: Path | None = None) -> Path:
    """Write the holdout file atomically. Creates the parent directory
    on demand."""
    target = (path or default_path()).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = json.dumps(holdouts.to_dict(), indent=2, sort_keys=False)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def add_entry(
    holdouts: Holdouts,
    *,
    trace_id: str,
    reason: str,
    owner: str,
    ttl_days: int = DEFAULT_TTL_DAYS,
    now: datetime,
) -> Holdouts:
    """Return a new :class:`Holdouts` with ``trace_id`` added or updated.

    Re-adding an existing trace id resets its ``added_at`` and updates
    ``reason`` / ``owner`` / ``ttl_days``. That's the same operation as
    :func:`reset` but with metadata changes — keeps the surface small.
    """
    entry = HoldoutEntry(
        trace_id=trace_id,
        reason=reason,
        owner=owner,
        added_at=_iso(now),
        ttl_days=ttl_days,
    )
    new = dict(holdouts.entries)
    new[trace_id] = entry
    return Holdouts(entries=new, schema_version=holdouts.schema_version)


def remove_entry(holdouts: Holdouts, trace_id: str) -> tuple[Holdouts, bool]:
    """Drop ``trace_id`` from the set. Returns the new
    :class:`Holdouts` and a ``found`` flag so callers can report
    "not in list" without re-querying.
    """
    if trace_id not in holdouts.entries:
        return holdouts, False
    new = {k: v for k, v in holdouts.entries.items() if k != trace_id}
    return (
        Holdouts(entries=new, schema_version=holdouts.schema_version),
        True,
    )


def reset_entry(
    holdouts: Holdouts,
    trace_id: str,
    *,
    now: datetime,
) -> tuple[Holdouts, bool]:
    """Restart the review window for an existing entry without
    changing its reason / owner / TTL. Returns ``(holdouts, found)``.
    """
    existing = holdouts.entries.get(trace_id)
    if existing is None:
        return holdouts, False
    fresh = HoldoutEntry(
        trace_id=existing.trace_id,
        reason=existing.reason,
        owner=existing.owner,
        added_at=_iso(now),
        ttl_days=existing.ttl_days,
    )
    new = dict(holdouts.entries)
    new[trace_id] = fresh
    return (
        Holdouts(entries=new, schema_version=holdouts.schema_version),
        True,
    )


# ---------------------------------------------------------------------------
# `--ttl` parsing
# ---------------------------------------------------------------------------


def parse_ttl(raw: str) -> int:
    """Parse a TTL string like ``"30d"`` / ``"12h"`` / ``"45m"`` into days.

    Sub-day units are rounded up to one day so a ``--ttl 1h`` doesn't
    silently become a zero-day TTL (which would mark every entry stale
    immediately). Days are the only unit that round-trips losslessly,
    so the public default is to use them.
    """
    m = _TTL_RE.match(raw)
    if m is None:
        raise ValueError(
            f"could not parse `--ttl {raw}`. Use a value like " "`30d`, `12h`, `45m`, or `60s`."
        )
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "d":
        return max(n, 1)
    seconds = {"s": 1, "m": 60, "h": 3600}[unit]
    days = (n * seconds + 86399) // 86400  # round up
    return max(days, 1)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _iso(now: datetime) -> str:
    """ISO-8601 UTC with microsecond precision and trailing ``Z``."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _parse_iso(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp into a tz-aware UTC :class:`datetime`."""
    raw = timestamp.rstrip("Z")
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        # Sentinel that's always in the past so callers see "stale".
        return datetime(1970, 1, 1, tzinfo=UTC)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    return t.astimezone(UTC)


def relative_added(timestamp: str, *, now: datetime) -> str:
    """Format ``"12d ago"`` / ``"3h ago"`` / ``"just now"`` for the table."""
    t = _parse_iso(timestamp)
    delta = now - t
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


# ---------------------------------------------------------------------------
# Visual rendering — kept here so the module is self-contained
# ---------------------------------------------------------------------------


def render(holdouts: Holdouts, *, now: datetime, console: Any | None = None) -> None:
    """Print the holdout set as a Rich panel.

    Imports Rich lazily so callers that only need data (CI scripts,
    JSON exporters) don't pay the cost.
    """
    from rich.box import HEAVY
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    target = console or Console()
    if not holdouts.entries:
        body = Group(
            Text("No held-out trace ids.", style="bold"),
            Text(""),
            Text(
                "Add one with `shadow holdout add <trace-id> " "--reason ... --owner @you`.",
                style="dim",
            ),
        )
        target.print(
            Panel(
                body,
                border_style="dim",
                box=HEAVY,
                padding=(1, 2),
                title=Text(" shadow holdout ", style="bold"),
                title_align="left",
            )
        )
        return

    table = Table(box=None, padding=(0, 2), show_header=True, header_style="dim")
    table.add_column("trace_id", style="cyan", no_wrap=True)
    table.add_column("owner", no_wrap=True)
    table.add_column("added", no_wrap=True, style="dim")
    table.add_column("ttl", no_wrap=True, style="dim")
    table.add_column("days_left", no_wrap=True)
    table.add_column("reason")

    stale = 0
    # Sort: stale first (most urgent), then most-recently-added.
    rows = sorted(
        holdouts.entries.values(),
        key=lambda e: (not e.is_stale(now=now), e.added_at),
        reverse=True,
    )
    for e in rows:
        if e.is_stale(now=now):
            stale += 1
            days_text = Text("STALE", style="bold red")
        else:
            n = e.days_left(now=now)
            days_text = Text(f"{n}d", style="green" if n > 7 else "yellow")
        table.add_row(
            e.trace_id,
            e.owner or "—",
            relative_added(e.added_at, now=now),
            f"{e.ttl_days}d",
            days_text,
            e.reason or "—",
        )

    sections: list[Any] = [table, Text("")]
    if stale > 0:
        warn = Text("  ")
        warn.append(
            f"{stale} stale entr{'y' if stale == 1 else 'ies'} — review overdue",
            style="bold red",
        )
        sections.append(warn)
    else:
        ok = Text("  ")
        ok.append("all entries within their review window", style="dim italic")
        sections.append(ok)

    target.print(
        Panel(
            Group(*sections),
            border_style="bold",
            box=HEAVY,
            padding=(1, 2),
            title=Text(" shadow holdout ", style="bold"),
            title_align="left",
        )
    )
