"""Append-only ledger store backed by per-entry JSON files.

Design choices worth flagging:

* **Atomic writes.** Every entry is written to a sibling ``.tmp`` file
  then renamed into place. ``Path.rename`` is atomic on POSIX and
  Windows (NTFS), so concurrent readers either see the old name or the
  finished entry — never a half-written file.

* **Filename uniqueness via microseconds + content ids.** The filename
  encodes the timestamp at microsecond resolution plus the short anchor
  and candidate ids so two near-simultaneous entries from different
  trace pairs can't collide. A monotonic-counter scheme would be
  cleaner but isn't needed at the volumes a single-developer flow
  produces.

* **Schema versioning.** Each entry carries an integer ``schema_version``.
  When the schema evolves we add a migration on read; old entries stay
  readable forever. v1 is the initial schema covering diff and call
  outputs.

* **Date bucketing.** Entries land under ``YYYYMMDD/`` so a directory
  listing of the ledger never gets unboundedly large and retention
  sweeps can drop whole day-buckets without scanning every file.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Bumped when the entry schema changes incompatibly. v1 covers the
#: initial diff/call shapes; future kinds (autopr, cert) are forward-
#: compatible additions to the ``kind`` enum without a version bump.
SCHEMA_VERSION = 1

#: Default ledger root, relative to the project root. Created lazily on
#: first write — never touched by import.
_DEFAULT_BASE = Path(".shadow") / "ledger"


def default_base_path() -> Path:
    """The conventional ledger directory, relative to the cwd.

    Resolved via ``cwd()`` rather than module-level so test fixtures
    that ``monkeypatch.chdir`` work as expected.
    """
    return Path.cwd() / _DEFAULT_BASE


@dataclass
class LedgerEntry:
    """One immutable record of a Shadow operation.

    Designed to be both round-trippable through JSON and forward-
    compatible: extra fields on read are preserved in ``extras`` so
    downstream consumers don't lose data when a newer Shadow writes
    fields the current one doesn't recognise.
    """

    # The canonical core. Every entry has these.
    kind: str  # "diff" | "call" | "autopr" | "cert"
    timestamp: str  # ISO-8601 in UTC, microsecond precision
    anchor_id: str  # short trace id (8-char SHA-256 prefix), or ""
    candidate_id: str  # short trace id, or ""

    # Optional rollup fields. Filled when the originating operation
    # produced one; left as default otherwise.
    tier: str | None = None  # "ship" | "hold" | "probe" | "stop"
    worst_severity: str | None = None  # "none" | "minor" | "moderate" | "severe"
    pair_count: int = 0
    driver_summary: str | None = None  # one-liner from the call layer
    primary_axis: str | None = None  # axis that drove the change
    source_command: str = ""  # e.g. "shadow call"
    extras: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Wire format. Preserves extras so unknown fields round-trip."""
        d = asdict(self)
        if not d["extras"]:
            d.pop("extras")
        # Drop None-valued optionals to keep the JSON terse.
        for k in list(d):
            if d[k] is None:
                del d[k]
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LedgerEntry:
        """Inverse of :meth:`to_dict`. Tolerates unknown fields."""
        known = {
            "kind",
            "timestamp",
            "anchor_id",
            "candidate_id",
            "tier",
            "worst_severity",
            "pair_count",
            "driver_summary",
            "primary_axis",
            "source_command",
            "schema_version",
        }
        kwargs: dict[str, Any] = {k: payload[k] for k in payload if k in known}
        # Defaults for required fields if a malformed entry slipped in.
        kwargs.setdefault("kind", "unknown")
        kwargs.setdefault("timestamp", "")
        kwargs.setdefault("anchor_id", "")
        kwargs.setdefault("candidate_id", "")
        extras = {k: payload[k] for k in payload if k not in known}
        return cls(extras=extras, **kwargs)


# ---------------------------------------------------------------------------
# Constructors from existing Shadow outputs
# ---------------------------------------------------------------------------


def entry_from_diff_report(
    report: dict[str, Any],
    *,
    source_command: str = "shadow diff",
    now: datetime | None = None,
) -> LedgerEntry:
    """Build a ``LedgerEntry`` from a diff report dict.

    The diff layer doesn't compute a tier itself, so the entry's
    ``tier`` stays None. ``worst_severity`` is read from the report's
    axis rows.
    """
    rows = report.get("rows") or []
    severity = _worst_severity(rows)
    divs = report.get("divergences") or []
    primary = (divs[0].get("primary_axis") if divs else None) or _worst_axis_name(rows)
    return LedgerEntry(
        kind="diff",
        timestamp=_now_iso(now),
        anchor_id=_short(report.get("baseline_trace_id", "")),
        candidate_id=_short(report.get("candidate_trace_id", "")),
        worst_severity=severity,
        pair_count=int(report.get("pair_count", 0) or 0),
        primary_axis=primary,
        source_command=source_command,
    )


def entry_from_call(
    call_dict: dict[str, Any],
    *,
    source_command: str = "shadow call",
    now: datetime | None = None,
) -> LedgerEntry:
    """Build a ``LedgerEntry`` from a :class:`shadow.call.CallResult` dict.

    The call layer carries the tier and a structured driver, so the
    entry is richer than a bare diff entry.
    """
    driver = call_dict.get("driver") or {}
    worst = call_dict.get("worst_axes") or []
    worst_sev = worst[0].get("severity") if worst else None
    return LedgerEntry(
        kind="call",
        timestamp=_now_iso(now),
        anchor_id=str(call_dict.get("anchor_id", "")),
        candidate_id=str(call_dict.get("candidate_id", "")),
        tier=str(call_dict.get("tier", "")) or None,
        worst_severity=worst_sev,
        pair_count=int(call_dict.get("pair_count", 0) or 0),
        driver_summary=str(driver.get("summary", "")) or None,
        primary_axis=str(driver.get("primary_axis", "")) or None,
        source_command=source_command,
    )


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def write_entry(
    entry: LedgerEntry,
    *,
    base: Path | None = None,
) -> Path:
    """Atomically write ``entry`` under ``base`` (default :func:`default_base_path`).

    Returns the resolved path of the entry. Creates the day-bucketed
    parent directory on demand.
    """
    base_path = (base or default_base_path()).resolve()
    day_dir = base_path / _day_bucket(entry.timestamp)
    day_dir.mkdir(parents=True, exist_ok=True)

    filename = _filename_for(entry)
    target = day_dir / filename
    tmp = target.with_suffix(target.suffix + ".tmp")

    payload = json.dumps(entry.to_dict(), indent=2, sort_keys=False)
    tmp.write_text(payload, encoding="utf-8")
    # POSIX/Windows-atomic on the same filesystem.
    os.replace(tmp, target)
    return target


def read_recent(
    *,
    base: Path | None = None,
    limit: int = 50,
) -> list[LedgerEntry]:
    """Read the most-recent ledger entries, newest first.

    Walks the date-bucketed directory tree in reverse-chronological
    order so the limit kicks in early and we don't read past it.
    Returns an empty list if the ledger directory doesn't exist —
    that's the normal case for users who haven't opted into logging.
    """
    base_path = (base or default_base_path()).resolve()
    if not base_path.is_dir():
        return []

    out: list[LedgerEntry] = []
    # Day buckets sort lexicographically; reverse for newest-first.
    for day in sorted((p for p in base_path.iterdir() if p.is_dir()), reverse=True):
        files = sorted(
            (f for f in day.iterdir() if f.suffix == ".json" and not f.name.endswith(".tmp")),
            reverse=True,
        )
        for f in files:
            try:
                payload = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # Skip unreadable entries — don't crash a daily-glance
                # command on a single broken file. Future versions
                # might collect these into a "warnings" channel.
                continue
            out.append(LedgerEntry.from_dict(payload))
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso(now: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp with microsecond precision.

    Microsecond precision keeps two near-simultaneous writes from
    colliding on the filename helper below.
    """
    t = now or datetime.now(UTC)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


_TIMESTAMP_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T")


def _day_bucket(timestamp: str) -> str:
    """Map an ISO timestamp to the YYYYMMDD bucket directory name."""
    m = _TIMESTAMP_DAY_RE.match(timestamp)
    if not m:
        return "00000000"  # bucket for malformed timestamps
    return f"{m.group(1)}{m.group(2)}{m.group(3)}"


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]")


def _filename_for(entry: LedgerEntry) -> str:
    """Filename for one entry — collision-safe, sortable.

    Format: ``HHMMSS-microseconds-<kind>-<anchor8>-<candidate8>.json``.
    Lexicographic sort within a day bucket reproduces chronological
    order, which is what :func:`read_recent` relies on.
    """
    # Pull the time-of-day portion out of the ISO timestamp.
    # Falls through to "000000" for malformed input rather than crashing.
    ts = entry.timestamp
    time_part = "000000-000000"
    if "T" in ts:
        head, tail = ts.split("T", 1)
        # tail = HH:MM:SS.ffffffZ
        time_only = tail.rstrip("Z").replace(":", "").replace(".", "-")
        if time_only:
            time_part = time_only
    safe_kind = _FILENAME_SAFE_RE.sub("", entry.kind) or "entry"
    a = (_FILENAME_SAFE_RE.sub("", entry.anchor_id) or "0")[:8]
    c = (_FILENAME_SAFE_RE.sub("", entry.candidate_id) or "0")[:8]
    return f"{time_part}-{safe_kind}-{a}-{c}.json"


def _short(content_id: str) -> str:
    """Short id (8 hex chars), tolerating ``sha256:`` prefix."""
    if not content_id:
        return ""
    if content_id.startswith("sha256:"):
        content_id = content_id[len("sha256:") :]
    return content_id[:8]


_SEVERITY_RANK = {"severe": 3, "moderate": 2, "minor": 1, "none": 0}


def _worst_severity(rows: Iterable[dict[str, Any]]) -> str | None:
    """Pick the worst severity across axis rows, or None when no axes."""
    worst: tuple[int, str] | None = None
    for r in rows:
        sev = str(r.get("severity", "none"))
        rank = _SEVERITY_RANK.get(sev, 0)
        if worst is None or rank > worst[0]:
            worst = (rank, sev)
    if worst is None:
        return None
    return worst[1]


def _worst_axis_name(rows: Iterable[dict[str, Any]]) -> str | None:
    """Axis name corresponding to the worst severity, for primary_axis fallback."""
    best_rank = -1
    best_axis: str | None = None
    for r in rows:
        sev = str(r.get("severity", "none"))
        rank = _SEVERITY_RANK.get(sev, 0)
        if rank > best_rank:
            best_rank = rank
            best_axis = str(r.get("axis", "")) or None
    return best_axis
