"""Pure helpers for ``shadow listen`` — the file-save trigger.

The CLI command in :mod:`shadow.cli.app` is a thin loop around the
functions here: scan, diff, render an event per change. Splitting the
pure logic out keeps the loop testable without simulating wall-clock
time or signal handling.

No I/O beyond ``Path.stat`` and ``Path.iterdir`` — both stdlib, both
synchronous, both already deterministic enough for this use case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Files this size or smaller are skipped — empty / placeholder
#: ``.agentlog`` files (e.g. ``touch run.agentlog``) trip the loop
#: every time they're written without carrying any actual records.
_MIN_FILE_BYTES = 16


@dataclass(frozen=True)
class FileEvent:
    """One change observation produced by :func:`listen_once`.

    ``kind`` is ``"added"`` (the file wasn't present in the previous
    snapshot) or ``"modified"`` (the file existed but its mtime
    advanced). ``mtime`` is the new file's modification timestamp,
    seconds since epoch — kept on the event so callers can format
    time-of-day strings without re-stat-ing.
    """

    path: Path
    kind: str  # "added" | "modified"
    mtime: float


def scan_dir(directory: Path, *, suffix: str = ".agentlog") -> dict[Path, float]:
    """Snapshot ``{path: mtime}`` for files matching ``suffix`` in ``directory``.

    Returns an empty dict when the directory doesn't exist — that's the
    normal first-run case before the user has recorded anything.
    Recursive walks are deliberately avoided: a watched directory is
    a developer's working area, not a deep tree.
    """
    if not directory.is_dir():
        return {}
    out: dict[Path, float] = {}
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix != suffix:
            continue
        try:
            stat = entry.stat()
        except OSError:
            # File vanished between iterdir and stat — skip silently.
            continue
        if stat.st_size < _MIN_FILE_BYTES:
            continue
        out[entry.resolve()] = stat.st_mtime
    return out


def diff_states(
    previous: dict[Path, float],
    current: dict[Path, float],
) -> list[FileEvent]:
    """Compute the events between two snapshots.

    Returns events sorted by mtime ascending so a caller iterating
    them prints the oldest change first — keeps the streaming output
    linear over time. Removed paths are intentionally not emitted as
    events: the listener cares about new candidates landing, not about
    cleanup.
    """
    events: list[FileEvent] = []
    for path, mtime in current.items():
        prev_mtime = previous.get(path)
        if prev_mtime is None:
            events.append(FileEvent(path=path, kind="added", mtime=mtime))
        elif mtime > prev_mtime:
            events.append(FileEvent(path=path, kind="modified", mtime=mtime))
    events.sort(key=lambda e: e.mtime)
    return events


def listen_once(
    previous_state: dict[Path, float],
    watch_dir: Path,
    *,
    anchor_path: Path | None = None,
    suffix: str = ".agentlog",
) -> tuple[dict[Path, float], list[FileEvent]]:
    """Run one polling tick.

    Returns the new state dict and the list of events to render. The
    anchor file itself, if it lives inside the watched directory, is
    suppressed so the user doesn't see an event for the trace the
    listener uses as its baseline.
    """
    new_state = scan_dir(watch_dir, suffix=suffix)
    events = diff_states(previous_state, new_state)
    if anchor_path is not None:
        anchor_resolved = anchor_path.resolve()
        events = [e for e in events if e.path != anchor_resolved]
    return new_state, events
