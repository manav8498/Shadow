"""shadow.ledger — append-only record of artifacts Shadow has produced.

Each entry captures the metadata of one Shadow operation (a diff, a
call, an autopr synthesis, a certified release) so downstream commands
that need recent state — ``shadow ledger`` for the daily glance,
``shadow trail`` for walking the causal chain, ``shadow brief`` for
broadcasting — have a stable substrate to read from.

The ledger is opt-in. Default ``shadow diff`` writes nothing; users
attach ``--log`` or call ``shadow log`` explicitly. This keeps the
existing pipeline behaviour unchanged for callers who don't want a
local artifact history.

Entries are stored under ``.shadow/ledger/<YYYYMMDD>/<HHMMSS.US>-<...>.json``
with atomic write (temp file + rename) so partial writes never expose
half-formed JSON to readers running concurrently.
"""

from shadow.ledger.render import render_ledger
from shadow.ledger.store import (
    LedgerEntry,
    default_base_path,
    entry_from_call,
    entry_from_diff_report,
    read_recent,
    write_entry,
)
from shadow.ledger.view import (
    LedgerView,
    PassRate,
    compute_view,
    parse_since,
    relative_time,
)

__all__ = [
    "LedgerEntry",
    "LedgerView",
    "PassRate",
    "compute_view",
    "default_base_path",
    "entry_from_call",
    "entry_from_diff_report",
    "parse_since",
    "read_recent",
    "relative_time",
    "render_ledger",
    "write_entry",
]
