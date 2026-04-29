"""Pure aggregation logic for ``shadow ledger``.

Takes a list of :class:`shadow.ledger.LedgerEntry` records and produces
a :class:`LedgerView` summarising the recent state — pass rate, Wilson
confidence interval, most-concerning entry, date range.

No I/O, no Rich, no clock. The ``now`` argument is always explicit so
tests can pin time without monkey-patching :mod:`datetime`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from shadow.ledger.store import LedgerEntry

#: Concerning tiers, ranked by severity. Higher number = more concerning.
#: ``ship`` and unknown tiers fall through with negative ranks below.
_TIER_RANK: dict[str, int] = {
    "probe": 0,
    "hold": 1,
    "stop": 2,
}

#: 95% confidence z-score for the Wilson interval. Hard-coded because
#: the daily glance always reports 95% — callers that want a different
#: CI should aggregate themselves.
_WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class PassRate:
    """Anchor pass rate with a Wilson-interval 95% confidence band.

    Wilson over normal-approximation because it remains honest at small
    n — the normal approximation can produce CI bounds outside ``[0, 1]``
    for n < 30, which is the regime most personal usage sits in.
    """

    successes: int  # number of `call` entries with tier == "ship"
    total: int  # total number of `call` entries
    rate: float  # successes / total, or NaN when total == 0
    ci_low: float
    ci_high: float

    @property
    def display_rate(self) -> str:
        """Format the rate as a percentage for display."""
        if self.total == 0:
            return "—"
        return f"{int(round(self.rate * 100))}%"

    @property
    def display_ci(self) -> str:
        """Format the 95% CI as ``[low%, high%]`` for display."""
        if self.total == 0:
            return "—"
        return f"[{int(round(self.ci_low * 100))}%, {int(round(self.ci_high * 100))}%]"


@dataclass(frozen=True)
class LedgerView:
    """The full output of :func:`compute_view`."""

    entries: list[LedgerEntry] = field(default_factory=list)
    pass_rate: PassRate = field(default_factory=lambda: PassRate(0, 0, float("nan"), 0.0, 0.0))
    most_concerning: LedgerEntry | None = None
    since: timedelta | None = None  # filter window applied (None = all)
    now: datetime | None = None  # reference time for "X ago" rendering


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_view(
    entries: list[LedgerEntry],
    *,
    now: datetime,
    since: timedelta | None = None,
) -> LedgerView:
    """Aggregate recent ledger entries into a :class:`LedgerView`.

    Parameters
    ----------
    entries:
        The full list of recent entries from :func:`read_recent`. The
        function does not assume sort order; it filters and aggregates
        independently.
    now:
        Reference time for the ``since`` filter and "X ago" rendering.
        Always explicit so tests can pin clock state without
        monkey-patching.
    since:
        Optional window. When supplied, entries older than ``now - since``
        are dropped before aggregation. ``None`` means "no filter".
    """
    if since is not None:
        cutoff = now - since
        entries = [e for e in entries if _parse_iso_or_zero(e.timestamp) >= cutoff]

    # Sort newest-first so downstream consumers get a stable order.
    entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

    pass_rate = _compute_pass_rate(entries)
    most_concerning = _pick_most_concerning(entries)
    return LedgerView(
        entries=entries,
        pass_rate=pass_rate,
        most_concerning=most_concerning,
        since=since,
        now=now,
    )


# ---------------------------------------------------------------------------
# Pass rate + Wilson interval
# ---------------------------------------------------------------------------


def _compute_pass_rate(entries: list[LedgerEntry]) -> PassRate:
    """Count `ship` calls / total calls, plus a Wilson 95% CI.

    Only ``call`` entries enter the count — diff and autopr entries
    don't carry a tier so they aren't part of the pass-rate concept.
    """
    calls = [e for e in entries if e.kind == "call" and e.tier is not None]
    total = len(calls)
    successes = sum(1 for e in calls if e.tier == "ship")
    if total == 0:
        return PassRate(successes=0, total=0, rate=float("nan"), ci_low=0.0, ci_high=0.0)

    rate = successes / total
    ci_low, ci_high = _wilson_interval(successes, total, _WILSON_Z_95)
    return PassRate(
        successes=successes,
        total=total,
        rate=rate,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More honest than the normal approximation at small n, and stays
    inside [0, 1]. Reference: Wilson (1927).
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    z_sq = z * z
    denominator = 1.0 + z_sq / total
    center = p + z_sq / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z_sq / (4.0 * total * total))
    low = max(0.0, (center - margin) / denominator)
    high = min(1.0, (center + margin) / denominator)
    return (low, high)


# ---------------------------------------------------------------------------
# Most-concerning entry
# ---------------------------------------------------------------------------


def _tier_rank(tier: str | None) -> int:
    """How concerning a tier is. Higher = worse.

    Returns ``-1`` for ``ship`` / ``None`` / unknown so they never beat
    a real concerning tier in the most-concerning ranking.
    """
    if tier is None:
        return -1
    return _TIER_RANK.get(tier, -1)


def _pick_most_concerning(entries: list[LedgerEntry]) -> LedgerEntry | None:
    """Pick the call entry that most warrants attention.

    Rules:
        1. Only ``call`` entries with a tier are considered.
        2. Tier rank is the primary sort key (``stop`` wins).
        3. Ties broken by most-recent timestamp.
        4. ``ship`` entries never count as concerning.
    """
    best: LedgerEntry | None = None
    best_key = (-3, "")  # (tier_rank, timestamp)
    for e in entries:
        if e.kind != "call" or e.tier is None or e.tier == "ship":
            continue
        rank = _tier_rank(e.tier)
        # Primary: highest rank wins. Tie-break: latest timestamp.
        key = (rank, e.timestamp)
        if key > best_key:
            best_key = key
            best = e
    return best


# ---------------------------------------------------------------------------
# `--since` parsing
# ---------------------------------------------------------------------------


_SINCE_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")


def parse_since(raw: str) -> timedelta:
    """Parse a ``--since`` argument like ``"7d"`` / ``"3h"`` / ``"30m"`` / ``"60s"``.

    Returns the equivalent :class:`timedelta`. Raises :class:`ValueError`
    on garbled input so callers (the CLI command) can surface a clear
    message instead of silently using a default.
    """
    m = _SINCE_RE.match(raw)
    if m is None:
        raise ValueError(
            f"could not parse `--since {raw}`. Use a value like " "`7d`, `3h`, `30m`, or `60s`."
        )
    n = int(m.group(1))
    unit = m.group(2)
    seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return timedelta(seconds=n * seconds)


# ---------------------------------------------------------------------------
# Relative-time rendering helper (used by render.py)
# ---------------------------------------------------------------------------


def relative_time(timestamp: str, *, now: datetime) -> str:
    """Format a timestamp as ``"2h ago"`` / ``"3d ago"`` / ``"just now"``.

    Used by the renderer so the table column reads at a glance.
    Returns ``"—"`` for unparseable timestamps rather than crashing.
    """
    t = _parse_iso_or_zero(timestamp)
    if t == _ZERO:
        return "—"
    delta = now - t
    secs = delta.total_seconds()
    if secs < 0:
        return "just now"  # clock skew; round to "just now" rather than future
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_ZERO = datetime(1970, 1, 1, tzinfo=UTC)


def _parse_iso_or_zero(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp, falling back to epoch-zero on error.

    Falling back to a sentinel rather than raising keeps the daily
    glance robust against any corruption that slipped past the read
    layer.
    """
    if not timestamp:
        return _ZERO
    raw = timestamp.rstrip("Z")
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        return _ZERO
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    return t.astimezone(UTC)
