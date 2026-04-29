"""Tests for `shadow.ledger.view` and the `shadow ledger` CLI command.

Three layers:

* Pure aggregation — pass rate, Wilson interval, most-concerning, since
  parsing, relative-time formatting.
* Renderer — headers, tables, footers, and the empty-state panel.
* CLI — end-to-end runs via Typer's CliRunner, plus the JSON mode.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from shadow.cli.app import app
from shadow.ledger import (
    LedgerEntry,
    PassRate,
    compute_view,
    parse_since,
    relative_time,
    render_ledger,
    write_entry,
)
from shadow.ledger.view import _wilson_interval

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    *,
    kind: str = "call",
    tier: str | None = "ship",
    anchor: str = "anchor00",
    candidate: str = "cand0000",
    when: datetime,
    summary: str | None = None,
    primary: str | None = None,
    severity: str | None = None,
) -> LedgerEntry:
    """Build a one-off ledger entry with a precise timestamp."""
    return LedgerEntry(
        kind=kind,
        timestamp=when.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        anchor_id=anchor,
        candidate_id=candidate,
        tier=tier,
        worst_severity=severity,
        pair_count=10,
        driver_summary=summary,
        primary_axis=primary,
    )


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_interval_returns_zero_for_empty() -> None:
    assert _wilson_interval(0, 0, 1.96) == (0.0, 0.0)


def test_wilson_interval_widens_for_small_n() -> None:
    """At n=2 with one success, the 95% interval should be wide."""
    low, high = _wilson_interval(1, 2, 1.96)
    assert 0.0 < low < 0.2
    assert 0.8 < high < 1.0


def test_wilson_interval_tightens_for_large_n() -> None:
    """At n=100 with 50 successes the interval should bracket 0.5 closely."""
    low, high = _wilson_interval(50, 100, 1.96)
    assert 0.39 < low < 0.41
    assert 0.59 < high < 0.61


def test_wilson_interval_clamps_to_unit_range() -> None:
    """The interval must stay inside ``[0, 1]`` even at extreme inputs."""
    low, high = _wilson_interval(10, 10, 1.96)
    assert low >= 0.0
    assert high <= 1.0


# ---------------------------------------------------------------------------
# Pass rate aggregation
# ---------------------------------------------------------------------------


def test_pass_rate_zero_when_no_calls() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    view = compute_view([], now=now)
    assert view.pass_rate.total == 0
    assert view.pass_rate.successes == 0


def test_pass_rate_ignores_diff_entries() -> None:
    """Diff entries don't carry a tier — they must not enter the count."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(kind="call", tier="ship", when=now - timedelta(hours=1)),
        _entry(kind="diff", tier=None, when=now - timedelta(hours=2)),
        _entry(kind="diff", tier=None, when=now - timedelta(hours=3)),
    ]
    view = compute_view(entries, now=now)
    assert view.pass_rate.total == 1
    assert view.pass_rate.successes == 1


def test_pass_rate_counts_only_ship_as_success() -> None:
    """ship counts; hold / probe / stop don't."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(tier="ship", when=now - timedelta(hours=1)),
        _entry(tier="ship", when=now - timedelta(hours=2)),
        _entry(tier="hold", when=now - timedelta(hours=3)),
        _entry(tier="stop", when=now - timedelta(hours=4)),
        _entry(tier="probe", when=now - timedelta(hours=5)),
    ]
    view = compute_view(entries, now=now)
    assert view.pass_rate.total == 5
    assert view.pass_rate.successes == 2


def test_pass_rate_display_helpers() -> None:
    """``display_rate`` / ``display_ci`` produce the strings the renderer
    uses. Empty state renders an em-dash so the panel doesn't read NaN."""
    pr_empty = PassRate(0, 0, float("nan"), 0.0, 0.0)
    assert pr_empty.display_rate == "—"
    assert pr_empty.display_ci == "—"

    pr_real = PassRate(1, 2, 0.5, 0.1, 0.9)
    assert pr_real.display_rate == "50%"
    assert pr_real.display_ci == "[10%, 90%]"


# ---------------------------------------------------------------------------
# Most-concerning ranking
# ---------------------------------------------------------------------------


def test_most_concerning_picks_stop_over_hold() -> None:
    """``stop`` beats ``hold`` regardless of timestamp."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(tier="hold", anchor="aH", when=now - timedelta(hours=1), summary="hold one"),
        _entry(tier="stop", anchor="aS", when=now - timedelta(hours=4), summary="stop one"),
    ]
    view = compute_view(entries, now=now)
    assert view.most_concerning is not None
    assert view.most_concerning.anchor_id == "aS"


def test_most_concerning_picks_most_recent_within_tier() -> None:
    """Within a tier, the most-recent entry wins."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(tier="stop", anchor="old", when=now - timedelta(days=1)),
        _entry(tier="stop", anchor="new", when=now - timedelta(hours=1)),
    ]
    view = compute_view(entries, now=now)
    assert view.most_concerning is not None
    assert view.most_concerning.anchor_id == "new"


def test_most_concerning_returns_none_when_only_ship() -> None:
    """No concerning entries means the field is None — not a ship entry."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [_entry(tier="ship", when=now - timedelta(hours=1))]
    view = compute_view(entries, now=now)
    assert view.most_concerning is None


def test_most_concerning_skips_diff_entries() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(kind="diff", tier=None, when=now - timedelta(hours=1), severity="severe"),
    ]
    view = compute_view(entries, now=now)
    assert view.most_concerning is None


# ---------------------------------------------------------------------------
# Date filter (`--since`)
# ---------------------------------------------------------------------------


def test_compute_view_filters_by_since() -> None:
    """Entries older than now - since must be dropped."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(tier="ship", anchor="recent", when=now - timedelta(hours=1)),
        _entry(tier="ship", anchor="old", when=now - timedelta(days=10)),
    ]
    view = compute_view(entries, now=now, since=timedelta(days=1))
    assert [e.anchor_id for e in view.entries] == ["recent"]


def test_compute_view_no_filter_when_since_none() -> None:
    """``since=None`` means "show everything", regardless of age."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(tier="ship", when=now - timedelta(days=365)),
    ]
    view = compute_view(entries, now=now, since=None)
    assert len(view.entries) == 1


# ---------------------------------------------------------------------------
# parse_since
# ---------------------------------------------------------------------------


def test_parse_since_supports_all_units() -> None:
    assert parse_since("60s") == timedelta(seconds=60)
    assert parse_since("30m") == timedelta(minutes=30)
    assert parse_since("3h") == timedelta(hours=3)
    assert parse_since("7d") == timedelta(days=7)


def test_parse_since_tolerates_whitespace() -> None:
    assert parse_since("  7d  ") == timedelta(days=7)


def test_parse_since_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        parse_since("yesterday")


# ---------------------------------------------------------------------------
# Relative time formatting
# ---------------------------------------------------------------------------


def test_relative_time_returns_just_now_when_close() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    ts = (now - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    assert relative_time(ts, now=now) == "just now"


def test_relative_time_minutes() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    ts = (now - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    assert relative_time(ts, now=now) == "15m ago"


def test_relative_time_hours() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    ts = (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    assert relative_time(ts, now=now) == "3h ago"


def test_relative_time_days() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    ts = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    assert relative_time(ts, now=now) == "7d ago"


def test_relative_time_handles_garbage_input() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    assert relative_time("not a timestamp", now=now) == "—"


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def test_render_ledger_empty_state_points_at_log_commands() -> None:
    """The first-run experience must tell users how to populate the ledger."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    view = compute_view([], now=now)
    buf = Console(record=True, width=120)
    render_ledger(view, console=buf)
    out = buf.export_text()
    assert "No artifacts logged yet" in out
    assert "shadow call" in out and "--log" in out
    assert "shadow log" in out


def test_render_ledger_shows_entries_pass_rate_and_actions() -> None:
    """Populated panel must show all three: entries, pass rate with CI,
    and at least one suggested next command."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(
            tier="stop",
            anchor="ba5e1a92",
            when=now - timedelta(hours=2),
            summary="structural change at turn 0",
            primary="trajectory",
        ),
        _entry(tier="ship", anchor="c0f2d3a4", when=now - timedelta(hours=1)),
    ]
    view = compute_view(entries, now=now, since=timedelta(days=7))
    buf = Console(record=True, width=120)
    render_ledger(view, console=buf)
    out = buf.export_text()
    assert "ba5e1a92" in out  # trace ids surface
    assert "stop" in out
    assert "Anchor pass rate" in out
    assert "95% CI" in out
    assert "What to do" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_ledger_empty_renders_friendly_panel(tmp_path: Path) -> None:
    """A fresh repo with no `.shadow/ledger/` must render the empty
    panel and exit 0."""
    base = tmp_path / "ledger"  # never created
    result = runner.invoke(app, ["ledger", "--base", str(base)])
    assert result.exit_code == 0, result.output
    assert "No artifacts logged yet" in result.output


def test_cli_ledger_renders_populated_panel(tmp_path: Path) -> None:
    """When the ledger has entries the panel must surface them."""
    base = tmp_path / "ledger"
    now = datetime.now(UTC)
    entry = LedgerEntry(
        kind="call",
        timestamp=(now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        anchor_id="ba5e1a92",
        candidate_id="c0f2d3a4",
        tier="stop",
        worst_severity="severe",
        pair_count=10,
        driver_summary="structural change at turn 0",
        primary_axis="trajectory",
        source_command="shadow call",
    )
    write_entry(entry, base=base)

    result = runner.invoke(app, ["ledger", "--base", str(base)])
    assert result.exit_code == 0, result.output
    assert "ba5e1a92" in result.output
    assert "stop" in result.output


def test_cli_ledger_json_mode_emits_payload(tmp_path: Path) -> None:
    """``--json`` swaps the panel for a structured payload that pipelines
    can consume."""
    base = tmp_path / "ledger"
    now = datetime.now(UTC)
    entry = LedgerEntry(
        kind="call",
        timestamp=(now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        anchor_id="ba5e1a92",
        candidate_id="c0f2d3a4",
        tier="ship",
        pair_count=10,
        source_command="shadow call",
    )
    write_entry(entry, base=base)

    result = runner.invoke(app, ["ledger", "--base", str(base), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload["pass_rate"]["successes"] == 1
    assert payload["pass_rate"]["total"] == 1
    assert isinstance(payload["entries"], list)
    assert payload["entries"][0]["anchor_id"] == "ba5e1a92"


def test_cli_ledger_rejects_garbled_since(tmp_path: Path) -> None:
    """A bad ``--since`` value must surface a clear error."""
    result = runner.invoke(app, ["ledger", "--since", "yesterday", "--base", str(tmp_path / "x")])
    assert result.exit_code == 1
    assert "could not parse" in result.output


def test_cli_ledger_default_does_not_create_directory(tmp_path: Path, monkeypatch) -> None:
    """Zero-regression invariant: `shadow ledger` must not create
    `.shadow/ledger/` when there's nothing to read."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["ledger"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".shadow" / "ledger").exists()
