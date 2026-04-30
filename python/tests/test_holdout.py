"""Tests for `shadow.holdout` and the `shadow holdout` CLI sub-app.

Three layers:

* Pure CRUD logic — add / remove / reset / load / save / parse_ttl /
  is_stale / days_left.
* Renderer — empty-state and populated panels (Rich output captured).
* CLI — every subcommand against a tmp-pathed holdout file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from shadow.cli.app import app
from shadow.holdout import (
    DEFAULT_TTL_DAYS,
    HoldoutEntry,
    Holdouts,
    add_entry,
    load,
    parse_ttl,
    relative_added,
    remove_entry,
    render,
    reset_entry,
    save,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    *,
    trace_id: str = "ba5e1a92",
    reason: str = "known flake",
    owner: str = "@alice",
    added: datetime,
    ttl_days: int = DEFAULT_TTL_DAYS,
) -> HoldoutEntry:
    return HoldoutEntry(
        trace_id=trace_id,
        reason=reason,
        owner=owner,
        added_at=added.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        ttl_days=ttl_days,
    )


# ---------------------------------------------------------------------------
# Schema round-trip + basic dataclass behaviour
# ---------------------------------------------------------------------------


def test_entry_round_trips_through_to_dict() -> None:
    when = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    e = _entry(added=when)
    assert HoldoutEntry.from_dict(e.to_dict()) == e


def test_holdouts_round_trips_through_to_dict() -> None:
    when = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    holdouts = Holdouts(entries={"a": _entry(trace_id="a", added=when)})
    rehydrated = Holdouts.from_dict(holdouts.to_dict())
    assert rehydrated == holdouts


def test_holdouts_from_dict_tolerates_garbage_entries() -> None:
    """A malformed entry must not crash load — surfaces as missing."""
    payload = {"schema_version": 1, "entries": {"a": "not a dict"}}
    holdouts = Holdouts.from_dict(payload)
    assert holdouts.entries == {}


# ---------------------------------------------------------------------------
# is_stale / days_left
# ---------------------------------------------------------------------------


def test_entry_is_stale_after_ttl_elapses() -> None:
    when = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    e = _entry(added=when, ttl_days=10)
    fresh_now = when + timedelta(days=5)
    stale_now = when + timedelta(days=11)
    assert e.is_stale(now=fresh_now) is False
    assert e.is_stale(now=stale_now) is True


def test_entry_days_left_is_negative_when_stale() -> None:
    when = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    e = _entry(added=when, ttl_days=10)
    now = when + timedelta(days=15)  # 5 days past expiry
    assert e.days_left(now=now) < 0


def test_holdouts_stale_count() -> None:
    fixed_now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    holdouts = Holdouts(
        entries={
            "fresh": _entry(trace_id="fresh", added=fixed_now - timedelta(days=5), ttl_days=30),
            "stale1": _entry(trace_id="stale1", added=fixed_now - timedelta(days=40), ttl_days=30),
            "stale2": _entry(trace_id="stale2", added=fixed_now - timedelta(days=100), ttl_days=30),
        }
    )
    assert holdouts.stale_count(now=fixed_now) == 2


# ---------------------------------------------------------------------------
# Atomic write + read
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    when = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    h = Holdouts(entries={"a": _entry(trace_id="a", added=when)})
    out = save(h, path=tmp_path / "h.json")
    assert out.is_file()
    rehydrated = load(path=out)
    assert rehydrated == h


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    """First-run state — load on a fresh repo must yield an empty
    Holdouts, not raise."""
    h = load(path=tmp_path / "never-created.json")
    assert h.entries == {}


def test_load_returns_empty_on_corrupt_file(tmp_path: Path) -> None:
    """A garbled file mustn't crash the daily glance."""
    f = tmp_path / "h.json"
    f.write_text("not json")
    assert load(path=f).entries == {}


def test_save_uses_atomic_rename_no_tmp_files(tmp_path: Path) -> None:
    """No `.tmp` files visible after a successful save."""
    when = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    save(Holdouts(entries={"a": _entry(trace_id="a", added=when)}), path=tmp_path / "h.json")
    tmps = list(tmp_path.rglob("*.tmp"))
    assert tmps == []


def test_save_rejects_directory_path_with_clear_message(tmp_path: Path) -> None:
    """Passing an existing directory to save() must raise a typed
    HoldoutPathError with a hint that names the file path the user
    probably meant — not leak IsADirectoryError from os.replace."""
    from shadow.holdout import HoldoutPathError

    when = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    h = Holdouts(entries={"a": _entry(trace_id="a", added=when)})
    with pytest.raises(HoldoutPathError, match="is a directory"):
        save(h, path=tmp_path)


def test_load_returns_empty_when_path_is_a_directory(tmp_path: Path) -> None:
    """Read side: a directory at `path` is treated as missing rather
    than crashing. Daily-glance behaviour stays graceful."""
    h = load(path=tmp_path)
    assert h.entries == {}


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_add_entry_inserts_a_new_trace() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    holdouts = add_entry(
        Holdouts(),
        trace_id="x",
        reason="r",
        owner="@me",
        ttl_days=30,
        now=now,
    )
    assert "x" in holdouts.entries
    assert holdouts.entries["x"].owner == "@me"


def test_add_entry_refreshes_existing_trace() -> None:
    """Re-adding the same trace id must reset added_at + update fields."""
    when_old = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    when_new = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    h = Holdouts(entries={"x": _entry(trace_id="x", added=when_old, owner="@old")})
    h2 = add_entry(h, trace_id="x", reason="new", owner="@new", ttl_days=15, now=when_new)
    e = h2.entries["x"]
    assert e.owner == "@new"
    assert e.ttl_days == 15
    assert e.added_at != _entry(trace_id="x", added=when_old).added_at


def test_remove_entry_drops_trace() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    h = Holdouts(entries={"x": _entry(trace_id="x", added=now)})
    h2, found = remove_entry(h, "x")
    assert found is True
    assert "x" not in h2.entries


def test_remove_entry_returns_found_false_for_unknown() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    h = Holdouts(entries={"x": _entry(trace_id="x", added=now)})
    h2, found = remove_entry(h, "ghost")
    assert found is False
    assert h2 == h


def test_reset_entry_only_changes_added_at() -> None:
    when_old = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    when_new = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    h = Holdouts(entries={"x": _entry(trace_id="x", added=when_old, ttl_days=42, owner="@me")})
    h2, found = reset_entry(h, "x", now=when_new)
    assert found is True
    e = h2.entries["x"]
    assert e.ttl_days == 42  # unchanged
    assert e.owner == "@me"  # unchanged
    # added_at must now reflect the new now (different ISO than the old).
    assert _entry(added=when_old, owner="@me").added_at != e.added_at


def test_reset_entry_returns_found_false_for_unknown() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    h2, found = reset_entry(Holdouts(), "ghost", now=now)
    assert found is False


# ---------------------------------------------------------------------------
# parse_ttl
# ---------------------------------------------------------------------------


def test_parse_ttl_days() -> None:
    assert parse_ttl("30d") == 30
    assert parse_ttl("1d") == 1


def test_parse_ttl_hours_round_up_to_one_day() -> None:
    """Sub-day units round up so a tiny TTL doesn't immediately go stale."""
    assert parse_ttl("1h") == 1
    assert parse_ttl("12h") == 1


def test_parse_ttl_minimum_is_one_day() -> None:
    assert parse_ttl("0d") == 1


def test_parse_ttl_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        parse_ttl("forever")


# ---------------------------------------------------------------------------
# Relative-added formatter
# ---------------------------------------------------------------------------


def test_relative_added_formats_just_now() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    ts = (now - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    assert relative_added(ts, now=now) == "just now"


def test_relative_added_formats_days() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    ts = (now - timedelta(days=12)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    assert relative_added(ts, now=now) == "12d ago"


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def test_render_empty_holdouts_shows_friendly_panel() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    buf = Console(record=True, width=120)
    render(Holdouts(), now=now, console=buf)
    out = buf.export_text()
    assert "No held-out" in out
    assert "shadow holdout add" in out


def test_render_populated_holdouts_shows_table_and_warning_when_stale() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    h = Holdouts(
        entries={
            "fresh": _entry(trace_id="fresh", added=now - timedelta(days=5), ttl_days=30),
            "stale": _entry(trace_id="stale", added=now - timedelta(days=40), ttl_days=30),
        }
    )
    buf = Console(record=True, width=120)
    render(h, now=now, console=buf)
    out = buf.export_text()
    assert "fresh" in out
    assert "stale" in out
    assert "STALE" in out
    assert "review overdue" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_holdout_list_empty_renders_friendly_panel(tmp_path: Path) -> None:
    base = tmp_path / "holdout.json"
    result = runner.invoke(app, ["holdout", "list", "--base", str(base)])
    assert result.exit_code == 0, result.output
    assert "No held-out" in result.output


def test_cli_holdout_add_then_list(tmp_path: Path) -> None:
    base = tmp_path / "holdout.json"
    result = runner.invoke(
        app,
        [
            "holdout",
            "add",
            "ba5e1a92",
            "--reason",
            "known flake",
            "--owner",
            "@alice",
            "--base",
            str(base),
        ],
    )
    assert result.exit_code == 0, result.output
    assert base.is_file()

    listed = runner.invoke(app, ["holdout", "list", "--base", str(base)])
    assert listed.exit_code == 0, listed.output
    assert "ba5e1a92" in listed.output
    assert "@alice" in listed.output


def test_cli_holdout_remove(tmp_path: Path) -> None:
    base = tmp_path / "holdout.json"
    runner.invoke(
        app,
        ["holdout", "add", "x", "--reason", "r", "--owner", "@me", "--base", str(base)],
    )
    rem = runner.invoke(app, ["holdout", "remove", "x", "--base", str(base)])
    assert rem.exit_code == 0, rem.output
    assert "removed" in rem.output


def test_cli_holdout_remove_unknown_warns(tmp_path: Path) -> None:
    """Removing a trace that isn't in the set must warn but not crash."""
    base = tmp_path / "holdout.json"
    result = runner.invoke(app, ["holdout", "remove", "ghost", "--base", str(base)])
    assert result.exit_code == 0, result.output
    assert "not in the holdout set" in result.output


def test_cli_holdout_reset_refreshes_window(tmp_path: Path) -> None:
    """`reset` must update the file even if the visible message is short."""
    base = tmp_path / "holdout.json"
    runner.invoke(
        app,
        [
            "holdout",
            "add",
            "x",
            "--reason",
            "r",
            "--owner",
            "@me",
            "--ttl",
            "5d",
            "--base",
            str(base),
        ],
    )
    before = json.loads(base.read_text())["entries"]["x"]["added_at"]
    # Sleep 1ms - microsecond timestamps differentiate easily.
    import time

    time.sleep(0.001)
    runner.invoke(app, ["holdout", "reset", "x", "--base", str(base)])
    after = json.loads(base.read_text())["entries"]["x"]["added_at"]
    assert after != before


def test_cli_holdout_list_stale_filters(tmp_path: Path) -> None:
    """`--stale` must hide non-stale entries."""
    base = tmp_path / "holdout.json"
    runner.invoke(
        app,
        [
            "holdout",
            "add",
            "fresh",
            "--reason",
            "r",
            "--owner",
            "@me",
            "--ttl",
            "30d",
            "--base",
            str(base),
        ],
    )
    runner.invoke(
        app,
        [
            "holdout",
            "add",
            "stale",
            "--reason",
            "r",
            "--owner",
            "@me",
            "--ttl",
            "1d",
            "--base",
            str(base),
        ],
    )
    # Manually backdate `stale` so it actually IS stale.
    payload = json.loads(base.read_text())
    payload["entries"]["stale"]["added_at"] = "2020-01-01T00:00:00.000000Z"
    base.write_text(json.dumps(payload))

    result = runner.invoke(app, ["holdout", "list", "--stale", "--json", "--base", str(base)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert "stale" in data["entries"]
    assert "fresh" not in data["entries"]


def test_cli_holdout_list_json_emits_payload(tmp_path: Path) -> None:
    base = tmp_path / "holdout.json"
    runner.invoke(
        app,
        ["holdout", "add", "x", "--reason", "r", "--owner", "@me", "--base", str(base)],
    )
    result = runner.invoke(app, ["holdout", "list", "--json", "--base", str(base)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert "schema_version" in data
    assert "x" in data["entries"]


def test_cli_holdout_add_directory_path_emits_friendly_error(tmp_path: Path) -> None:
    """Regression: passing a directory to `--base` used to leak
    IsADirectoryError. Now it surfaces as a clean error line with a
    hint that names the file path the user probably meant."""
    # tmp_path is itself a directory.
    result = runner.invoke(
        app,
        [
            "holdout",
            "add",
            "abc",
            "--reason",
            "r",
            "--owner",
            "@me",
            "--base",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "is a directory" in result.output
    assert "holdout.json" in result.output


def test_cli_holdout_list_directory_path_renders_empty(tmp_path: Path) -> None:
    """`shadow holdout list --base <dir>` must render the empty-state
    panel rather than crash on the directory."""
    result = runner.invoke(app, ["holdout", "list", "--base", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "No held-out" in result.output


def test_cli_holdout_add_garbled_ttl_errors(tmp_path: Path) -> None:
    base = tmp_path / "holdout.json"
    result = runner.invoke(
        app,
        [
            "holdout",
            "add",
            "x",
            "--reason",
            "r",
            "--owner",
            "@me",
            "--ttl",
            "forever",
            "--base",
            str(base),
        ],
    )
    assert result.exit_code == 1
    assert "could not parse" in result.output
