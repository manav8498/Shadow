"""Tests for `shadow.listen` and the `shadow listen` CLI command.

Two layers:

* Pure helpers — scan_dir, diff_states, listen_once.
* CLI — `--once` mode against the bundled fixtures, with friendly
  errors when paths are missing. The streaming mode (infinite loop) is
  not tested end-to-end; the loop is a thin shell around the helpers
  which are covered.
"""

from __future__ import annotations

import os
import time
from importlib import resources
from pathlib import Path

from typer.testing import CliRunner

import shadow.quickstart_data as _qs_data
from shadow.cli.app import app
from shadow.listen import diff_states, listen_once, scan_dir

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drop_fixture(target: Path, *, source_name: str = "candidate.agentlog") -> Path:
    """Copy a bundled fixture to ``target`` so it has real `.agentlog`
    bytes — empty files are skipped by ``scan_dir``."""
    root = resources.files(_qs_data) / "fixtures"
    target.write_bytes(root.joinpath(source_name).read_bytes())
    return target


# ---------------------------------------------------------------------------
# scan_dir
# ---------------------------------------------------------------------------


def test_scan_dir_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    """Missing directory is the normal first-run case — must not raise."""
    assert scan_dir(tmp_path / "never-created") == {}


def test_scan_dir_picks_up_agentlog_files_only(tmp_path: Path) -> None:
    _drop_fixture(tmp_path / "a.agentlog")
    (tmp_path / "b.txt").write_bytes(b"x" * 100)  # different suffix
    state = scan_dir(tmp_path)
    paths = {p.name for p in state}
    assert paths == {"a.agentlog"}


def test_scan_dir_skips_empty_files(tmp_path: Path) -> None:
    """An empty `.agentlog` is a placeholder, not a real trace.
    Including it would trip the loop on every `touch`."""
    (tmp_path / "empty.agentlog").write_bytes(b"")
    _drop_fixture(tmp_path / "real.agentlog")
    state = scan_dir(tmp_path)
    assert {p.name for p in state} == {"real.agentlog"}


def test_scan_dir_returns_resolved_paths(tmp_path: Path) -> None:
    """Resolved paths are what diff_states keys on so symlinks /
    relative-vs-absolute don't cause spurious events."""
    _drop_fixture(tmp_path / "a.agentlog")
    state = scan_dir(tmp_path)
    for p in state:
        assert p.is_absolute()


# ---------------------------------------------------------------------------
# diff_states
# ---------------------------------------------------------------------------


def test_diff_states_no_changes_returns_empty() -> None:
    state = {Path("/x/a.agentlog"): 100.0, Path("/x/b.agentlog"): 200.0}
    assert diff_states(state, state) == []


def test_diff_states_added_paths_become_added_events() -> None:
    prev: dict[Path, float] = {}
    curr = {Path("/x/a.agentlog"): 100.0}
    events = diff_states(prev, curr)
    assert len(events) == 1
    assert events[0].kind == "added"
    assert events[0].path == Path("/x/a.agentlog")


def test_diff_states_increased_mtime_becomes_modified() -> None:
    prev = {Path("/x/a.agentlog"): 100.0}
    curr = {Path("/x/a.agentlog"): 200.0}
    events = diff_states(prev, curr)
    assert len(events) == 1
    assert events[0].kind == "modified"


def test_diff_states_unchanged_mtime_emits_nothing() -> None:
    prev = {Path("/x/a.agentlog"): 100.0}
    curr = {Path("/x/a.agentlog"): 100.0}  # unchanged
    assert diff_states(prev, curr) == []


def test_diff_states_orders_events_oldest_first() -> None:
    prev: dict[Path, float] = {}
    curr = {
        Path("/x/late.agentlog"): 200.0,
        Path("/x/early.agentlog"): 100.0,
    }
    events = diff_states(prev, curr)
    assert [e.path.name for e in events] == ["early.agentlog", "late.agentlog"]


def test_diff_states_does_not_emit_for_removed_paths() -> None:
    """The listener cares about new candidates landing, not cleanup —
    removal is intentionally silent."""
    prev = {Path("/x/a.agentlog"): 100.0}
    curr: dict[Path, float] = {}
    assert diff_states(prev, curr) == []


# ---------------------------------------------------------------------------
# listen_once
# ---------------------------------------------------------------------------


def test_listen_once_emits_added_event_for_new_file(tmp_path: Path) -> None:
    target = tmp_path / "candidate.agentlog"
    _drop_fixture(target)
    state, events = listen_once({}, tmp_path)
    assert len(events) == 1
    assert events[0].kind == "added"
    assert state.get(target.resolve()) is not None


def test_listen_once_suppresses_anchor_path(tmp_path: Path) -> None:
    """The anchor file is the baseline — it must never produce an event
    even though it lives in the watched directory."""
    anchor = tmp_path / "anchor.agentlog"
    candidate = tmp_path / "candidate.agentlog"
    _drop_fixture(anchor, source_name="baseline.agentlog")
    _drop_fixture(candidate)

    state, events = listen_once({}, tmp_path, anchor_path=anchor)
    paths = {e.path.name for e in events}
    assert paths == {"candidate.agentlog"}


def test_listen_once_modified_event_on_mtime_advance(tmp_path: Path) -> None:
    """A second tick after touching the file must emit a modified event."""
    candidate = tmp_path / "candidate.agentlog"
    _drop_fixture(candidate)

    state, events = listen_once({}, tmp_path)
    assert events[0].kind == "added"

    # Advance the mtime by one second — coarser than fs resolution on
    # any reasonable platform.
    new_mtime = time.time() + 1.0
    os.utime(candidate, (new_mtime, new_mtime))

    state, events = listen_once(state, tmp_path)
    assert len(events) == 1
    assert events[0].kind == "modified"


def test_listen_once_idempotent_when_no_changes(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.agentlog"
    _drop_fixture(candidate)
    state, _ = listen_once({}, tmp_path)
    state2, events = listen_once(state, tmp_path)
    assert events == []
    assert state == state2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_listen_once_emits_one_line_for_new_file(tmp_path: Path) -> None:
    """End-to-end: drop a candidate file in the watched directory and
    confirm the listener emits the change line in `--once` mode."""
    runs = tmp_path / "runs"
    runs.mkdir()
    anchor = tmp_path / "anchor.agentlog"
    _drop_fixture(anchor, source_name="baseline.agentlog")
    _drop_fixture(runs / "r1.agentlog")

    result = runner.invoke(
        app,
        ["listen", str(runs), "--anchor", str(anchor), "--once"],
    )
    assert result.exit_code == 0, result.output
    assert "+" in result.output  # added marker
    assert "r1.agentlog" in result.output
    # The bundled fixtures regress on trajectory — must surface in output.
    assert "probe" in result.output or "stop" in result.output or "hold" in result.output


def test_cli_listen_anchor_missing_emits_friendly_error(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    result = runner.invoke(
        app,
        ["listen", str(runs), "--anchor", str(tmp_path / "missing.agentlog"), "--once"],
    )
    assert result.exit_code == 1
    assert "anchor file not found" in result.output


def test_cli_listen_watch_dir_missing_emits_friendly_error(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.agentlog"
    _drop_fixture(anchor, source_name="baseline.agentlog")
    result = runner.invoke(
        app,
        ["listen", str(tmp_path / "no-such-dir"), "--anchor", str(anchor), "--once"],
    )
    assert result.exit_code == 1
    assert "watch directory not found" in result.output


def test_cli_listen_streaming_mode_default_does_not_create_ledger(tmp_path: Path) -> None:
    """Zero-regression invariant: even with `--once`, the listener
    must not create `.shadow/ledger/` unless `--log` was passed."""
    runs = tmp_path / "runs"
    runs.mkdir()
    anchor = tmp_path / "anchor.agentlog"
    _drop_fixture(anchor, source_name="baseline.agentlog")
    _drop_fixture(runs / "r1.agentlog")

    result = runner.invoke(
        app,
        ["listen", str(runs), "--anchor", str(anchor), "--once"],
    )
    assert result.exit_code == 0
    assert not (tmp_path / ".shadow" / "ledger").exists()


def test_cli_listen_log_flag_writes_ledger_entry(tmp_path: Path, monkeypatch) -> None:
    """`--log` must append a ledger entry per emitted event."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    anchor = tmp_path / "anchor.agentlog"
    _drop_fixture(anchor, source_name="baseline.agentlog")
    _drop_fixture(runs / "r1.agentlog")

    result = runner.invoke(
        app,
        ["listen", str(runs), "--anchor", str(anchor), "--once", "--log"],
    )
    assert result.exit_code == 0, result.output
    base = tmp_path / ".shadow" / "ledger"
    assert base.exists()
    # At least one entry must have landed.
    entries = list(base.rglob("*.json"))
    assert entries
