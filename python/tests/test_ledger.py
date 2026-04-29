"""Tests for `shadow.ledger` — the opt-in artifact record store.

Three layers:

    * Pure store logic — round-trip, atomic write, bucketing, sort order.
    * Constructors from real Shadow output — diff report and CallResult.
    * CLI surface — `shadow log <report.json>`, `--log` flag on `shadow call`.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import shadow.quickstart_data as _qs_data
from shadow import _core
from shadow.call import compute_call
from shadow.cli.app import app
from shadow.ledger import (
    LedgerEntry,
    entry_from_call,
    entry_from_diff_report,
    read_recent,
    write_entry,
)
from shadow.ledger.store import (
    SCHEMA_VERSION,
    _day_bucket,
    _filename_for,
    _short,
    _worst_severity,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bundled() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = resources.files(_qs_data) / "fixtures"
    b = _core.parse_agentlog(root.joinpath("baseline.agentlog").read_bytes())
    c = _core.parse_agentlog(root.joinpath("candidate.agentlog").read_bytes())
    return b, c


def _sample_call_payload() -> dict[str, Any]:
    """Build a CallResult-shaped dict from the bundled fixtures."""
    b, c = _bundled()
    report = _core.compute_diff_report(b, c, None, 42)
    return compute_call(report).to_dict()


def _sample_diff_report() -> dict[str, Any]:
    b, c = _bundled()
    return _core.compute_diff_report(b, c, None, 42)


# ---------------------------------------------------------------------------
# Pure store: round-trip, atomic write, sorting
# ---------------------------------------------------------------------------


def test_entry_round_trips_through_to_dict_and_from_dict() -> None:
    """Every field a writer puts in must come out the other side."""
    entry = LedgerEntry(
        kind="call",
        timestamp="2026-04-29T18:32:04.123456Z",
        anchor_id="ba5e1a92",
        candidate_id="c0f2d3a4",
        tier="stop",
        worst_severity="severe",
        pair_count=20,
        driver_summary="structural change at turn 0",
        primary_axis="trajectory",
        source_command="shadow call",
    )
    rehydrated = LedgerEntry.from_dict(entry.to_dict())
    assert rehydrated == entry


def test_entry_to_dict_drops_none_optionals() -> None:
    """Terse JSON: an entry with no tier shouldn't carry `tier: null`."""
    entry = LedgerEntry(
        kind="diff",
        timestamp="2026-04-29T18:32:04.123456Z",
        anchor_id="abc",
        candidate_id="def",
    )
    d = entry.to_dict()
    assert "tier" not in d
    assert "driver_summary" not in d


def test_entry_from_dict_preserves_unknown_fields_in_extras() -> None:
    """Forward-compat: a future Shadow that adds a `cert_id` field
    must round-trip through this Shadow without losing data."""
    payload = {
        "schema_version": 99,
        "kind": "cert",
        "timestamp": "2099-01-01T00:00:00.000000Z",
        "anchor_id": "x",
        "candidate_id": "y",
        "future_field": "preserve me",
    }
    e = LedgerEntry.from_dict(payload)
    assert e.extras == {"future_field": "preserve me"}


def test_write_then_read_recent_round_trips(tmp_path: Path) -> None:
    """Write three entries and confirm read_recent returns all of them
    newest first."""
    base = tmp_path / "ledger"
    timestamps = [
        "2026-04-29T10:00:00.000000Z",
        "2026-04-29T11:00:00.000000Z",
        "2026-04-29T12:00:00.000000Z",
    ]
    for ts in timestamps:
        write_entry(
            LedgerEntry(
                kind="call",
                timestamp=ts,
                anchor_id="a",
                candidate_id="b",
                tier="ship",
            ),
            base=base,
        )
    recent = read_recent(base=base, limit=10)
    assert [e.timestamp for e in recent] == list(reversed(timestamps))


def test_read_recent_respects_limit(tmp_path: Path) -> None:
    base = tmp_path / "ledger"
    for i in range(5):
        write_entry(
            LedgerEntry(
                kind="call",
                timestamp=f"2026-04-29T10:00:0{i}.000000Z",
                anchor_id="a",
                candidate_id="b",
            ),
            base=base,
        )
    assert len(read_recent(base=base, limit=3)) == 3


def test_read_recent_returns_empty_when_ledger_does_not_exist(tmp_path: Path) -> None:
    """Default state for a user who never opted into logging — must
    not raise, must not create the directory."""
    base = tmp_path / "never-created"
    assert read_recent(base=base) == []
    assert not base.exists()


def test_read_recent_skips_unreadable_entries(tmp_path: Path) -> None:
    """A garbled or partially-written file must not crash the daily
    glance — broken entries are silently skipped."""
    base = tmp_path / "ledger"
    write_entry(
        LedgerEntry(
            kind="call",
            timestamp="2026-04-29T10:00:00.000000Z",
            anchor_id="a",
            candidate_id="b",
            tier="ship",
        ),
        base=base,
    )
    # Drop a corrupt file alongside the good one.
    (base / "20260429" / "999999-999999-call-bad-bad.json").write_text("not json")

    recent = read_recent(base=base)
    # Only the good entry comes back.
    assert len(recent) == 1
    assert recent[0].tier == "ship"


def test_atomic_write_leaves_no_tmp_files(tmp_path: Path) -> None:
    """The temp file used during atomic-rename must never be visible
    after a successful write."""
    base = tmp_path / "ledger"
    write_entry(
        LedgerEntry(
            kind="diff",
            timestamp="2026-04-29T10:00:00.000000Z",
            anchor_id="a",
            candidate_id="b",
        ),
        base=base,
    )
    tmps = list(base.rglob("*.tmp"))
    assert tmps == []


# ---------------------------------------------------------------------------
# Filename + bucket helpers
# ---------------------------------------------------------------------------


def test_day_bucket_extracts_yyyymmdd() -> None:
    assert _day_bucket("2026-04-29T18:32:04.123456Z") == "20260429"


def test_day_bucket_falls_back_on_garbage() -> None:
    assert _day_bucket("not a timestamp") == "00000000"


def test_filename_includes_kind_and_short_ids() -> None:
    entry = LedgerEntry(
        kind="call",
        timestamp="2026-04-29T18:32:04.123456Z",
        anchor_id="ba5e1a9201234567",
        candidate_id="c0f2d3a4abcdef",
    )
    name = _filename_for(entry)
    assert name.startswith("183204-123456-call-ba5e1a92-c0f2d3a4")
    assert name.endswith(".json")


def test_short_id_strips_sha256_prefix() -> None:
    assert _short("sha256:abcdef0123456789") == "abcdef01"


# ---------------------------------------------------------------------------
# Constructors from real Shadow output
# ---------------------------------------------------------------------------


def test_entry_from_call_carries_tier_and_driver() -> None:
    payload = _sample_call_payload()
    entry = entry_from_call(payload)
    assert entry.kind == "call"
    assert entry.tier in {"ship", "hold", "probe", "stop"}
    # Bundled fixtures regress on trajectory; primary_axis must surface.
    assert entry.primary_axis == "trajectory"
    assert entry.driver_summary is not None


def test_entry_from_diff_report_carries_worst_severity() -> None:
    report = _sample_diff_report()
    entry = entry_from_diff_report(report)
    assert entry.kind == "diff"
    # Bundled fixtures have several severe axes.
    assert entry.worst_severity == "severe"
    assert entry.pair_count == 3
    # Diff entries don't compute a tier — that's the call layer's job.
    assert entry.tier is None


def test_worst_severity_picks_highest_rank() -> None:
    rows = [
        {"axis": "a", "severity": "minor"},
        {"axis": "b", "severity": "severe"},
        {"axis": "c", "severity": "moderate"},
    ]
    assert _worst_severity(rows) == "severe"


def test_worst_severity_returns_none_for_empty_rows() -> None:
    assert _worst_severity([]) is None


# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------


def test_schema_version_is_set_on_new_entries() -> None:
    entry = LedgerEntry(
        kind="call", timestamp="2026-04-29T10:00:00.000000Z", anchor_id="a", candidate_id="b"
    )
    assert entry.schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# CLI: `shadow log <report.json>`
# ---------------------------------------------------------------------------


def test_cli_log_writes_diff_entry(tmp_path: Path) -> None:
    """`shadow log` on a diff-shaped JSON file must land a diff entry."""
    report = _sample_diff_report()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    base = tmp_path / "ledger"

    result = runner.invoke(app, ["log", str(report_path), "--base", str(base)])
    assert result.exit_code == 0, result.output
    recent = read_recent(base=base)
    assert len(recent) == 1
    assert recent[0].kind == "diff"


def test_cli_log_writes_call_entry(tmp_path: Path) -> None:
    """`shadow log` on a call-shaped JSON file must land a call entry."""
    payload = _sample_call_payload()
    report_path = tmp_path / "call.json"
    report_path.write_text(json.dumps(payload))
    base = tmp_path / "ledger"

    result = runner.invoke(app, ["log", str(report_path), "--base", str(base)])
    assert result.exit_code == 0, result.output
    recent = read_recent(base=base)
    assert len(recent) == 1
    assert recent[0].kind == "call"


def test_cli_log_rejects_non_shadow_json(tmp_path: Path) -> None:
    """A JSON file that's not a Shadow report must surface an error."""
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps({"hello": "world"}))
    result = runner.invoke(app, ["log", str(odd), "--base", str(tmp_path / "ledger")])
    assert result.exit_code == 1
    assert "doesn't look like" in result.output


def test_cli_log_emits_friendly_error_on_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["log", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
    # The shared `_fail()` hint must surface.
    assert "shadow demo" in result.output


# ---------------------------------------------------------------------------
# CLI: `--log` flag on `shadow call`
# ---------------------------------------------------------------------------


def test_cli_call_default_writes_no_ledger_entry(tmp_path: Path, monkeypatch: Any) -> None:
    """Zero-regression invariant: `shadow call` without `--log` must not
    create `.shadow/ledger/` or any file inside it."""
    monkeypatch.chdir(tmp_path)
    root = resources.files(_qs_data) / "fixtures"
    b_path = tmp_path / "anchor.agentlog"
    c_path = tmp_path / "candidate.agentlog"
    b_path.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c_path.write_bytes(root.joinpath("candidate.agentlog").read_bytes())

    result = runner.invoke(app, ["call", str(b_path), str(c_path)])
    assert result.exit_code == 0
    assert not (tmp_path / ".shadow" / "ledger").exists()


def test_cli_call_log_flag_writes_one_ledger_entry(tmp_path: Path, monkeypatch: Any) -> None:
    """With `--log`, `shadow call` must land exactly one entry."""
    monkeypatch.chdir(tmp_path)
    root = resources.files(_qs_data) / "fixtures"
    b_path = tmp_path / "anchor.agentlog"
    c_path = tmp_path / "candidate.agentlog"
    b_path.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c_path.write_bytes(root.joinpath("candidate.agentlog").read_bytes())

    result = runner.invoke(app, ["call", str(b_path), str(c_path), "--log"])
    assert result.exit_code == 0
    base = tmp_path / ".shadow" / "ledger"
    assert base.exists()
    recent = read_recent(base=base)
    assert len(recent) == 1
    assert recent[0].kind == "call"
