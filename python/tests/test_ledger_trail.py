"""Tests for `shadow.ledger.trail` and the `shadow trail` CLI command.

Three layers:

* Pure walk logic — chain construction, cycle detection, depth limit,
  unknown trace, terminal anchor, self-loop handling.
* Renderer — populated chain panel and unknown-trace empty state.
* CLI — end-to-end via Typer's CliRunner.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from shadow.cli.app import app
from shadow.ledger import (
    LedgerEntry,
    TrailResult,
    compute_trail,
    render_trail,
    write_entry,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    *,
    anchor: str,
    candidate: str,
    when: datetime,
    tier: str | None = "ship",
    summary: str | None = None,
    primary: str | None = None,
) -> LedgerEntry:
    return LedgerEntry(
        kind="call",
        timestamp=when.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        anchor_id=anchor,
        candidate_id=candidate,
        tier=tier,
        worst_severity="severe" if tier == "stop" else "minor" if tier == "hold" else "none",
        pair_count=10,
        driver_summary=summary,
        primary_axis=primary,
    )


# ---------------------------------------------------------------------------
# Walk: 3-step chain
# ---------------------------------------------------------------------------


def test_compute_trail_walks_three_steps_back() -> None:
    """A → B → C should yield steps in reverse: C, B, A."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(
            anchor="A",
            candidate="B",
            when=now - timedelta(hours=12),
            tier="hold",
            summary="verbosity moved up",
        ),
        _entry(
            anchor="B",
            candidate="C",
            when=now - timedelta(hours=2),
            tier="stop",
            summary="structural change at turn 0",
        ),
    ]
    result = compute_trail(entries, trace_id="C", depth=5)
    assert [s.trace_id for s in result.steps] == ["C", "B", "A"]
    assert result.steps[0].tier == "stop"
    assert result.steps[1].tier == "hold"
    # Terminal anchor — A only appears as anchor in the chain, no further history.
    assert result.steps[2].role == "anchor"


def test_compute_trail_marks_first_step_as_candidate() -> None:
    """The starting trace id walks in via an anchor→candidate edge,
    so the first step's role is candidate."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(anchor="A", candidate="B", when=now, tier="ship"),
    ]
    result = compute_trail(entries, trace_id="B", depth=5)
    assert result.steps[0].role == "candidate"


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def test_compute_trail_detects_cycle() -> None:
    """A → B and B → A must trip the cycle guard."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(anchor="A", candidate="B", when=now - timedelta(hours=2)),
        _entry(anchor="B", candidate="A", when=now - timedelta(hours=1)),
    ]
    result = compute_trail(entries, trace_id="A", depth=10)
    assert result.truncated_by_cycle is True
    # Steps walked: A (via entry 2), then B (via entry 1), then A again -> cycle break.
    assert [s.trace_id for s in result.steps] == ["A", "B"]


def test_compute_trail_skips_self_loop_when_alternative_exists() -> None:
    """A self-loop entry (anchor == candidate) must not be preferred
    over a real anchor → candidate edge."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(anchor="A", candidate="A", when=now - timedelta(hours=2)),  # self-loop
        _entry(anchor="A", candidate="B", when=now - timedelta(hours=3)),  # real edge
    ]
    result = compute_trail(entries, trace_id="B", depth=5)
    # Should walk B -> A (via the real edge), not get stuck on the self-loop.
    assert [s.trace_id for s in result.steps] == ["B", "A"]
    assert result.truncated_by_cycle is False


def test_compute_trail_terminates_on_self_loop_only() -> None:
    """When the only entry is a self-loop, the walk emits one step
    and stops without crying cycle (the chain just has nowhere to go)."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(anchor="A", candidate="A", when=now),
    ]
    result = compute_trail(entries, trace_id="A", depth=5)
    assert len(result.steps) == 1
    assert result.steps[0].parent_trace_id is None
    assert result.truncated_by_cycle is False


# ---------------------------------------------------------------------------
# Depth limit
# ---------------------------------------------------------------------------


def test_compute_trail_honours_depth_limit() -> None:
    """A long chain must stop at depth N."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(anchor="A", candidate="B", when=now - timedelta(hours=4)),
        _entry(anchor="B", candidate="C", when=now - timedelta(hours=3)),
        _entry(anchor="C", candidate="D", when=now - timedelta(hours=2)),
        _entry(anchor="D", candidate="E", when=now - timedelta(hours=1)),
    ]
    result = compute_trail(entries, trace_id="E", depth=2)
    assert [s.trace_id for s in result.steps] == ["E", "D"]
    assert result.truncated_by_depth is True


# ---------------------------------------------------------------------------
# Unknown trace
# ---------------------------------------------------------------------------


def test_compute_trail_returns_empty_for_unknown_trace() -> None:
    """A trace id not referenced by any entry must yield no steps."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(anchor="A", candidate="B", when=now),
    ]
    result = compute_trail(entries, trace_id="ghost", depth=5)
    assert result.steps == []
    assert result.found is False


def test_compute_trail_emits_terminal_anchor_when_only_appears_as_anchor() -> None:
    """A trace that only ever appears as an anchor (never a candidate)
    must produce a single terminal step rather than nothing."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(anchor="A", candidate="B", when=now),
    ]
    result = compute_trail(entries, trace_id="A", depth=5)
    assert len(result.steps) == 1
    assert result.steps[0].trace_id == "A"
    assert result.steps[0].role == "anchor"


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def test_render_trail_unknown_trace_shows_friendly_message() -> None:
    result = TrailResult(root_trace_id="ghost", steps=[])
    buf = Console(record=True, width=120)
    render_trail(result, console=buf)
    out = buf.export_text()
    assert "ghost" in out
    assert "No ledger entry references" in out


def test_render_trail_chain_shows_each_step_and_actions() -> None:
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(
            anchor="ba5e1a92",
            candidate="c0f2d3a4",
            when=now,
            tier="stop",
            summary="structural change at turn 0",
            primary="trajectory",
        ),
    ]
    result = compute_trail(entries, trace_id="c0f2d3a4", depth=5)
    buf = Console(record=True, width=120)
    render_trail(result, console=buf)
    out = buf.export_text()
    assert "c0f2d3a4" in out
    assert "ba5e1a92" in out
    assert "stop" in out
    assert "What to do" in out
    # Action lines reference both endpoints
    assert "shadow call ba5e1a92 c0f2d3a4" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_trail_unknown_trace_exits_zero_with_message(tmp_path: Path) -> None:
    """An unknown trace id must render the friendly empty panel and
    exit 0 — it's an information request, not a failure."""
    base = tmp_path / "ledger"
    result = runner.invoke(app, ["trail", "ghost", "--base", str(base)])
    assert result.exit_code == 0, result.output
    assert "No ledger entry references" in result.output


def test_cli_trail_renders_populated_chain(tmp_path: Path) -> None:
    """A real chain in the ledger must render with both anchor and candidate."""
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

    result = runner.invoke(app, ["trail", "c0f2d3a4", "--base", str(base)])
    assert result.exit_code == 0, result.output
    assert "ba5e1a92" in result.output
    assert "c0f2d3a4" in result.output
    assert "stop" in result.output


def test_cli_trail_json_emits_structured_payload(tmp_path: Path) -> None:
    base = tmp_path / "ledger"
    now = datetime.now(UTC)
    entry = LedgerEntry(
        kind="call",
        timestamp=(now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        anchor_id="A",
        candidate_id="B",
        tier="hold",
        pair_count=10,
        driver_summary="verbosity moved up",
        source_command="shadow call",
    )
    write_entry(entry, base=base)

    result = runner.invoke(app, ["trail", "B", "--json", "--base", str(base)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload["root_trace_id"] == "B"
    assert len(payload["steps"]) >= 1
    assert payload["steps"][0]["tier"] == "hold"


def test_cli_trail_depth_flag_truncates_chain(tmp_path: Path) -> None:
    """`--depth 1` must cut the walk after one step."""
    base = tmp_path / "ledger"
    now = datetime.now(UTC)
    entries = [
        ("A", "B", "ship", now - timedelta(hours=3)),
        ("B", "C", "stop", now - timedelta(hours=1)),
    ]
    for anchor, cand, tier, ts in entries:
        e = LedgerEntry(
            kind="call",
            timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
            anchor_id=anchor,
            candidate_id=cand,
            tier=tier,
            pair_count=10,
            source_command="shadow call",
        )
        write_entry(e, base=base)

    result = runner.invoke(app, ["trail", "C", "--depth", "1", "--json", "--base", str(base)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert len(payload["steps"]) == 1
    assert payload["truncated_by_depth"] is True
