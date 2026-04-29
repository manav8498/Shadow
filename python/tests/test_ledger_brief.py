"""Tests for `shadow.ledger.brief` and the `shadow brief` CLI command.

Three layers:

* Pure formatters — terminal, markdown, slack Block Kit shape.
* Webhook poster — payload assembly + fail-soft on network errors
  (urllib monkey-patched, no real network calls in CI).
* CLI — three formats, webhook path, garbled flags.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from shadow.cli.app import app
from shadow.ledger import (
    LedgerEntry,
    compute_view,
    format_brief_markdown,
    format_brief_slack,
    format_brief_terminal,
    post_to_slack,
    write_entry,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    *,
    when: datetime,
    tier: str | None = "ship",
    summary: str | None = None,
    anchor: str = "ba5e1a92",
    candidate: str = "c0f2d3a4",
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
        primary_axis="trajectory" if tier == "stop" else None,
    )


def _populated_view() -> Any:
    """Build a small populated LedgerView for formatter tests."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    entries = [
        _entry(tier="ship", when=now - timedelta(hours=8), anchor="ship0001"),
        _entry(
            tier="stop",
            when=now - timedelta(hours=2),
            anchor="ba5e1a92",
            candidate="c0f2d3a4",
            summary="structural change at turn 0",
        ),
    ]
    return compute_view(entries, now=now, since=timedelta(days=1))


# ---------------------------------------------------------------------------
# Terminal formatter
# ---------------------------------------------------------------------------


def test_format_brief_terminal_starts_with_dated_header() -> None:
    view = _populated_view()
    out = format_brief_terminal(view)
    assert out.startswith("Shadow brief — 2026-04-29")


def test_format_brief_terminal_includes_pass_rate_and_ci() -> None:
    view = _populated_view()
    out = format_brief_terminal(view)
    assert "Anchor pass rate" in out
    assert "1 of 2 calls" in out
    assert "95% CI" in out


def test_format_brief_terminal_lists_most_concerning_first() -> None:
    view = _populated_view()
    out = format_brief_terminal(view)
    assert "Most concerning" in out
    # The stop-tier entry's driver must surface in the brief.
    assert "structural change at turn 0" in out


def test_format_brief_terminal_empty_view_points_at_log_command() -> None:
    """No entries means a friendly hint, not an empty payload."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
    view = compute_view([], now=now)
    out = format_brief_terminal(view)
    assert "no calls logged yet" in out
    assert "shadow call" in out and "--log" in out


def test_format_brief_terminal_suggests_next_command() -> None:
    """Brief always ends with one suggested next command."""
    view = _populated_view()
    out = format_brief_terminal(view)
    last = [line for line in out.split("\n") if line.strip()][-1]
    assert last.startswith("Next:")


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------


def test_format_brief_markdown_uses_heading_for_default() -> None:
    view = _populated_view()
    md = format_brief_markdown(view)
    assert md.startswith("### Shadow brief —")


def test_format_brief_markdown_renders_trace_ids_as_code_spans() -> None:
    """Trace ids must be code spans so they copy back into shadow trail."""
    view = _populated_view()
    md = format_brief_markdown(view)
    assert "`ba5e1a92`" in md


def test_format_brief_markdown_lists_recent_entries_as_bullets() -> None:
    view = _populated_view()
    md = format_brief_markdown(view)
    assert "**Recent**" in md
    bullet_lines = [line for line in md.split("\n") if line.startswith("- ")]
    assert len(bullet_lines) >= 1


# ---------------------------------------------------------------------------
# Slack formatter
# ---------------------------------------------------------------------------


def test_format_brief_slack_returns_block_kit_shape() -> None:
    view = _populated_view()
    payload = format_brief_slack(view)
    assert "blocks" in payload
    assert payload["blocks"][0]["type"] == "section"
    assert payload["blocks"][0]["text"]["type"] == "mrkdwn"
    # Trailing context block carries the verifiable footer.
    assert payload["blocks"][-1]["type"] == "context"


def test_format_brief_slack_uses_bold_not_heading() -> None:
    """Slack mrkdwn doesn't render `###`; use `*bold*` instead."""
    view = _populated_view()
    payload = format_brief_slack(view)
    text = payload["blocks"][0]["text"]["text"]
    assert "###" not in text
    assert text.startswith("*Shadow brief —")


def test_format_brief_slack_round_trips_through_json() -> None:
    """The payload is something Slack will accept — must serialise."""
    view = _populated_view()
    payload = format_brief_slack(view)
    # Round-trip through json.dumps + loads to confirm no non-serialisable
    # types crept in (datetime, set, etc).
    rehydrated = json.loads(json.dumps(payload))
    assert rehydrated == payload


# ---------------------------------------------------------------------------
# Webhook poster
# ---------------------------------------------------------------------------


def test_post_to_slack_returns_ok_on_2xx() -> None:
    class _FakeResp:
        status = 200

        def read(self) -> bytes:
            return b"ok"

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_: Any) -> None:
            pass

    with patch("urllib.request.urlopen", return_value=_FakeResp()):
        ok, msg = post_to_slack("https://hooks.example.com/x", {"blocks": []})
    assert ok is True
    assert msg == "ok"


def test_post_to_slack_returns_error_on_http_failure() -> None:
    import urllib.error

    err = urllib.error.HTTPError(
        url="https://hooks.example.com/x",
        code=400,
        msg="Bad Request",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=err):
        ok, msg = post_to_slack("https://hooks.example.com/x", {"blocks": []})
    assert ok is False
    assert "400" in msg


def test_post_to_slack_returns_error_on_network_failure() -> None:
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("dns")):
        ok, msg = post_to_slack("https://hooks.example.com/x", {"blocks": []})
    assert ok is False
    assert "network error" in msg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_brief_default_format_is_terminal(tmp_path: Path) -> None:
    base = tmp_path / "ledger"
    _populate_ledger(base, tmp_path)
    result = runner.invoke(app, ["brief", "--base", str(base)])
    assert result.exit_code == 0, result.output
    # Plain-text header sentinel (terminal format only)
    assert "Shadow brief —" in result.output
    assert "Anchor pass rate" in result.output


def test_cli_brief_markdown_format(tmp_path: Path) -> None:
    base = tmp_path / "ledger"
    _populate_ledger(base, tmp_path)
    result = runner.invoke(app, ["brief", "--base", str(base), "--format", "markdown"])
    assert result.exit_code == 0, result.output
    assert "###" in result.output  # markdown heading
    assert "**Anchor pass rate**" in result.output


def test_cli_brief_slack_format_emits_block_kit_json(tmp_path: Path) -> None:
    base = tmp_path / "ledger"
    _populate_ledger(base, tmp_path)
    result = runner.invoke(app, ["brief", "--base", str(base), "--format", "slack"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert "blocks" in payload


def test_cli_brief_unknown_format_errors(tmp_path: Path) -> None:
    base = tmp_path / "ledger"
    _populate_ledger(base, tmp_path)
    result = runner.invoke(app, ["brief", "--base", str(base), "--format", "wat"])
    assert result.exit_code == 1
    assert "unknown format" in result.output


def test_cli_brief_webhook_fail_soft(tmp_path: Path) -> None:
    """A bad webhook URL must surface a warning and exit 0."""
    base = tmp_path / "ledger"
    _populate_ledger(base, tmp_path)
    result = runner.invoke(
        app,
        [
            "brief",
            "--base",
            str(base),
            "--slack-webhook",
            "https://hooks.invalid.example/x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "warning" in result.output


def test_cli_brief_empty_ledger_renders_friendly(tmp_path: Path) -> None:
    """A repo with no ledger must produce a graceful brief, not a crash."""
    base = tmp_path / "ledger"
    result = runner.invoke(app, ["brief", "--base", str(base)])
    assert result.exit_code == 0, result.output
    assert "no calls logged yet" in result.output


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def _populate_ledger(base: Path, tmp_path: Path) -> None:
    """Write a minimal call entry into the given base path."""
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
