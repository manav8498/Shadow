"""Tests for `shadow inspect` and the `shadow.inspect` library surface.

The terminal renderer itself is exercised end-to-end via the CLI;
the library tests pin the row-extraction logic so future record-
shape changes can't quietly drift the column data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from shadow.inspect import (
    TraceRow,
    first_divergence_index,
    load_trace_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "examples" / "refund-causal-diagnosis" / "baseline_traces"
CANDIDATE_DIR = REPO_ROOT / "examples" / "refund-causal-diagnosis" / "candidate_traces"


# ---- library-level tests --------------------------------------------------


def test_load_trace_rows_parses_real_agentlog() -> None:
    src = next(BASELINE_DIR.glob("*.agentlog"))
    rows = load_trace_rows(src)
    assert len(rows) >= 2
    # First row is the metadata envelope.
    assert rows[0].kind == "metadata"
    # Subsequent rows include chat_request + chat_response pairs.
    assert any(r.kind == "chat_request" for r in rows)
    assert any(r.kind == "chat_response" for r in rows)


def test_load_trace_rows_assigns_increasing_turn_numbers() -> None:
    src = next(BASELINE_DIR.glob("*.agentlog"))
    rows = load_trace_rows(src)
    turns = [r.turn for r in rows if r.kind == "chat_request"]
    # chat_request rows bump the turn counter; turns should be 1..N
    # in order, no duplicates, no gaps.
    assert turns == sorted(turns)
    assert turns == list(range(1, len(turns) + 1))


def test_load_trace_rows_summary_includes_user_message_text() -> None:
    src = next(BASELINE_DIR.glob("*.agentlog"))
    rows = load_trace_rows(src)
    user_summaries = [r.summary for r in rows if r.kind == "chat_request"]
    # Every chat_request should have a non-empty summary — the user
    # message text. Empty would be a regression.
    assert all(s for s in user_summaries)


def test_load_trace_rows_extracts_token_usage_when_present() -> None:
    """Most chat_response records in the demo carry usage; a few
    might not. We assert at least one row has usage so the
    extractor's good path is covered."""
    src = next(BASELINE_DIR.glob("*.agentlog"))
    rows = load_trace_rows(src)
    has_usage = [r for r in rows if r.input_tokens is not None or r.output_tokens is not None]
    assert has_usage, "no rows extracted token usage; check Session metadata path"


def test_load_trace_rows_redaction_count_zero_on_clean_trace() -> None:
    """The committed baseline traces have already been swept by the
    Session's Redactor, so the per-record `redactions` count should
    be 0 across the file. A non-zero would mean we shipped a leak."""
    src = next(BASELINE_DIR.glob("*.agentlog"))
    rows = load_trace_rows(src)
    assert all(r.redactions == 0 for r in rows)


def test_first_divergence_returns_none_for_identical_traces() -> None:
    """Inspecting a trace against itself — no divergence."""
    src = next(BASELINE_DIR.glob("*.agentlog"))
    rows = load_trace_rows(src)
    assert first_divergence_index(rows, rows) is None


def test_first_divergence_finds_real_drift_in_refund_demo() -> None:
    """The refund-causal-diagnosis candidate dropped the `confirm`
    step. Walking baseline ↔ candidate should land on the turn
    where they part ways."""
    base_files = sorted(BASELINE_DIR.glob("*.agentlog"))
    cand_files = sorted(CANDIDATE_DIR.glob("*.agentlog"))
    base_rows = load_trace_rows(base_files[0])
    cand_rows = load_trace_rows(cand_files[0])
    idx = first_divergence_index(base_rows, cand_rows)
    assert idx is not None
    assert idx >= 0


def test_first_divergence_ignores_jittery_fields() -> None:
    """Latency and cost fluctuate run-to-run; a divergence in those
    must NOT count as a behaviour change. The detector is meant for
    structural / decision drift only."""
    base = [
        TraceRow(
            turn=1,
            kind="chat_response",
            role_or_tool="assistant",
            summary="hi",
            input_tokens=10,
            output_tokens=2,
            latency_ms=100,
            cost_usd=0.001,
            redactions=0,
            stop_reason="stop",
            record_id="x",
        ),
    ]
    cand = [
        TraceRow(
            turn=1,
            kind="chat_response",
            role_or_tool="assistant",
            summary="hi",
            input_tokens=10,
            output_tokens=2,
            latency_ms=2000,  # jitter
            cost_usd=0.020,  # jitter
            redactions=0,
            stop_reason="stop",
            record_id="y",
        ),
    ]
    assert first_divergence_index(base, cand) is None


# ---- CLI-level tests ------------------------------------------------------


def _run_inspect(*args: str) -> subprocess.CompletedProcess[str]:
    # Force a wide terminal so rich doesn't word-wrap the column
    # headers across lines (which makes substring assertions
    # fragile across CI runners with different default widths).
    import os as _os

    env = dict(_os.environ)
    env["COLUMNS"] = "200"
    return subprocess.run(
        [sys.executable, "-m", "shadow.cli.app", "inspect", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_inspect_single_trace_prints_table() -> None:
    src = next(BASELINE_DIR.glob("*.agentlog"))
    result = _run_inspect(str(src))
    assert result.returncode == 0, result.stderr
    # Table header columns.
    assert "turn" in result.stdout
    assert "kind" in result.stdout
    assert "summary" in result.stdout
    # First chat record should appear by name.
    assert "chat_request" in result.stdout or "chat_re…" in result.stdout


def test_cli_inspect_comparison_mode_reports_first_divergence() -> None:
    base = sorted(BASELINE_DIR.glob("*.agentlog"))[0]
    cand = sorted(CANDIDATE_DIR.glob("*.agentlog"))[0]
    result = _run_inspect(str(base), str(cand))
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    # Comparison mode includes the divergence summary line.
    assert "first divergence" in combined.lower() or "no divergence" in combined.lower()


def test_cli_inspect_identical_files_shows_no_divergence() -> None:
    src = next(BASELINE_DIR.glob("*.agentlog"))
    result = _run_inspect(str(src), str(src))
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "no divergence" in combined.lower() or "agree end-to-end" in combined.lower()
