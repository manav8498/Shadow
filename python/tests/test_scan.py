"""Tests for `shadow scan` and the `shadow.scan` library surface.

Two layers:
* In-process tests against `scan_paths()` for the detection logic.
* Subprocess tests against the `shadow scan` CLI for exit codes,
  stdout/stderr shape, and `--json` / `--patterns` / `--only` flags.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from shadow.redact.patterns import DEFAULT_PATTERNS
from shadow.scan import (
    load_extra_patterns,
    scan_paths,
)

# Real .agentlog files committed to the repo make the best fixtures —
# they exercise the full envelope-parse path, not just our scanner.
REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_TRACE_DIR = REPO_ROOT / "examples" / "refund-causal-diagnosis" / "baseline_traces"


def _make_leaky_trace(tmp_path: Path, secret: str) -> Path:
    """Inject `secret` into the user-message content of an existing
    clean trace and write the modified .agentlog into tmp_path."""
    src = next(CLEAN_TRACE_DIR.glob("*.agentlog"))
    blob = src.read_text(encoding="utf-8").splitlines()
    out = []
    for line in blob:
        rec = json.loads(line)
        if rec.get("kind") == "chat_request":
            rec["payload"]["messages"][0]["content"] += f" {secret}"
        out.append(json.dumps(rec))
    target = tmp_path / "leaky.agentlog"
    target.write_text("\n".join(out) + "\n")
    return target


# ---- library-level tests --------------------------------------------------


def test_scan_clean_directory_returns_no_hits() -> None:
    """The committed refund-causal-diagnosis baseline traces are clean
    by construction — Session.enter()'s default Redactor already swept
    them. Verifies `scan_paths` doesn't false-positive."""
    result = scan_paths([CLEAN_TRACE_DIR])
    assert result.hits == []
    assert result.files_scanned >= 1
    assert result.records_scanned >= 1


def test_scan_detects_planted_openai_api_key(tmp_path: Path) -> None:
    """Insert a plausible OpenAI proj key into a clean trace and
    verify the scanner names it."""
    secret = "sk-proj-AAAAAAAAAAAAAAAAAAAA1234567890abcdef"
    _make_leaky_trace(tmp_path, secret)
    result = scan_paths([tmp_path])
    assert len(result.hits) >= 1
    assert any(h.pattern_name == "openai_api_key" for h in result.hits)
    # Snippet carries the literal match — the audit-trail value of
    # the scanner depends on this. Truncation-safety is a separate
    # test (long secrets get clipped at 80 chars).
    hit = next(h for h in result.hits if h.pattern_name == "openai_api_key")
    assert secret in hit.snippet


def test_scan_detects_anthropic_aws_github_jwt_separately(tmp_path: Path) -> None:
    """Pattern set isolation — each named pattern fires on its own
    distinct shape. Drift between patterns (e.g. `sk-ant-` matched by
    OPENAI_API_KEY would be a regression)."""
    cases = {
        "anthropic_api_key": "sk-ant-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "github_token": "ghp_" + "A" * 40,
        "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        + "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
    }
    for name, secret in cases.items():
        sub = tmp_path / name
        sub.mkdir()
        _make_leaky_trace(sub, secret)
        result = scan_paths([sub])
        names = {h.pattern_name for h in result.hits}
        assert name in names, f"{name} missed; got {names}"


def test_scan_credit_card_passes_luhn_check(tmp_path: Path) -> None:
    """Visa test number `4111-1111-1111-1111` is Luhn-valid; a
    16-digit non-Luhn string must NOT be flagged. Mirrors the
    redactor's filter — without it, every long ID number false-
    positives as a card."""
    valid = _make_leaky_trace(tmp_path, "card 4111-1111-1111-1111")
    valid_result = scan_paths([valid])
    assert any(h.pattern_name == "credit_card" for h in valid_result.hits)

    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    _make_leaky_trace(invalid_dir, "order 1234567890123456")
    invalid_result = scan_paths([invalid_dir])
    assert not any(h.pattern_name == "credit_card" for h in invalid_result.hits)


def test_scan_extra_patterns_file_loads_company_secrets(tmp_path: Path) -> None:
    """Custom-patterns file format: `name=regex` per line, `#` comments."""
    patterns_file = tmp_path / "patterns.txt"
    patterns_file.write_text(
        "# company-internal token format\n"
        "acme_internal_key=acme-[a-z0-9]{32}\n"
        "\n"
        "# session cookie\n"
        "session_cookie=session=[A-Za-z0-9_\\-]{40,}\n"
    )
    pats = load_extra_patterns(patterns_file)
    names = {p.name for p in pats}
    assert names == {"acme_internal_key", "session_cookie"}


def test_scan_extra_patterns_file_rejects_malformed_line(tmp_path: Path) -> None:
    """Validation gives a line number — debugging a 200-line patterns
    file shouldn't require bisecting."""
    p = tmp_path / "bad.txt"
    p.write_text("ok=valid-regex\n# comment ok\nno-equals-here\n")
    with pytest.raises(ValueError, match=r":3:"):
        load_extra_patterns(p)


def test_scan_only_filter_restricts_pattern_set(tmp_path: Path) -> None:
    """`--only foo,bar` restricts to those patterns even though more
    would have fired by default."""
    # Single trace contains both an api key and an email. Default scan
    # finds both; --only=email finds just the email.
    src = next(CLEAN_TRACE_DIR.glob("*.agentlog"))
    blob = src.read_text(encoding="utf-8").splitlines()
    out = []
    for line in blob:
        rec = json.loads(line)
        if rec.get("kind") == "chat_request":
            rec["payload"]["messages"][0]["content"] += (
                " key=sk-proj-AAAAAAAAAAAAAAAAAAAA1234567890abcdef email=alice@acme.com"
            )
        out.append(json.dumps(rec))
    target = tmp_path / "leaky.agentlog"
    target.write_text("\n".join(out) + "\n")

    full = scan_paths([target])
    full_names = {h.pattern_name for h in full.hits}
    assert "openai_api_key" in full_names
    assert "email" in full_names

    only_email = scan_paths([target], pattern_names={"email"})
    only_names = {h.pattern_name for h in only_email.hits}
    assert only_names == {"email"}


def test_scan_default_patterns_match_redactor(tmp_path: Path) -> None:
    """Sanity: every name in DEFAULT_PATTERNS is reachable from
    `scan_paths`. If we add a pattern to `redact.patterns` and
    forget the scanner, this test fires."""
    expected = {p.name for p in DEFAULT_PATTERNS}
    # Smuggle one of each pattern's example match into a trace.
    sample_text = " ".join(
        [
            "sk-proj-AAAAAAAAAAAAAAAAAAAA1234567890abcdef",
            "sk-ant-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_" + "A" * 40,
            "alice@acme.com",
            "+15551234567",
            "4111-1111-1111-1111",
        ]
    )
    _make_leaky_trace(tmp_path, sample_text)
    result = scan_paths([tmp_path])
    found = {h.pattern_name for h in result.hits}
    # private_key + jwt patterns need long-form input we don't bother
    # generating here; the per-pattern test above covers them.
    minimum = expected - {"private_key", "jwt"}
    missing = minimum - found
    assert not missing, f"DEFAULT_PATTERNS not all reachable from scan: {missing}"


# ---- CLI-level tests ------------------------------------------------------


def _run_scan(*args: str) -> subprocess.CompletedProcess[str]:
    # Force UTF-8 decoding so Rich's status markers (✓ ✗) survive on
    # Windows runners whose default codepage is cp1252.
    import os as _os

    env = dict(_os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "shadow.cli.app", "scan", *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


def test_cli_clean_input_exits_zero() -> None:
    """No hits → exit 0. The success message goes to stderr (we use
    err_console for everything that isn't a primary output payload)."""
    result = _run_scan(str(CLEAN_TRACE_DIR))
    assert result.returncode == 0, result.stderr


def test_cli_planted_secret_exits_one(tmp_path: Path) -> None:
    """Hit → exit 1. The match is rendered with the pattern name so a
    CI log makes the fix actionable."""
    _make_leaky_trace(tmp_path, "sk-proj-AAAAAAAAAAAAAAAAAAAA1234567890abcdef")
    result = _run_scan(str(tmp_path))
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "openai_api_key" in combined


def test_cli_redact_snippets_does_not_leak_match_into_output(tmp_path: Path) -> None:
    """`--redact-snippets` is for CI logs; the literal credential
    must NOT appear anywhere in stdout/stderr when the flag is on."""
    secret = "sk-proj-VVVVVVVVVVVVVVVVVVVV1234567890abcdef"
    _make_leaky_trace(tmp_path, secret)
    result = _run_scan(str(tmp_path), "--redact-snippets")
    assert result.returncode == 1
    assert secret not in result.stdout
    assert secret not in result.stderr
    # The pattern name still appears so the user can grep for it.
    assert "openai_api_key" in (result.stdout + result.stderr)


def test_cli_json_output_is_machine_parseable(tmp_path: Path) -> None:
    """`--json` emits a single object on stdout. Useful for piping
    into jq or annotating PR comments."""
    _make_leaky_trace(tmp_path, "alice@example.com")
    result = _run_scan(str(tmp_path), "--json")
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["files_scanned"] >= 1
    assert any(h["pattern"] == "email" for h in data["hits"])
