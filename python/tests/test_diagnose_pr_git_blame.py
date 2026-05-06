"""Tests for the line-level prompt-blame helper.

Covers the parser (deterministic — no git invocation) and the
end-to-end shape against a real `git init` workspace populated
inside `tmp_path`. The parser tests are the most important ones:
they pin the hunk-header regex against weird-but-valid unified
diff shapes (single-line hunks, additions before removals,
context interleaving, pure-addition fallback).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from shadow.diagnose_pr.git_blame import (
    PromptHunk,
    _parse_first_added_line,
    _parse_first_removed_line,
    blame_prompt_files,
)

# ---- parser unit tests -----------------------------------------------------


def test_parse_simple_replacement_returns_paired_lines() -> None:
    diff = (
        "diff --git a/p.md b/p.md\n"
        "--- a/p.md\n"
        "+++ b/p.md\n"
        "@@ -16,3 +16,3 @@\n"
        "  context above\n"
        "- always confirm before refunding\n"
        "+ refund without confirmation\n"
        "  context below\n"
    )
    parsed = _parse_first_removed_line(diff)
    assert parsed is not None
    line_no, removed, added = parsed
    assert line_no == 17  # 1 context line consumed before the removed hunk-line
    assert removed.strip() == "always confirm before refunding"
    assert added.strip() == "refund without confirmation"


def test_parse_pure_deletion_returns_no_added_text() -> None:
    diff = "@@ -10,2 +10,1 @@\n" "  surrounding line\n" "- removed instruction\n"
    parsed = _parse_first_removed_line(diff)
    assert parsed is not None
    line_no, removed, added = parsed
    assert line_no == 11
    assert removed.strip() == "removed instruction"
    assert added is None


def test_parse_pure_addition_falls_through_to_added_helper() -> None:
    diff = "@@ -10,1 +10,2 @@\n" "  context\n" "+ a brand new instruction\n"
    # No removed lines — primary parser returns None.
    assert _parse_first_removed_line(diff) is None
    # Fallback parser should find the addition with the candidate-side line no.
    fallback = _parse_first_added_line(diff)
    assert fallback is not None
    line_no, added = fallback
    assert line_no == 11
    assert added.strip() == "a brand new instruction"


def test_parse_handles_single_line_hunk_header_without_comma() -> None:
    """`@@ -47 +47 @@` — `,len` is omitted when len == 1."""
    diff = "@@ -47 +47 @@\n- Always confirm.\n+ Confirm sometimes.\n"
    parsed = _parse_first_removed_line(diff)
    assert parsed is not None
    line_no, removed, added = parsed
    assert line_no == 47
    assert removed.strip() == "Always confirm."
    assert added.strip() == "Confirm sometimes."


def test_parse_returns_first_removal_when_multiple_hunks_present() -> None:
    diff = (
        "@@ -5,3 +5,3 @@\n"
        "  ctx\n"
        "- first removed\n"
        "+ first added\n"
        "@@ -50,3 +50,3 @@\n"
        "  ctx\n"
        "- second removed\n"
        "+ second added\n"
    )
    parsed = _parse_first_removed_line(diff)
    assert parsed is not None
    _line_no, removed, _added = parsed
    assert removed.strip() == "first removed"


def test_parse_returns_none_for_diff_with_no_changes() -> None:
    diff = "  context only\n  more context\n"
    assert _parse_first_removed_line(diff) is None
    assert _parse_first_added_line(diff) is None


# ---- end-to-end with a real git workspace ----------------------------------


def _git_available() -> bool:
    return shutil.which("git") is not None


@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
def test_blame_prompt_files_finds_removed_instruction(tmp_path: Path) -> None:
    """Realistic shape: `git init`, commit a prompt, mutate it,
    verify blame_prompt_files returns the right hunk."""
    # Initialise a throwaway repo.
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Shadow Test"], cwd=tmp_path, check=True)

    prompt_path = tmp_path / "prompts" / "refund.md"
    prompt_path.parent.mkdir()
    prompt_path.write_text(
        "You are a refund agent.\n"
        "When a user requests a refund:\n"
        "1. Look up the order.\n"
        "2. Always confirm the refund amount before issuing the refund.\n"
        "3. Issue the refund.\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path, check=True)
    baseline_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()

    # Candidate edit: drop the confirmation step.
    prompt_path.write_text(
        "You are a refund agent.\n"
        "When a user requests a refund:\n"
        "1. Look up the order.\n"
        "2. Issue the refund.\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "candidate: drop confirmation"], cwd=tmp_path, check=True
    )

    blame = blame_prompt_files(
        repo_root=tmp_path,
        baseline_ref=baseline_sha,
        paths=["prompts/refund.md"],
    )
    assert "prompts/refund.md" in blame
    hunk = blame["prompts/refund.md"]
    assert isinstance(hunk, PromptHunk)
    assert hunk.line_no >= 1
    assert hunk.removed_text is not None
    assert "Always confirm the refund amount" in hunk.removed_text


def test_blame_returns_empty_when_baseline_ref_does_not_exist(tmp_path: Path) -> None:
    """Bad refs are not fatal — caller falls back to flat config-path
    attribution. blame_prompt_files swallows the failure."""
    blame = blame_prompt_files(
        repo_root=tmp_path,
        baseline_ref="nope-not-a-ref",
        paths=["prompts/refund.md"],
    )
    assert blame == {}


def test_blame_skips_non_prompt_paths() -> None:
    """The path heuristic restricts to .md/.txt/.prompt or paths
    containing 'prompt'. Other file types should never be diff'd."""
    blame = blame_prompt_files(
        repo_root=Path.cwd(),
        baseline_ref="HEAD",
        paths=["src/main.rs", "package.json"],
    )
    assert blame == {}
