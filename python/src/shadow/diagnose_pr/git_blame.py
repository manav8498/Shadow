"""Git-diff hunk extraction for prompt-level blame.

When a PR changes a prompt file, `extract_deltas()` can only see
"prompt.system contents differ" — not which line of which file
moved, or what instruction was removed. That makes the PR comment
say `prompts/candidate.md` when the user wants to read
`prompts/refund.md:17 removed: "Always confirm the refund amount
before issuing refund."`.

This module bridges that gap. Given:

  * `repo_root` — the working tree we can run `git -C` against;
  * `baseline_ref` — what the candidate is being compared to
    (typically `origin/main` or the PR base SHA);
  * `paths` — files that the PR touched (from `git diff --name-only`);

it returns a `dict[str, PromptHunk]` keyed by file path, where each
`PromptHunk` carries the *first* removed line + its line number +
the corresponding added line (best-effort one-to-one within the
hunk). v1 surfaces only the first removed line per file — that's
the load-bearing instruction in 90% of prompt-edit regressions
(the others tend to be additions, which surface via `added_text`
when no removal exists).

Subprocess hygiene:
  * No shell — argv list only, fixed git binary lookup via PATH.
  * 5-second timeout: a single git-diff on a prompt file should
    return in milliseconds; anything longer is a configuration
    issue we'd rather surface as "no blame" than hang the PR run.
  * stderr captured — never leaked to the user's stdout.
  * Non-existent baseline ref / non-git repo / git unavailable: all
    return an empty mapping so the caller falls back to flat
    config-path attribution. Blame is best-effort.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# A unified-diff "hunk header" line.
#   @@ -<old-start>,<old-len> +<new-start>,<new-len> @@ <ctx>
# `,len` is optional when len == 1.
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

# How long to wait for a single `git diff` to return. Prompt files
# are small (KBs); 5 s is generous.
_GIT_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class PromptHunk:
    """First-removed-line blame for one changed prompt file.

    `line_no` is the 1-based line number on the *baseline* side —
    that's the line the reader would `git blame` against to find
    the original author. `added_text` is whichever line replaced
    it on the candidate side (best-effort one-to-one within the
    hunk; None when the hunk is a pure deletion or pure addition).
    """

    file_path: str
    line_no: int
    removed_text: str | None
    added_text: str | None


def _is_prompt_path(path: str) -> bool:
    """Heuristic: a file path that looks like a prompt asset.

    Matches the same shape `extract_deltas()` already uses to attach
    `--changed-files` filenames — `.md` extension, or contains
    `prompt` in the basename or any directory component.
    """
    p = path.lower()
    if p.endswith((".md", ".txt", ".prompt")):
        return True
    return "prompt" in p


def _git_available() -> bool:
    """Cheap check for git-on-PATH. Avoids subprocess startup cost
    when blame is impossible (CI runner with no git, sandboxed env)."""
    return shutil.which("git") is not None


def _git_diff_hunks(
    *,
    repo_root: Path,
    baseline_ref: str,
    file_path: str,
) -> str | None:
    """Return the unified-diff text for one file vs `baseline_ref`,
    or None if anything goes wrong (no such ref, file untracked,
    git missing, timeout, working tree not a git repo).

    Uses `--no-color` to keep parsing predictable across user
    git configs that force color output. `-U0` would lose context
    lines but also collapse adjacent edits; we use the default
    `-U3` so the hunk header reflects the surrounding region.
    """
    if not _git_available():
        return None
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "diff",
                "--no-color",
                "--no-ext-diff",
                f"{baseline_ref}...HEAD",
                "--",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        # Common non-zero cases: bad ref, file untracked, repo missing.
        # We don't need to distinguish — caller gets a None, attribution
        # falls back gracefully.
        return None
    return result.stdout


def _parse_first_removed_line(diff_text: str) -> tuple[int, str | None, str | None] | None:
    """Walk a unified-diff body and return (line_no, removed,
    added) for the FIRST removal we find. `line_no` is 1-based on
    the baseline side. Returns None when the diff has no removals
    (pure addition).

    Pairing rule for `added`: scan forward within the same hunk
    after the removal; the next line starting with `+` (and not
    `+++`) is paired as the replacement. Pure deletes return
    (line_no, removed_text, None).
    """
    in_hunk = False
    base_cursor = 0
    pending_removed: tuple[int, str] | None = None
    for raw in diff_text.splitlines():
        m = _HUNK_RE.match(raw)
        if m is not None:
            in_hunk = True
            base_cursor = int(m.group(1))
            pending_removed = None
            continue
        if not in_hunk:
            continue
        if raw.startswith("---") or raw.startswith("+++"):
            # Header lines, not hunk content.
            continue
        if raw.startswith("-"):
            if pending_removed is None:
                pending_removed = (base_cursor, raw[1:])
            base_cursor += 1
            continue
        if raw.startswith("+"):
            if pending_removed is not None:
                line_no, removed = pending_removed
                return line_no, removed, raw[1:]
            # Pure addition without a pending removal — keep walking;
            # we prefer to surface a removal/replacement pair when the
            # diff has one.
            continue
        # Context line (' ') or blank — advance baseline cursor.
        base_cursor += 1
        # If we've seen a pure removal and the hunk ended without a
        # paired addition, surface it now.
        if pending_removed is not None:
            line_no, removed = pending_removed
            return line_no, removed, None
    if pending_removed is not None:
        line_no, removed = pending_removed
        return line_no, removed, None
    return None


def _parse_first_added_line(diff_text: str) -> tuple[int, str] | None:
    """Fallback path for hunks that are pure additions (no removed
    lines at all). Returns (candidate-side line number, added text)
    — useful when the user added an instruction whose absence is
    the regression cause.
    """
    in_hunk = False
    cand_cursor = 0
    for raw in diff_text.splitlines():
        m = _HUNK_RE.match(raw)
        if m is not None:
            in_hunk = True
            cand_cursor = int(m.group(2))
            continue
        if not in_hunk:
            continue
        if raw.startswith("---") or raw.startswith("+++"):
            continue
        if raw.startswith("+"):
            return cand_cursor, raw[1:]
        if raw.startswith("-"):
            # We only reach this branch when the diff had no addition
            # we can pair to a removal — fall through to next line.
            continue
        cand_cursor += 1
    return None


def blame_prompt_files(
    *,
    repo_root: Path,
    baseline_ref: str,
    paths: list[str],
) -> dict[str, PromptHunk]:
    """Run `git diff <ref>...HEAD <path>` for each prompt-shaped
    path in `paths` and return a path→PromptHunk mapping.

    Only paths that pass `_is_prompt_path()` are diff'd — running
    git on every changed file in a large PR is wasteful, and the
    blame fields are only useful for prompt-kind deltas anyway.
    """
    out: dict[str, PromptHunk] = {}
    for path in paths:
        if not _is_prompt_path(path):
            continue
        diff_text = _git_diff_hunks(
            repo_root=repo_root,
            baseline_ref=baseline_ref,
            file_path=path,
        )
        if not diff_text:
            continue
        parsed = _parse_first_removed_line(diff_text)
        if parsed is not None:
            line_no, removed, added = parsed
            out[path] = PromptHunk(
                file_path=path,
                line_no=line_no,
                removed_text=removed,
                added_text=added,
            )
            continue
        added_only = _parse_first_added_line(diff_text)
        if added_only is not None:
            line_no, added = added_only
            out[path] = PromptHunk(
                file_path=path,
                line_no=line_no,
                removed_text=None,
                added_text=added,
            )
    return out


__all__ = [
    "PromptHunk",
    "blame_prompt_files",
]
