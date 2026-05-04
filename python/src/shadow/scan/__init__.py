"""Trace-content scanner for credentials, PII, and custom secret patterns.

Companion to `shadow.redact`. The Redactor *substitutes* matches at
write time so traces written through a `Session` never contain
credentials. This module *detects* matches at read time so you can
audit traces that were already committed to the repo:

    shadow scan path/to/traces/             # scan a directory
    shadow scan baseline.agentlog           # scan one file
    shadow scan -p custom-secrets.txt ...   # extra patterns

Exits non-zero on any hit so it composes cleanly into CI:

    shadow scan baseline_traces/ candidate_traces/  # exit 1 on hit
    shadow gate-pr ...                              # gated downstream

The default pattern set is `shadow.redact.DEFAULT_PATTERNS` — the
same patterns the Redactor uses to substitute, run in detect-only
mode here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from shadow import _core
from shadow.redact.patterns import DEFAULT_PATTERNS, Pattern, luhn_valid

__all__ = ["Hit", "ScanResult", "load_extra_patterns", "scan_paths"]


@dataclass(frozen=True)
class Hit:
    """One match found inside a trace.

    `snippet` carries the matched substring directly so the user can
    eyeball the hit; we DO NOT redact it in the scan output because
    the whole point is "show me what was leaked." For machine-readable
    workflows that don't want the literal credential in CI logs, the
    `--redact-snippets` flag on the CLI replaces the snippet with the
    pattern's `[REDACTED:<name>]` token.
    """

    file_path: str
    """Path to the .agentlog file the hit was found in."""
    record_id: str
    """Content id of the record (e.g. `sha256:abc...`)."""
    record_kind: str
    """`metadata` / `chat_request` / `chat_response` / etc."""
    pattern_name: str
    """Which pattern matched — e.g. `openai_api_key`, `email`."""
    snippet: str
    """The exact match text. Truncated to 80 chars."""


@dataclass(frozen=True)
class ScanResult:
    files_scanned: int
    records_scanned: int
    hits: list[Hit]


def _make_extra_pattern(name: str, pattern_src: str) -> Pattern:
    """Compile one user-supplied regex into a Pattern. Replacement is
    `[REDACTED:<name>]` for symmetry with DEFAULT_PATTERNS."""
    return Pattern(
        name=name,
        regex=re.compile(pattern_src),
        replacement=f"[REDACTED:{name}]",
    )


def load_extra_patterns(path: Path) -> list[Pattern]:
    """Parse a custom-patterns file into Pattern objects.

    File format: one rule per line, in `<name>=<regex>` form. Lines
    beginning with `#` are treated as comments. Blank lines ignored.

    Example::

        # company-internal token format
        acme_internal_key=acme-[a-z0-9]{32}
        # bearer-style cookie
        session_cookie=session=[A-Za-z0-9_\\-]{40,}

    Raises `ValueError` on malformed lines so the user sees the line
    number on the very first invalid entry rather than running with a
    silently-empty pattern list.
    """
    patterns: list[Pattern] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"{path}:{lineno}: expected `name=regex`, got: {line!r}\n"
                "  hint: use `# comment` for comments and `name=regex` for patterns."
            )
        name, regex_src = line.split("=", 1)
        name = name.strip()
        regex_src = regex_src.strip()
        if not name or not regex_src:
            raise ValueError(f"{path}:{lineno}: empty name or pattern: {line!r}")
        try:
            patterns.append(_make_extra_pattern(name, regex_src))
        except re.error as exc:
            raise ValueError(f"{path}:{lineno}: invalid regex {regex_src!r}: {exc}") from exc
    return patterns


def _iter_agentlog_files(roots: list[Path]) -> list[Path]:
    """Expand directories recursively; pass through files. Sorted for
    deterministic output."""
    out: list[Path] = []
    for root in roots:
        if root.is_dir():
            out.extend(sorted(root.rglob("*.agentlog")))
        elif root.is_file():
            out.append(root)
        else:
            raise FileNotFoundError(f"path does not exist: {root}")
    return out


def _scan_record_payload(
    payload: object,
    patterns: tuple[Pattern, ...],
) -> list[tuple[str, str]]:
    """Stringify the payload as canonical JSON and run every pattern.
    Returns list of `(pattern_name, snippet)` for each match.

    Canonical JSON is what the Redactor sees too — keeping the input
    surface aligned means a Redactor configured with the same pattern
    set would have substituted exactly these spans at write time. So
    a hit here = "this trace was written by a Session that didn't
    redact this pattern."
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    hits: list[tuple[str, str]] = []
    for pat in patterns:
        for m in pat.regex.finditer(text):
            # Credit-card pattern is regex + Luhn (matches the Redactor).
            if pat.name == "credit_card":
                digits = re.sub(r"[^0-9]", "", m.group(0))
                if not luhn_valid(digits):
                    continue
            snippet = m.group(0)
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            hits.append((pat.name, snippet))
    return hits


def scan_paths(
    paths: list[Path],
    *,
    extra_patterns: list[Pattern] | None = None,
    pattern_names: set[str] | None = None,
) -> ScanResult:
    """Scan every `.agentlog` file under `paths` for secret/PII patterns.

    `paths` may mix files and directories. Directories are walked
    recursively for `*.agentlog`. Each record's payload is canonicalised
    to JSON and matched against the pattern set.

    `extra_patterns` adds user-supplied rules on top of DEFAULT_PATTERNS.
    `pattern_names`, if set, restricts to patterns whose `name` is in
    the set (useful for `--only openai_api_key,email`).
    """
    files = _iter_agentlog_files(paths)
    all_patterns = list(DEFAULT_PATTERNS) + list(extra_patterns or [])
    if pattern_names is not None:
        all_patterns = [p for p in all_patterns if p.name in pattern_names]
    pattern_tuple = tuple(all_patterns)

    hits: list[Hit] = []
    records_scanned = 0
    for f in files:
        try:
            blob = f.read_bytes()
        except OSError as exc:
            raise OSError(f"could not read {f}: {exc}") from exc
        try:
            records = _core.parse_agentlog(blob)
        except Exception as exc:
            raise ValueError(f"could not parse {f} as .agentlog: {exc}") from exc
        for rec in records:
            records_scanned += 1
            for name, snippet in _scan_record_payload(rec.get("payload"), pattern_tuple):
                hits.append(
                    Hit(
                        file_path=str(f),
                        record_id=str(rec.get("id", "")),
                        record_kind=str(rec.get("kind", "")),
                        pattern_name=name,
                        snippet=snippet,
                    )
                )
    return ScanResult(
        files_scanned=len(files),
        records_scanned=records_scanned,
        hits=hits,
    )
