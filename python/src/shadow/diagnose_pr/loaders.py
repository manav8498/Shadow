"""Loaders for `shadow diagnose-pr`.

Two responsibilities:
  * Reading a YAML config (same schema as `shadow replay`).
  * Reading one or more `.agentlog` files into typed `LoadedTrace`
    records, with the file path preserved for downstream rendering.

Both raise `ShadowConfigError` / `ShadowParseError` from
`shadow.errors` so the CLI can produce a clean diagnostic without a
stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from shadow import _core
from shadow.errors import ShadowConfigError, ShadowParseError


@dataclass(frozen=True)
class LoadedTrace:
    """One parsed `.agentlog` file with its source path preserved.

    `trace_id` is the metadata record id (the trace's content
    address); see SPEC.md §8.1. `records` is the full envelope list
    in file order.
    """

    path: Path
    trace_id: str
    records: list[dict[str, Any]]


def load_config(path: Path) -> dict[str, Any]:
    """Read a YAML config file and return its parsed dict.

    The schema matches `shadow replay`'s baseline-config: top-level
    `model`, `params`, `prompt`, `tools`. We validate readability and
    YAML syntax; semantic validation lives in the delta extractor.
    """
    if not path.is_file():
        raise ShadowConfigError(f"config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ShadowConfigError(f"could not read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ShadowConfigError(f"could not parse {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ShadowConfigError(
            f"{path}: top-level must be a mapping, got {type(data).__name__}"
        )
    return data


def load_traces(paths: list[Path]) -> list[LoadedTrace]:
    """Load every `.agentlog` file under the given paths.

    Each `paths` entry is either:
      * a `.agentlog` file — loaded directly;
      * a directory — globbed recursively for `*.agentlog`;
      * any other file — silently ignored (the caller may pass a
        directory of mixed contents).

    Result order is sorted by absolute file path for determinism.
    """
    files: list[Path] = []
    for p in paths:
        if p.is_file():
            if p.suffix == ".agentlog":
                files.append(p)
            # non-.agentlog file: ignore
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.agentlog")))
        else:
            raise ShadowConfigError(f"path does not exist: {p}")
    files.sort()

    out: list[LoadedTrace] = []
    for f in files:
        try:
            blob = f.read_bytes()
        except OSError as exc:
            raise ShadowConfigError(f"could not read {f}: {exc}") from exc
        try:
            records = _core.parse_agentlog(blob)
        except Exception as exc:  # _core raises a typed error from Rust
            raise ShadowParseError(f"could not parse {f}: {exc}") from exc

        if not records:
            raise ShadowParseError(f"{f}: empty .agentlog (no records)")
        trace_id = str(records[0].get("id", ""))
        if not trace_id:
            raise ShadowParseError(f"{f}: first record missing id")
        out.append(LoadedTrace(path=f, trace_id=trace_id, records=records))
    return out


__all__ = ["LoadedTrace", "load_config", "load_traces"]
