"""Config-delta extraction.

Compares two parsed YAML configs (baseline + candidate) and emits
one `ConfigDelta` per atomic change. The taxonomy is coarse on
purpose — `DeltaKind` has seven values; the renderer only needs to
phrase the cause, not decide what to do about it.

Canonicalisation matters: a trivially-reformatted YAML (key reorder,
whitespace) must NOT register as a delta. We canonicalise to JSON
with sorted keys and tight separators (matching
`shadow.causal.replay.openai_replayer._canonical_config_hash`'s
shape) before hashing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from shadow.diagnose_pr.models import ConfigDelta, DeltaKind


def _canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes — sorted keys, no whitespace, UTF-8.

    Same shape as `shadow.causal.replay.openai_replayer.
    _canonical_config_hash`; we share a hash space so future tooling
    can correlate.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


_TOP_LEVEL_KIND: dict[str, DeltaKind] = {
    "model": "model",
    "prompt": "prompt",
    "tools": "tool_schema",
    "retriever": "retriever",
}


def _kind_for_path(path: str) -> DeltaKind:
    """Map a dotted config path to its DeltaKind.

    Examples:
        params.temperature -> temperature
        prompt.system      -> prompt
        tools              -> tool_schema
        model              -> model
    """
    if path == "params.temperature":
        return "temperature"
    head = path.split(".", 1)[0]
    return _TOP_LEVEL_KIND.get(head, "unknown")


def _walk_diff(
    base: Any,
    cand: Any,
    path: str,
    out: list[tuple[str, Any, Any]],
) -> None:
    """Depth-first walk emitting (path, old, new) for every leaf
    difference. Dicts recurse on shared keys; missing keys count as
    leaf-level changes (None on the missing side)."""
    if isinstance(base, dict) and isinstance(cand, dict):
        keys = set(base) | set(cand)
        for k in sorted(keys):
            sub = f"{path}.{k}" if path else k
            _walk_diff(base.get(k), cand.get(k), sub, out)
        return
    if _canonical_bytes(base) == _canonical_bytes(cand):
        return
    out.append((path, base, cand))


def _is_prompt_path(path: str) -> bool:
    """Heuristic: any path under `prompt.*` is a prompt change."""
    return path == "prompt" or path.startswith("prompt.")


def _format_display(path: str, old: Any, new: Any) -> str:
    if isinstance(old, str) and isinstance(new, str) and len(old) <= 40 and len(new) <= 40:
        return f"{path}: {old!r} → {new!r}"
    if isinstance(old, (int, float, bool)) and isinstance(new, (int, float, bool)):
        return f"{path}: {old} → {new}"
    return f"{path} (changed)"


def extract_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    changed_files: list[str] | None = None,
) -> list[ConfigDelta]:
    """Extract atomic config deltas from a baseline/candidate pair.

    `changed_files` is a list of file paths the PR touched (e.g. the
    output of `git diff --name-only`). When a prompt change is
    detected and a file in `changed_files` matches a prompt path, we
    attach that filename to the delta `id` so the PR comment can
    cite "prompts/system.md" rather than just "prompt.system".
    """
    leaves: list[tuple[str, Any, Any]] = []
    _walk_diff(baseline, candidate, "", leaves)

    # Coalesce paths that should be grouped at a coarser level. v1
    # rule: a change anywhere under `prompt.*` is one prompt delta;
    # anywhere under `tools.*` is one tool_schema delta.
    grouped: dict[str, tuple[Any, Any]] = {}
    for path, old, new in leaves:
        if _is_prompt_path(path):
            if "prompt.system" not in grouped:
                grouped["prompt.system"] = (
                    baseline.get("prompt", {}).get("system"),
                    candidate.get("prompt", {}).get("system"),
                )
        elif path == "tools" or path.startswith("tools."):
            grouped.setdefault("tools", (baseline.get("tools"), candidate.get("tools")))
        else:
            grouped[path] = (old, new)

    out: list[ConfigDelta] = []
    prompt_file = next(
        (f for f in (changed_files or []) if f.endswith(".md") or "prompt" in f.lower()),
        None,
    )
    for path, (old, new) in sorted(grouped.items()):
        kind = _kind_for_path(path)
        if kind == "prompt" and prompt_file is not None:
            delta_id = prompt_file
        elif kind == "model":
            delta_id = f"model:{old}->{new}"
        elif kind == "temperature":
            delta_id = f"params.temperature:{old}->{new}"
        else:
            delta_id = path
        out.append(
            ConfigDelta(
                id=delta_id,
                kind=kind,
                path=path,
                old_hash=_hash(old) if old is not None else None,
                new_hash=_hash(new) if new is not None else None,
                display=_format_display(path, old, new),
            )
        )
    return out


__all__ = ["extract_deltas"]
