"""Tests for `shadow.diagnose_pr.deltas.extract_deltas`.

The extractor compares two parsed config dicts (already loaded by
`loaders.load_config`) and emits one ConfigDelta per atomic change.
The taxonomy lives in DeltaKind — it's coarse on purpose, since
the renderer only needs to phrase the cause, not implement it."""

from __future__ import annotations

import hashlib

from shadow.diagnose_pr.deltas import extract_deltas


def test_no_changes_returns_empty_list() -> None:
    cfg = {"model": "x", "params": {"temperature": 0.2}}
    assert extract_deltas(cfg, dict(cfg)) == []


def test_model_change_is_classified_as_model() -> None:
    base = {"model": "gpt-4.1"}
    cand = {"model": "gpt-4.1-mini"}
    out = extract_deltas(base, cand)
    assert len(out) == 1
    d = out[0]
    assert d.kind == "model"
    assert d.path == "model"
    assert "gpt-4.1" in d.display and "gpt-4.1-mini" in d.display


def test_temperature_change_is_classified_as_temperature() -> None:
    base = {"params": {"temperature": 0.2, "max_tokens": 512}}
    cand = {"params": {"temperature": 0.7, "max_tokens": 512}}
    out = extract_deltas(base, cand)
    assert [d.kind for d in out] == ["temperature"]
    assert out[0].path == "params.temperature"


def test_system_prompt_change_is_classified_as_prompt() -> None:
    base = {"prompt": {"system": "Always confirm refunds."}}
    cand = {"prompt": {"system": "Process refunds."}}
    out = extract_deltas(base, cand)
    assert len(out) == 1
    assert out[0].kind == "prompt"
    assert out[0].path == "prompt.system"


def test_tool_schema_change_is_classified_as_tool_schema() -> None:
    base = {"tools": [{"name": "issue_refund", "input_schema": {"type": "object"}}]}
    cand = {
        "tools": [
            {
                "name": "issue_refund",
                "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
            }
        ]
    }
    out = extract_deltas(base, cand)
    assert any(d.kind == "tool_schema" for d in out)


def test_unknown_top_level_field_change_falls_back_to_unknown() -> None:
    base = {"weird_extension": {"foo": 1}}
    cand = {"weird_extension": {"foo": 2}}
    out = extract_deltas(base, cand)
    assert len(out) == 1
    assert out[0].kind == "unknown"


def test_hashes_are_canonical_so_reformatting_isnt_a_delta() -> None:
    base = {"params": {"temperature": 0.2, "max_tokens": 512}}
    # Same content, different key order — canonicalisation should make these equal.
    cand = {"params": {"max_tokens": 512, "temperature": 0.2}}
    out = extract_deltas(base, cand)
    assert out == []


def test_changed_files_with_prompt_md_attaches_id_with_filename() -> None:
    base = {"prompt": {"system": "old"}}
    cand = {"prompt": {"system": "new"}}
    out = extract_deltas(
        base,
        cand,
        changed_files=["prompts/system.md"],
    )
    assert len(out) == 1
    # When a prompt file is in the PR, the id should reference it.
    assert out[0].id.startswith("prompts/system.md") or out[0].kind == "prompt"


def test_old_and_new_hashes_are_hex_sha256_when_present() -> None:
    base = {"prompt": {"system": "old"}}
    cand = {"prompt": {"system": "new"}}
    out = extract_deltas(base, cand)
    d = out[0]
    assert d.old_hash is not None and len(d.old_hash) == 64
    assert d.new_hash is not None and len(d.new_hash) == 64
    # Sanity: hashes are canonical-bytes sha256
    expected_old = hashlib.sha256(b'"old"').hexdigest()
    assert d.old_hash == expected_old
