"""Tests for `shadow baseline create / update / approve / verify`.

Two layers:
* In-process tests against the `shadow.baseline` library functions
  (`compute_baseline_hash`, `load_shadow_yaml`, …).
* Subprocess tests against `shadow baseline ...` for end-to-end
  workflow correctness (the friction-flag rules, the file-copy
  semantics of `approve`, etc.).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from shadow.baseline import (
    ShadowConfig,
    compute_baseline_hash,
    load_shadow_yaml,
    save_shadow_yaml,
    verify_baseline_pinned,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_FIXTURE = REPO_ROOT / "examples" / "refund-causal-diagnosis" / "baseline_traces"
CANDIDATE_FIXTURE = REPO_ROOT / "examples" / "refund-causal-diagnosis" / "candidate_traces"


# ---- library tests --------------------------------------------------------


def test_compute_baseline_hash_same_dir_twice_same_digest(tmp_path: Path) -> None:
    """Pure function — same input bytes, same digest. Anything else
    would mean the hash leaks file ordering or filesystem metadata."""
    target = tmp_path / "traces"
    shutil.copytree(BASELINE_FIXTURE, target)
    h1 = compute_baseline_hash(target)
    h2 = compute_baseline_hash(target)
    assert h1.stamp == h2.stamp
    assert h1.n_files == h2.n_files
    assert h1.n_records == h2.n_records


def test_compute_baseline_hash_invariant_across_filename(tmp_path: Path) -> None:
    """Two baselines with the same records (different filenames)
    must hash to the same digest. Otherwise renaming a fixture would
    look like a behaviour change to reviewers."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    src_files = sorted(BASELINE_FIXTURE.glob("*.agentlog"))
    for i, src in enumerate(src_files):
        # Same bytes, different names.
        (a / f"trace-{i}.agentlog").write_bytes(src.read_bytes())
        (b / f"renamed-{i}.agentlog").write_bytes(src.read_bytes())
    assert compute_baseline_hash(a).stamp == compute_baseline_hash(b).stamp


def test_compute_baseline_hash_changes_when_a_record_changes(tmp_path: Path) -> None:
    """Mutate one trace; the digest must flip. The whole point of the
    pin is to surface drift in `git diff` exactly here."""
    target = tmp_path / "traces"
    shutil.copytree(BASELINE_FIXTURE, target)
    before = compute_baseline_hash(target)
    # Append a new record line; parser will reject malformed envelopes,
    # so we re-write the first file with one fewer record.
    first = sorted(target.glob("*.agentlog"))[0]
    lines = first.read_text(encoding="utf-8").splitlines()
    first.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    after = compute_baseline_hash(target)
    assert before.stamp != after.stamp


def test_compute_baseline_hash_empty_dir_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        compute_baseline_hash(empty)


def test_compute_baseline_hash_detects_payload_tamper_with_stale_id(tmp_path: Path) -> None:
    """Audit-chain regression test: editing a record's payload while
    leaving the existing `id` field unchanged MUST fail. Otherwise an
    attacker (or a sloppy editor) can mutate a baseline trace and the
    stored hash stays identical, defeating the content-addressing
    integrity claim.

    Concrete scenario (the same one a reviewer reproduced): copy a
    baseline trace, edit the first record's `payload` in place,
    re-write the same `id`, run `compute_baseline_hash`. Before the
    fix, the digest matched the untampered baseline. After the fix,
    the call raises with the tampered id surfaced.
    """
    import json

    target = tmp_path / "traces"
    shutil.copytree(BASELINE_FIXTURE, target)

    first = sorted(target.glob("*.agentlog"))[0]
    lines = first.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    stale_id = rec["id"]
    # Tamper with payload bytes (a tag value) but keep the stored id.
    tags = rec.get("payload", {}).get("tags") or {}
    tags["scenario"] = "TAMPERED"
    rec["payload"]["tags"] = tags
    lines[0] = json.dumps(rec, separators=(",", ":"), sort_keys=True)
    first.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        compute_baseline_hash(target)
    assert stale_id in str(excinfo.value) or "tamper" in str(excinfo.value).lower()


def test_compute_baseline_hash_dir_with_no_records_raises(tmp_path: Path) -> None:
    """A `.agentlog` file with no records is a degenerate state —
    pinning a hash over zero records would let any future trace
    pass `verify`. Bail loudly instead."""
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "empty.agentlog").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        compute_baseline_hash(bad)


def test_load_save_shadow_yaml_round_trip(tmp_path: Path) -> None:
    """Save then load must produce an equivalent ShadowConfig.
    Empty fields drop out so the on-disk file stays minimal."""
    cfg = ShadowConfig(
        baseline_dir="baseline",
        baseline_hash="sha256:" + "a" * 64,
        backend="recorded",
        policy="policy.yaml",
    )
    p = tmp_path / "shadow.yaml"
    save_shadow_yaml(p, cfg)
    loaded = load_shadow_yaml(p)
    assert loaded.baseline_dir == cfg.baseline_dir
    assert loaded.baseline_hash == cfg.baseline_hash
    assert loaded.policy == cfg.policy


def test_load_shadow_yaml_missing_file_returns_default() -> None:
    """No file = no pin = empty config; `shadow init` writes a
    fresh one. Importing the function on a clean repo must not
    raise."""
    cfg = load_shadow_yaml(Path("/nonexistent/shadow.yaml"))
    assert cfg.baseline_hash is None
    assert cfg.baseline_dir is None


def test_load_shadow_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    """Top-level scalar / list = malformed config. Bail with a
    pointer at the file."""
    p = tmp_path / "shadow.yaml"
    p.write_text("- just_a_list\n")
    with pytest.raises(ValueError):
        load_shadow_yaml(p)


def test_verify_baseline_pinned_detects_drift(tmp_path: Path) -> None:
    """The drift message names both digests so a reviewer can
    decide quickly whether the change is intentional."""
    target = tmp_path / "traces"
    shutil.copytree(BASELINE_FIXTURE, target)
    actual = compute_baseline_hash(target)
    cfg = ShadowConfig(baseline_dir=str(target), baseline_hash="sha256:" + "f" * 64)
    ok, msg = verify_baseline_pinned(cfg, target)
    assert not ok
    assert msg is not None
    assert "drift" in msg.lower()
    assert actual.stamp in msg


def test_verify_baseline_pinned_passes_on_match(tmp_path: Path) -> None:
    target = tmp_path / "traces"
    shutil.copytree(BASELINE_FIXTURE, target)
    actual = compute_baseline_hash(target)
    cfg = ShadowConfig(baseline_dir=str(target), baseline_hash=actual.stamp)
    ok, msg = verify_baseline_pinned(cfg, target)
    assert ok
    assert msg is None


# ---- CLI tests ------------------------------------------------------------


def _run_in(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # Force UTF-8 decoding so Rich's status markers (✓ ✗) survive on
    # Windows runners whose default codepage is cp1252.
    import os as _os

    env = dict(_os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "shadow.cli.app", *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=cwd,
        check=False,
    )


def test_cli_baseline_create_writes_shadow_yaml(tmp_path: Path) -> None:
    target = tmp_path / "traces"
    shutil.copytree(BASELINE_FIXTURE, target)
    result = _run_in(tmp_path, "baseline", "create", "traces")
    assert result.returncode == 0, result.stderr
    cfg = load_shadow_yaml(tmp_path / "shadow.yaml")
    assert cfg.baseline_hash is not None
    assert cfg.baseline_hash.startswith("sha256:")
    assert cfg.baseline_dir == "traces"


def test_cli_baseline_create_refuses_to_overwrite_existing_pin(tmp_path: Path) -> None:
    target = tmp_path / "traces"
    shutil.copytree(BASELINE_FIXTURE, target)
    assert _run_in(tmp_path, "baseline", "create", "traces").returncode == 0
    # Second create on the same dir must error out.
    second = _run_in(tmp_path, "baseline", "create", "traces")
    assert second.returncode != 0
    assert "already pinned" in (second.stdout + second.stderr).lower()


def test_cli_baseline_update_requires_force(tmp_path: Path) -> None:
    """The friction flag is intentional — without --force, a typo
    can't approve a regression silently."""
    target = tmp_path / "traces"
    shutil.copytree(BASELINE_FIXTURE, target)
    _run_in(tmp_path, "baseline", "create", "traces")
    no_force = _run_in(tmp_path, "baseline", "update", "traces")
    assert no_force.returncode != 0
    assert "force" in (no_force.stdout + no_force.stderr).lower()


def test_cli_baseline_approve_promotes_candidate(tmp_path: Path) -> None:
    """`approve` copies candidate `.agentlog` files into the baseline
    dir and re-pins the hash. The end state is "this trace set is
    now the gold standard for future gate-pr runs."""
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    shutil.copytree(BASELINE_FIXTURE, base)
    shutil.copytree(CANDIDATE_FIXTURE, cand)
    _run_in(tmp_path, "baseline", "create", "baseline")
    pre_hash = load_shadow_yaml(tmp_path / "shadow.yaml").baseline_hash

    result = _run_in(tmp_path, "baseline", "approve", "candidate", "--force")
    assert result.returncode == 0, result.stderr
    post_hash = load_shadow_yaml(tmp_path / "shadow.yaml").baseline_hash
    assert post_hash != pre_hash, "approve must re-pin to a new hash"

    # Files were copied.
    assert any(base.rglob("s1.agentlog"))


def test_cli_baseline_verify_exits_zero_on_match_one_on_drift(tmp_path: Path) -> None:
    target = tmp_path / "traces"
    shutil.copytree(BASELINE_FIXTURE, target)
    _run_in(tmp_path, "baseline", "create", "traces")
    assert _run_in(tmp_path, "baseline", "verify").returncode == 0

    # Mutate a record so the hash flips.
    first = sorted(target.glob("*.agentlog"))[0]
    lines = first.read_text().splitlines()
    first.write_text("\n".join(lines[:-1]) + "\n")
    drifted = _run_in(tmp_path, "baseline", "verify")
    assert drifted.returncode == 1
    assert "drift" in (drifted.stdout + drifted.stderr).lower()
