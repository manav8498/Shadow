"""Frozen-baseline workflow — `shadow baseline create / update / approve`.

Mirrors snapshot-testing conventions (Insta in Rust, Jest's
`--updateSnapshot`) so engineers don't have to learn a new mental
model. A baseline is a frozen `.agentlog` set committed to the
repo, and updates are explicit human-approved actions whose result
is a content hash recorded in `shadow.yaml`.

Three commands:

* `shadow baseline create <baseline-dir>` — first-time creation.
  Walks `.agentlog` files in the directory, computes a single
  aggregate content-id over the canonical bytes, writes the hash
  into `shadow.yaml` under `baseline.hash`. Refuses to overwrite
  an existing pin (use `update` for that).

* `shadow baseline update <baseline-dir>` — re-pin after a
  deliberate baseline regeneration. The user has presumably
  re-recorded traces because behaviour intentionally changed.
  Requires `--force` so a fat-fingered run can't silently approve
  a regression.

* `shadow baseline approve <candidate-dir>` — promote a candidate
  trace set to baseline. Copies `.agentlog` files into the
  configured baseline directory and re-pins the hash. Used
  immediately after `shadow gate-pr` ships a verdict the user
  agrees with.

The hash lives in `shadow.yaml` so PRs that change the baseline
show up in `git diff` as a single line:

    baseline:
      dir: .shadow/baseline
      hash: sha256:abc123...

Reviewers see "the baseline hash changed in this PR" — a deliberate
signal that the agent's expected behaviour was reset.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from shadow import _core

__all__ = [
    "BaselineHash",
    "ShadowConfig",
    "compute_baseline_hash",
    "load_shadow_yaml",
    "save_shadow_yaml",
    "verify_baseline_pinned",
]

SHADOW_YAML_NAME = "shadow.yaml"
"""Conventional config-file name. Lives at the project root next to
`pyproject.toml` / `package.json` / `Cargo.toml` so it's discoverable
by users who already know shadow exists. Multi-project monorepos can
override the search path via `--config <path>` on the CLI."""


@dataclass(frozen=True)
class BaselineHash:
    """Stable content fingerprint of a baseline trace set.

    Computed as `sha256(sorted(content_id(record) for record in
    every .agentlog under the dir))` so two baselines with the same
    records (in any file order) hash the same — only behavioural
    changes flip the hash.
    """

    algo: str  # always "sha256" for v1
    digest: str  # hex digest, no prefix
    n_files: int
    n_records: int

    @property
    def stamp(self) -> str:
        """The string written to and read from `shadow.yaml`. Format:
        `sha256:<hex>` — symmetric with the per-record content ids
        Shadow already uses everywhere else."""
        return f"{self.algo}:{self.digest}"


@dataclass
class ShadowConfig:
    """Parsed `shadow.yaml`. Fields are intentionally flat — five
    top-level keys is the cap. Anything more belongs in a separate
    config layer or per-command flag."""

    baseline_dir: str | None = None
    baseline_hash: str | None = None
    backend: str = "recorded"  # default --backend for diagnose-pr / gate-pr
    policy: str | None = None  # default --policy path
    extra_secrets_patterns: str | None = None
    """Path to a custom `--patterns` file passed to `shadow scan`."""

    @classmethod
    def from_dict(cls, data: dict) -> ShadowConfig:  # type: ignore[type-arg]
        """Tolerant parse of a YAML dict. Missing keys default; unknown
        keys are silently ignored so v1 configs survive a v2 upgrade."""
        baseline = data.get("baseline") or {}
        if not isinstance(baseline, dict):
            baseline = {}
        return cls(
            baseline_dir=str(baseline.get("dir")) if baseline.get("dir") else None,
            baseline_hash=str(baseline.get("hash")) if baseline.get("hash") else None,
            backend=str(data.get("backend", "recorded")),
            policy=str(data.get("policy")) if data.get("policy") else None,
            extra_secrets_patterns=(
                str(data.get("scan_patterns")) if data.get("scan_patterns") else None
            ),
        )

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        """Round-trip back to YAML-friendly dict. Empty / None fields
        are omitted so the on-disk file stays minimal."""
        out: dict = {}  # type: ignore[type-arg]
        baseline: dict = {}  # type: ignore[type-arg]
        if self.baseline_dir is not None:
            baseline["dir"] = self.baseline_dir
        if self.baseline_hash is not None:
            baseline["hash"] = self.baseline_hash
        if baseline:
            out["baseline"] = baseline
        if self.backend != "recorded":
            out["backend"] = self.backend
        if self.policy is not None:
            out["policy"] = self.policy
        if self.extra_secrets_patterns is not None:
            out["scan_patterns"] = self.extra_secrets_patterns
        return out


def _iter_agentlogs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.agentlog"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"path does not exist: {path}")


def compute_baseline_hash(baseline_dir: Path) -> BaselineHash:
    """Walk the directory, recompute every record's content id from its
    canonical payload bytes, verify the stored id matches, and return a
    `BaselineHash` covering the whole set.

    Two baselines with the same records (regardless of which file
    they're in or what the file names are) produce the same digest.
    A reviewer seeing the digest change in `git diff` knows behaviour
    was intentionally regenerated; a reviewer seeing the digest
    unchanged knows the baseline is the same set Shadow gated against
    last time.

    Audit-chain integrity: the stored `id` on every record is verified
    against `sha256(canonical_bytes(payload))`. A mismatch means the
    payload was tampered with while the stored id was left stale, and
    we raise `ValueError` immediately. The baseline hash itself is
    computed over the recomputed ids so a tampered file cannot ride
    through even if the verify step were ever skipped.
    """
    files = _iter_agentlogs(baseline_dir)
    if not files:
        raise FileNotFoundError(
            f"no .agentlog files under {baseline_dir!r} — did you record any traces?"
        )
    record_ids: list[str] = []
    n_records = 0
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
            stored = rec.get("id")
            payload = rec.get("payload")
            if payload is None:
                raise ValueError(
                    f"{f}: record is missing `payload` field; "
                    "cannot verify content-address integrity"
                )
            expected = _core.content_id(payload)
            if not isinstance(stored, str) or not stored:
                raise ValueError(f"{f}: record has empty or non-string `id`; expected {expected}")
            if stored != expected:
                raise ValueError(
                    f"{f}: baseline tamper detected. "
                    f"Stored id {stored} does not match the content id of its "
                    f"payload (expected {expected}). The record has been edited "
                    "without re-deriving its content address. Re-record the "
                    "trace or restore the original file."
                )
            record_ids.append(expected)
            n_records += 1
    if n_records == 0:
        raise ValueError(f"{baseline_dir} has .agentlog files but no records inside")

    record_ids.sort()
    h = hashlib.sha256()
    for rid in record_ids:
        h.update(rid.encode("utf-8"))
        h.update(b"\n")
    return BaselineHash(
        algo="sha256",
        digest=h.hexdigest(),
        n_files=len(files),
        n_records=n_records,
    )


def load_shadow_yaml(path: Path) -> ShadowConfig:
    """Load `shadow.yaml` from disk. Returns a default-empty config
    when the file is absent — that's the legitimate state for a
    fresh `shadow init` run before any baseline has been pinned."""
    if not path.is_file():
        return ShadowConfig()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}"
        )
    return ShadowConfig.from_dict(data)


def save_shadow_yaml(path: Path, cfg: ShadowConfig) -> None:
    """Write the config back. Uses `default_flow_style=False` so the
    on-disk shape is the conventional block style users expect."""
    serialised = yaml.safe_dump(
        cfg.to_dict(),
        default_flow_style=False,
        sort_keys=True,
    )
    if not serialised.strip():
        # Don't leave behind an empty file.
        if path.exists():
            path.unlink()
        return
    path.write_text(
        "# Shadow project config. See `shadow init --help` and the docs.\n" + serialised,
        encoding="utf-8",
    )


def verify_baseline_pinned(cfg: ShadowConfig, baseline_dir: Path) -> tuple[bool, str | None]:
    """Compute the actual baseline hash and compare against the
    pinned value in `cfg`. Returns `(ok, message)`. `ok` is True
    when the pin matches; False (with message) when there's drift
    or no pin at all."""
    if cfg.baseline_hash is None:
        return False, "no baseline pinned — run `shadow baseline create` first."
    actual = compute_baseline_hash(baseline_dir)
    if actual.stamp != cfg.baseline_hash:
        return False, (
            f"baseline drift detected.\n"
            f"  pinned:  {cfg.baseline_hash}\n"
            f"  actual:  {actual.stamp}\n"
            f"  fix:     re-record + run `shadow baseline update --force` "
            f"if the change is intentional."
        )
    return True, None
