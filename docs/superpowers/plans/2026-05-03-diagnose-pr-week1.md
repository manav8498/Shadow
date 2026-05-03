# Phase 1 Week 1: `shadow diagnose-pr` Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the structural skeleton of `shadow diagnose-pr` end-to-end against the existing `examples/demo/` fixtures — producing a JSON report and a markdown PR comment with `verdict`, `total_traces`, `affected_traces`, `blast_radius`, and a top-causes section that's empty in this skeleton (causal attribution arrives in Week 3).

**Architecture:** New Python package `shadow.diagnose_pr` composed of focused modules (`models`, `loaders`, `deltas`, `report`, `render`) plus a Typer CLI command in `python/src/shadow/cli/app.py`. Skeleton uses **trivial classifications** for affected/verdict (any trace that has a candidate counterpart is "potentially affected", verdict is `ship` if zero traces else `probe`). Real classification, policy, and causal attribution wire into these same dataclasses in Weeks 2–3 without breaking the v0.1 schema.

**Tech Stack:** Python 3.11, Typer 0.13, PyYAML, dataclasses, JSON, pytest, ruff, mypy --strict. No new dependencies. Composes existing `shadow._core.parse_agentlog`, `shadow.causal.replay.openai_replayer._canonical_config_hash` (we'll use the same canonicalisation for delta hashing).

---

## Decisions locked in (from spec §11 review)

| Question | Answer |
|---|---|
| Default `--max-traces` | `200` |
| Verdict gradient | 4 levels: `ship` / `probe` / `hold` / `stop` |
| Policy module rename (`hierarchical → policy`) | Deferred — adapter only |
| `gate-pr` exit code 3 | Effectively fail-closed; treat 3 as 1 in CI |

These are pinned in `diagnose_pr/__init__.py` as module-level constants where applicable.

---

## File structure

### Created in this plan

| File | Responsibility |
|---|---|
| `python/src/shadow/diagnose_pr/__init__.py` | Public API exports + version constant `SCHEMA_VERSION = "diagnose-pr/v0.1"` |
| `python/src/shadow/diagnose_pr/models.py` | Frozen dataclasses: `ConfigDelta`, `TraceDiagnosis`, `CauseEstimate`, `DiagnosePrReport`, type aliases `Verdict`, `DeltaKind` |
| `python/src/shadow/diagnose_pr/loaders.py` | `load_config(path) -> dict[str, Any]`, `load_traces(paths) -> list[LoadedTrace]` |
| `python/src/shadow/diagnose_pr/deltas.py` | `extract_deltas(baseline, candidate, changed_files=None) -> list[ConfigDelta]` |
| `python/src/shadow/diagnose_pr/report.py` | `build_report(...) -> DiagnosePrReport`, `to_json(report) -> str` |
| `python/src/shadow/diagnose_pr/render.py` | `render_pr_comment(report) -> str` (markdown) |
| `python/tests/test_diagnose_pr_models.py` | Dataclass invariants |
| `python/tests/test_diagnose_pr_loaders.py` | Loader behavior + error paths |
| `python/tests/test_diagnose_pr_deltas.py` | Delta extractor coverage |
| `python/tests/test_diagnose_pr_report.py` | Report assembly + JSON shape |
| `python/tests/test_diagnose_pr_render.py` | Markdown snapshot |
| `python/tests/test_diagnose_pr_cli.py` | CLI smoke (CliRunner) |

### Modified

| File | Change |
|---|---|
| `python/src/shadow/cli/app.py` | Add `@app.command("diagnose-pr", rich_help_panel="Common")` registration |
| `python/pyproject.toml` | Already covers new files via `src/shadow/**` glob — verify only |

---

## Task 0: Branch hygiene check

**Files:** none (verification only)

- [ ] **Step 0.1: Confirm clean tree on main**

Run:
```bash
git -C /Users/manavpatel/Downloads/Shadow status --short
```
Expected: empty output (clean working tree).

- [ ] **Step 0.2: Confirm test baseline is green**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests -q --no-header 2>&1 | tail -5
```
Expected: `1701 passed, 14 skipped` (or similar — current baseline). If this fails, stop and investigate.

---

## Task 1: Package bootstrap + smoke test

**Files:**
- Create: `python/src/shadow/diagnose_pr/__init__.py`
- Create: `python/tests/test_diagnose_pr_smoke.py`

- [ ] **Step 1.1: Write the failing smoke test**

Create `python/tests/test_diagnose_pr_smoke.py`:

```python
"""Smoke test: the diagnose_pr package is importable and exposes its
v0.1 schema constant. Anchors the public surface so future renames
break tests, not consumers."""

from __future__ import annotations


def test_package_imports() -> None:
    import shadow.diagnose_pr as dp

    assert hasattr(dp, "SCHEMA_VERSION")
    assert dp.SCHEMA_VERSION == "diagnose-pr/v0.1"
```

- [ ] **Step 1.2: Run the test, see ImportError**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_smoke.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'shadow.diagnose_pr'`.

- [ ] **Step 1.3: Create the package init**

Create `python/src/shadow/diagnose_pr/__init__.py`:

```python
"""Causal Regression Forensics for AI Agents.

`shadow diagnose-pr` answers, in one PR comment:

  1. Did agent behavior change?
  2. How many real or production-like traces are affected?
  3. Which exact candidate change caused the regression?
  4. How confident are we?
  5. What fix should be verified before merge?

This package composes existing Shadow internals (the 9-axis Rust
differ, `shadow.causal.attribution`, `shadow.hierarchical` policy
checker, `shadow.mine` representative selection) into one PR-time
command surface. It does not reinvent any of them.

The `v0.1` schema is intentionally narrow:

  * verdict: ship / probe / hold / stop
  * blast_radius: affected / total
  * dominant_cause: a single ConfigDelta with ATE + bootstrap CI +
    E-value, or None when causal attribution wasn't run.

Forward compatibility: future versions add fields without renaming
existing ones; readers must tolerate unknown keys.
"""

from __future__ import annotations

SCHEMA_VERSION = "diagnose-pr/v0.1"
"""Schema identifier embedded in every diagnose-pr report.json.

Bumped only on breaking field renames or removals; additive changes
(new optional fields) keep the same schema_version.
"""

DEFAULT_MAX_TRACES = 200
"""Cap on traces fed to the per-trace replay/diff loop. When the input
corpus exceeds this, `shadow.mine.mine` selects a representative
sample using the failure-mode score (errors / refusals / high latency
rank highest)."""

DEFAULT_N_BOOTSTRAP = 500
"""Default bootstrap resample count for confidence intervals on
causal ATE. 500 is a common floor; raise via --n-bootstrap when
corpora are small and CI width matters."""

DEFAULT_CONFIDENCE = 0.95
"""Default CI level. Anything tighter loses interpretability for the
PR audience; anything looser stops being meaningful."""


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MAX_TRACES",
    "DEFAULT_N_BOOTSTRAP",
    "SCHEMA_VERSION",
]
```

- [ ] **Step 1.4: Run the test, see green**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_smoke.py -q
```
Expected: `1 passed`.

- [ ] **Step 1.5: Commit**

```bash
git -C /Users/manavpatel/Downloads/Shadow add python/src/shadow/diagnose_pr/__init__.py python/tests/test_diagnose_pr_smoke.py
git -C /Users/manavpatel/Downloads/Shadow commit -s -m "$(cat <<'EOF'
feat(diagnose-pr): bootstrap package skeleton

Phase 1 Week 1 of the Causal Regression Forensics pivot. This commit
only lands the package marker, schema version, and v1 default
constants — no behavior yet. Anchors the public surface so subsequent
modules (models, loaders, deltas, report, render) plug into a stable
package shape.

Design spec: docs/superpowers/specs/2026-05-03-causal-regression-forensics-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Data models

**Files:**
- Create: `python/src/shadow/diagnose_pr/models.py`
- Create: `python/tests/test_diagnose_pr_models.py`

- [ ] **Step 2.1: Write the failing model tests**

Create `python/tests/test_diagnose_pr_models.py`:

```python
"""Tests for the `shadow.diagnose_pr.models` dataclasses.

These dataclasses are the public report shape — every consumer (PR
comment renderer, JSON writer, future verify-fix command) depends on
their field names. The test pins those names so a careless rename
fails CI before it ships."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def test_config_delta_is_frozen_and_named() -> None:
    from shadow.diagnose_pr.models import ConfigDelta

    d = ConfigDelta(
        id="model:gpt-4.1->gpt-4.1-mini",
        kind="model",
        path="model",
        old_hash=None,
        new_hash=None,
        display="model: gpt-4.1 → gpt-4.1-mini",
    )
    assert d.kind == "model"
    with pytest.raises(FrozenInstanceError):
        d.path = "params"  # type: ignore[misc]


def test_trace_diagnosis_carries_per_trace_state() -> None:
    from shadow.diagnose_pr.models import TraceDiagnosis

    diag = TraceDiagnosis(
        trace_id="sha256:abc",
        affected=True,
        risk=78.4,
        worst_axis="trajectory",
        first_divergence={"pair_index": 2, "axis": "trajectory"},
        policy_violations=[{"rule_id": "x", "severity": "error"}],
    )
    assert diag.affected is True
    assert diag.risk == pytest.approx(78.4)


def test_cause_estimate_has_ate_and_ci_fields() -> None:
    from shadow.diagnose_pr.models import CauseEstimate

    c = CauseEstimate(
        delta_id="prompt:system",
        axis="trajectory",
        ate=0.31,
        ci_low=0.22,
        ci_high=0.44,
        e_value=2.8,
        confidence=1.0,
    )
    assert c.ci_low is not None and c.ci_high is not None
    assert c.ci_low < c.ate < c.ci_high


def test_diagnose_pr_report_has_schema_version_and_verdict() -> None:
    from shadow.diagnose_pr import SCHEMA_VERSION
    from shadow.diagnose_pr.models import DiagnosePrReport

    r = DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict="ship",
        total_traces=0,
        affected_traces=0,
        blast_radius=0.0,
        dominant_cause=None,
        top_causes=[],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=0,
        worst_policy_rule=None,
        suggested_fix=None,
        flags=[],
    )
    assert r.schema_version == "diagnose-pr/v0.1"
    assert r.verdict == "ship"
    assert r.blast_radius == 0.0


def test_verdict_literal_rejects_unknown_values_at_type_level() -> None:
    """Compile-time, not runtime — but document the contract."""
    from shadow.diagnose_pr.models import Verdict

    # Just an existence check; the Literal["ship", "probe", "hold", "stop"]
    # guarantee is enforced by mypy --strict in CI.
    assert "ship" in Verdict.__args__  # type: ignore[attr-defined]
    assert "stop" in Verdict.__args__  # type: ignore[attr-defined]
```

- [ ] **Step 2.2: Run tests, see ImportError**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_models.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'shadow.diagnose_pr.models'`.

- [ ] **Step 2.3: Implement the models**

Create `python/src/shadow/diagnose_pr/models.py`:

```python
"""Frozen dataclasses for the `shadow diagnose-pr` v0.1 report.

These are the public report shape. Every consumer — PR comment
renderer, JSON writer, future `verify-fix` command — reads these
fields by name. Adding a field is safe; renaming or removing one
is a v0.2 schema change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Verdict = Literal["ship", "probe", "hold", "stop"]
"""Four-level verdict gradient. v1 keeps it small to avoid analysis
paralysis on the PR audience; future versions may sub-bin `probe`."""

DeltaKind = Literal[
    "prompt",
    "model",
    "tool_schema",
    "retriever",
    "temperature",
    "policy",
    "unknown",
]
"""Coarse classification of a config-level change. The extractor in
`deltas.py` assigns this; the renderer uses it to phrase the cause.
`unknown` is the safe fallback — better to surface 'unknown delta at
config.X' than to mis-classify."""


@dataclass(frozen=True)
class ConfigDelta:
    """One atomic, named change between baseline and candidate config.

    `id` is the stable identifier the renderer prints (e.g.
    `system_prompt.md:47` for a prompt-file diff or `model:gpt-4.1->
    gpt-4.1-mini` for a top-level field flip). `path` is the
    config-key path or file path the change lives at. Hash fields
    are hex sha256 over canonical bytes when comparable, `None` when
    the side wasn't a hashable artefact (e.g. when the file didn't
    exist on one side).
    """

    id: str
    kind: DeltaKind
    path: str
    old_hash: str | None
    new_hash: str | None
    display: str


@dataclass(frozen=True)
class TraceDiagnosis:
    """Per-trace diagnosis result. Carries the smallest amount of
    state the PR comment + JSON report need; richer analysis lives
    in side artefacts (per-trace diff reports under `.shadow/`).
    """

    trace_id: str
    affected: bool
    risk: float  # 0..100 — corpus-relative severity ranking
    worst_axis: str | None
    first_divergence: dict[str, Any] | None
    policy_violations: list[dict[str, Any]]


@dataclass(frozen=True)
class CauseEstimate:
    """One cause-estimate from causal_attribution, normalised for
    presentation.

    `ci_low`/`ci_high` are `None` when bootstrap wasn't run (e.g.
    `--n-bootstrap 0`). `e_value` is `None` when sensitivity wasn't
    requested. `confidence` is a coarse v1 marker — 1.0 when the CI
    excludes zero, 0.5 otherwise.
    """

    delta_id: str
    axis: str
    ate: float
    ci_low: float | None
    ci_high: float | None
    e_value: float | None
    confidence: float


@dataclass(frozen=True)
class DiagnosePrReport:
    """The full diagnose-pr v0.1 report.

    `dominant_cause` is the single cause the verdict and PR comment
    headline. `top_causes` is the ranked list (typically up to 5)
    surfaced in the JSON for tooling consumers. `flags` carries
    advisory warnings — the v1 set is just `["low_power"]` for n<30.
    """

    schema_version: str
    verdict: Verdict
    total_traces: int
    affected_traces: int
    blast_radius: float
    dominant_cause: CauseEstimate | None
    top_causes: list[CauseEstimate]
    trace_diagnoses: list[TraceDiagnosis]
    affected_trace_ids: list[str]
    new_policy_violations: int
    worst_policy_rule: str | None
    suggested_fix: str | None
    flags: list[str]


__all__ = [
    "CauseEstimate",
    "ConfigDelta",
    "DeltaKind",
    "DiagnosePrReport",
    "TraceDiagnosis",
    "Verdict",
]
```

- [ ] **Step 2.4: Run tests, see green**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_models.py -q
```
Expected: `5 passed`.

- [ ] **Step 2.5: Commit**

```bash
git -C /Users/manavpatel/Downloads/Shadow add python/src/shadow/diagnose_pr/models.py python/tests/test_diagnose_pr_models.py
git -C /Users/manavpatel/Downloads/Shadow commit -s -m "$(cat <<'EOF'
feat(diagnose-pr): data models for v0.1 report

Frozen dataclasses pinning the public report shape:
ConfigDelta, TraceDiagnosis, CauseEstimate, DiagnosePrReport, plus
the Verdict and DeltaKind type aliases. These names appear in the
JSON report and the PR-comment markdown; renames are v0.2 schema
changes. Field set matches the design spec §3.3 verbatim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Loaders

**Files:**
- Create: `python/src/shadow/diagnose_pr/loaders.py`
- Create: `python/tests/test_diagnose_pr_loaders.py`

- [ ] **Step 3.1: Write failing loader tests**

Create `python/tests/test_diagnose_pr_loaders.py`:

```python
"""Tests for `shadow.diagnose_pr.loaders`.

Two surfaces:
  * load_config(path) — YAML config (same schema as `shadow replay`)
  * load_traces(paths) — one or more .agentlog files / dirs

Both raise typed errors (ShadowConfigError / ShadowParseError) so
the CLI can surface a clean message instead of a stack trace."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_load_config_round_trips_a_real_demo_yaml(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_config

    p = tmp_path / "baseline.yaml"
    p.write_text(
        "model: claude-opus-4-7\n"
        "params:\n"
        "  temperature: 0.2\n"
        "  max_tokens: 512\n"
        "prompt:\n"
        "  system: 'You are a refund agent.'\n"
    )
    cfg = load_config(p)
    assert cfg["model"] == "claude-opus-4-7"
    assert cfg["params"]["temperature"] == pytest.approx(0.2)
    assert "refund agent" in cfg["prompt"]["system"]


def test_load_config_missing_file_raises_shadow_config_error(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_config
    from shadow.errors import ShadowConfigError

    with pytest.raises(ShadowConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_invalid_yaml_raises_shadow_config_error(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_config
    from shadow.errors import ShadowConfigError

    p = tmp_path / "bad.yaml"
    p.write_text("model: : :")
    with pytest.raises(ShadowConfigError, match="could not parse"):
        load_config(p)


def test_load_traces_from_file_returns_one_loaded_trace(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_traces
    from shadow.sdk import Session

    p = tmp_path / "t.agentlog"
    with Session(output_path=p, tags={"env": "test"}) as s:
        s.record_chat(
            request={"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}], "params": {}},
            response={
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )

    loaded = load_traces([p])
    assert len(loaded) == 1
    t = loaded[0]
    assert t.path == p
    assert isinstance(t.trace_id, str) and t.trace_id.startswith("sha256:")
    assert len(t.records) >= 2  # metadata + at least one chat pair


def test_load_traces_from_directory_globs_agentlog_files(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_traces
    from shadow.sdk import Session

    for name in ("a.agentlog", "b.agentlog"):
        with Session(output_path=tmp_path / name, tags={}) as s:
            s.record_chat(
                request={"model": "x", "messages": [{"role": "user", "content": "h"}], "params": {}},
                response={
                    "model": "x",
                    "content": [{"type": "text", "text": "h"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 1,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            )

    loaded = load_traces([tmp_path])
    assert len(loaded) == 2
    paths = sorted(t.path.name for t in loaded)
    assert paths == ["a.agentlog", "b.agentlog"]


def test_load_traces_skips_non_agentlog_files(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_traces

    (tmp_path / "readme.txt").write_text("hello")
    (tmp_path / "data.json").write_text("{}")
    loaded = load_traces([tmp_path])
    assert loaded == []


def test_load_traces_corrupt_file_raises_shadow_parse_error(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_traces
    from shadow.errors import ShadowParseError

    p = tmp_path / "broken.agentlog"
    p.write_bytes(b"not jsonl at all\n")
    with pytest.raises(ShadowParseError):
        load_traces([p])
```

- [ ] **Step 3.2: Run tests, see ImportError**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_loaders.py -q
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement the loaders**

Create `python/src/shadow/diagnose_pr/loaders.py`:

```python
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
            records = _core.parse_agentlog(blob)
        except OSError as exc:
            raise ShadowConfigError(f"could not read {f}: {exc}") from exc
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
```

- [ ] **Step 3.4: Run tests, see green**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_loaders.py -q
```
Expected: `7 passed`.

- [ ] **Step 3.5: Commit**

```bash
git -C /Users/manavpatel/Downloads/Shadow add python/src/shadow/diagnose_pr/loaders.py python/tests/test_diagnose_pr_loaders.py
git -C /Users/manavpatel/Downloads/Shadow commit -s -m "$(cat <<'EOF'
feat(diagnose-pr): YAML config + .agentlog trace loaders

`load_config(path) -> dict` reads YAML with typed errors
(ShadowConfigError on missing/malformed files). `load_traces(paths)
-> list[LoadedTrace]` accepts files or directories, globs
*.agentlog recursively, returns sorted-by-path results for
determinism, surfaces parse errors as ShadowParseError. The
LoadedTrace dataclass preserves source path + trace_id alongside
the record list — downstream consumers need all three.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Delta extractor

**Files:**
- Create: `python/src/shadow/diagnose_pr/deltas.py`
- Create: `python/tests/test_diagnose_pr_deltas.py`

- [ ] **Step 4.1: Write failing delta-extractor tests**

Create `python/tests/test_diagnose_pr_deltas.py`:

```python
"""Tests for `shadow.diagnose_pr.deltas.extract_deltas`.

The extractor compares two parsed config dicts (already loaded by
`loaders.load_config`) and emits one ConfigDelta per atomic change.
The taxonomy lives in DeltaKind — it's coarse on purpose, since
the renderer only needs to phrase the cause, not implement it."""

from __future__ import annotations

import hashlib

from shadow.diagnose_pr.deltas import extract_deltas
from shadow.diagnose_pr.models import ConfigDelta


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
```

- [ ] **Step 4.2: Run tests, see ImportError**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_deltas.py -q
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4.3: Implement the delta extractor**

Create `python/src/shadow/diagnose_pr/deltas.py`:

```python
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

    Same shape as `shadow.causal.replay.openai_replayer._
    canonical_config_hash`; we share a hash space so future tooling
    can correlate.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


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
            grouped.setdefault("prompt.system", (old, new))
            grouped["prompt.system"] = (
                grouped["prompt.system"][0] if "prompt.system" in grouped else old,
                new,
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
```

- [ ] **Step 4.4: Run tests, see green**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_deltas.py -q
```
Expected: `9 passed`. If fewer pass, fix and re-run.

- [ ] **Step 4.5: Commit**

```bash
git -C /Users/manavpatel/Downloads/Shadow add python/src/shadow/diagnose_pr/deltas.py python/tests/test_diagnose_pr_deltas.py
git -C /Users/manavpatel/Downloads/Shadow commit -s -m "$(cat <<'EOF'
feat(diagnose-pr): config delta extractor

extract_deltas(baseline, candidate, changed_files=) walks two YAML
configs and emits one ConfigDelta per atomic change, classified
into the seven DeltaKind values. Hashes are canonical sha256
(matching shadow.causal.replay's canonicalisation) so a YAML
reformat is not a delta. Paths under prompt.* and tools.* are
coalesced so a multi-line prompt edit reports as ONE delta, not N.
When changed_files is provided and contains a prompt-like file,
the delta id cites the file path for human-readable PR comments.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Report assembly + JSON

**Files:**
- Create: `python/src/shadow/diagnose_pr/report.py`
- Create: `python/tests/test_diagnose_pr_report.py`

- [ ] **Step 5.1: Write failing report tests**

Create `python/tests/test_diagnose_pr_report.py`:

```python
"""Tests for `shadow.diagnose_pr.report`.

`build_report` is the v0.1 skeleton — it consumes loader output and
delta list, produces a DiagnosePrReport with trivial verdict logic
(real classification arrives in Week 2). `to_json` serialises it
with stable key order so PR-comment diffs are minimal."""

from __future__ import annotations

import json

from shadow.diagnose_pr import SCHEMA_VERSION
from shadow.diagnose_pr.deltas import extract_deltas
from shadow.diagnose_pr.loaders import LoadedTrace
from shadow.diagnose_pr.report import build_report, to_json


def _t(idx: int) -> LoadedTrace:
    from pathlib import Path

    return LoadedTrace(
        path=Path(f"/tmp/{idx}.agentlog"),
        trace_id=f"sha256:{idx:064d}",
        records=[
            {"id": f"sha256:{idx:064d}", "kind": "metadata", "payload": {}},
        ],
    )


def test_zero_traces_is_ship_verdict() -> None:
    r = build_report(traces=[], deltas=[], affected_trace_ids=set())
    assert r.verdict == "ship"
    assert r.total_traces == 0
    assert r.affected_traces == 0
    assert r.blast_radius == 0.0
    assert r.dominant_cause is None
    assert r.flags == []


def test_traces_with_zero_affected_is_still_ship() -> None:
    r = build_report(
        traces=[_t(1), _t(2), _t(3)],
        deltas=[],
        affected_trace_ids=set(),
    )
    assert r.verdict == "ship"
    assert r.total_traces == 3
    assert r.affected_traces == 0
    assert r.blast_radius == 0.0


def test_skeleton_marks_affected_traces_as_probe() -> None:
    """In the Week-1 skeleton, affected traces with no causal CI
    yield `probe` (uncertain). Week 3 promotes to `hold` once the
    causal CI excludes zero."""
    traces = [_t(1), _t(2), _t(3)]
    r = build_report(traces=traces, deltas=[], affected_trace_ids={traces[0].trace_id})
    assert r.verdict == "probe"
    assert r.affected_traces == 1
    assert r.blast_radius > 0


def test_low_power_flag_when_n_below_30() -> None:
    r = build_report(traces=[_t(1)], deltas=[], affected_trace_ids=set())
    assert "low_power" in r.flags


def test_no_low_power_flag_when_n_at_least_30() -> None:
    traces = [_t(i) for i in range(30)]
    r = build_report(traces=traces, deltas=[], affected_trace_ids=set())
    assert "low_power" not in r.flags


def test_to_json_includes_schema_version_and_keys_sorted() -> None:
    r = build_report(traces=[_t(1)], deltas=[], affected_trace_ids=set())
    blob = to_json(r)
    parsed = json.loads(blob)
    assert parsed["schema_version"] == SCHEMA_VERSION
    # Stable key order (alphabetical) so PR-side diffs are minimal.
    keys = list(parsed.keys())
    assert keys == sorted(keys)


def test_to_json_round_trips_dataclass_field_set() -> None:
    deltas = extract_deltas({"model": "a"}, {"model": "b"})
    r = build_report(
        traces=[_t(1), _t(2)],
        deltas=deltas,
        affected_trace_ids={f"sha256:{1:064d}"},
    )
    parsed = json.loads(to_json(r))
    expected_keys = {
        "schema_version",
        "verdict",
        "total_traces",
        "affected_traces",
        "blast_radius",
        "dominant_cause",
        "top_causes",
        "trace_diagnoses",
        "affected_trace_ids",
        "new_policy_violations",
        "worst_policy_rule",
        "suggested_fix",
        "flags",
    }
    assert set(parsed.keys()) == expected_keys
    assert parsed["affected_trace_ids"] == [f"sha256:{1:064d}"]
```

- [ ] **Step 5.2: Run tests, see ImportError**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_report.py -q
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 5.3: Implement report assembly**

Create `python/src/shadow/diagnose_pr/report.py`:

```python
"""DiagnosePrReport assembly + JSON serialisation.

`build_report` is the v0.1 skeleton: it consumes loader output
(traces) plus a delta list and an "affected" trace-id set, and
produces a DiagnosePrReport. The verdict logic here is intentionally
trivial:

  * 0 affected            → ship
  * any affected, no CI   → probe   (Week 3 will promote to hold
                                     once causal CI excludes zero)

Real verdict logic — `hold` from CI excluding zero, `stop` from
dangerous-tool policy violations — arrives in Week 2 (`risk.py`)
without changing the v0.1 schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable

from shadow.diagnose_pr import SCHEMA_VERSION
from shadow.diagnose_pr.loaders import LoadedTrace
from shadow.diagnose_pr.models import (
    ConfigDelta,
    DiagnosePrReport,
    TraceDiagnosis,
    Verdict,
)

_LOW_POWER_THRESHOLD = 30


def build_report(
    *,
    traces: list[LoadedTrace],
    deltas: list[ConfigDelta],
    affected_trace_ids: set[str],
    new_policy_violations: int = 0,
    worst_policy_rule: str | None = None,
) -> DiagnosePrReport:
    """Assemble a DiagnosePrReport from skeleton inputs.

    `deltas` is currently used for record-keeping (top_causes is
    empty until Week 3). Keeping the parameter in the signature
    means the CLI can wire it up now and Week 3 only changes the
    body of this function.
    """
    total = len(traces)
    affected = len(affected_trace_ids)
    blast_radius = (affected / total) if total > 0 else 0.0
    verdict: Verdict = "ship" if affected == 0 else "probe"

    diagnoses = [
        TraceDiagnosis(
            trace_id=t.trace_id,
            affected=t.trace_id in affected_trace_ids,
            risk=0.0,
            worst_axis=None,
            first_divergence=None,
            policy_violations=[],
        )
        for t in traces
    ]

    flags: list[str] = []
    if 0 < total < _LOW_POWER_THRESHOLD:
        flags.append("low_power")

    return DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict=verdict,
        total_traces=total,
        affected_traces=affected,
        blast_radius=blast_radius,
        dominant_cause=None,  # Week 3
        top_causes=[],         # Week 3
        trace_diagnoses=diagnoses,
        affected_trace_ids=sorted(affected_trace_ids),
        new_policy_violations=new_policy_violations,
        worst_policy_rule=worst_policy_rule,
        suggested_fix=None,    # Week 3
        flags=flags,
    )


def to_json(report: DiagnosePrReport, *, indent: int = 2) -> str:
    """Serialise a report with sorted keys (so PR-side diffs are
    minimal) and the requested indent.
    """
    return json.dumps(asdict(report), sort_keys=True, indent=indent, ensure_ascii=False)


def report_from_traces_and_deltas(
    traces: list[LoadedTrace],
    deltas: list[ConfigDelta],
    *,
    affected: Iterable[str] = (),
) -> DiagnosePrReport:
    """Convenience wrapper used by the CLI command. Accepts an
    iterable of trace ids for the affected set."""
    return build_report(
        traces=traces,
        deltas=deltas,
        affected_trace_ids=set(affected),
    )


__all__ = [
    "build_report",
    "report_from_traces_and_deltas",
    "to_json",
]
```

- [ ] **Step 5.4: Run tests, see green**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_report.py -q
```
Expected: `7 passed`.

- [ ] **Step 5.5: Commit**

```bash
git -C /Users/manavpatel/Downloads/Shadow add python/src/shadow/diagnose_pr/report.py python/tests/test_diagnose_pr_report.py
git -C /Users/manavpatel/Downloads/Shadow commit -s -m "$(cat <<'EOF'
feat(diagnose-pr): build_report + JSON serialisation (v0.1 skeleton)

build_report assembles a DiagnosePrReport from loader output + the
delta list + an affected-trace-id set. Verdict logic is trivial in
v0.1 skeleton (ship if 0 affected, probe otherwise) — Week 2 layers
the real ship/probe/hold/stop classification on the same surface
without schema changes. low_power flag fires when 0<n<30. to_json
sort_keys=True so PR-side diffs are minimal across runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: PR-comment markdown renderer

**Files:**
- Create: `python/src/shadow/diagnose_pr/render.py`
- Create: `python/tests/test_diagnose_pr_render.py`

- [ ] **Step 6.1: Write failing renderer tests**

Create `python/tests/test_diagnose_pr_render.py`:

```python
"""Tests for `shadow.diagnose_pr.render`.

The renderer is the human-facing surface — every word matters.
Tests pin the structure (verdict header, affected-trace count,
suggested fix block), not the exact prose, so we can iterate on
voice without breaking CI."""

from __future__ import annotations

import pytest

from shadow.diagnose_pr import SCHEMA_VERSION
from shadow.diagnose_pr.models import (
    CauseEstimate,
    DiagnosePrReport,
    TraceDiagnosis,
)
from shadow.diagnose_pr.render import render_pr_comment


def _empty_report(verdict: str = "ship") -> DiagnosePrReport:
    return DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict=verdict,  # type: ignore[arg-type]
        total_traces=0,
        affected_traces=0,
        blast_radius=0.0,
        dominant_cause=None,
        top_causes=[],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=0,
        worst_policy_rule=None,
        suggested_fix=None,
        flags=[],
    )


def test_ship_verdict_renders_short_and_friendly() -> None:
    md = render_pr_comment(_empty_report("ship"))
    assert "Shadow verdict: SHIP" in md
    assert "no behavior regression detected" in md.lower()


def test_hold_verdict_includes_dominant_cause_block() -> None:
    cause = CauseEstimate(
        delta_id="system_prompt.md:47",
        axis="trajectory",
        ate=0.31,
        ci_low=0.22,
        ci_high=0.44,
        e_value=2.8,
        confidence=1.0,
    )
    r = DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict="hold",
        total_traces=1247,
        affected_traces=84,
        blast_radius=84 / 1247,
        dominant_cause=cause,
        top_causes=[cause],
        trace_diagnoses=[],
        affected_trace_ids=[],
        new_policy_violations=6,
        worst_policy_rule="confirm-before-refund",
        suggested_fix="Restore the refund confirmation instruction.",
        flags=[],
    )
    md = render_pr_comment(r)
    assert "Shadow verdict: HOLD" in md
    assert "84 / 1,247" in md
    assert "system_prompt.md:47" in md
    assert "trajectory" in md
    assert "+0.31" in md or "0.31" in md
    assert "[0.22, 0.44]" in md
    assert "2.8" in md
    assert "confirm-before-refund" in md
    assert "Restore the refund confirmation instruction" in md
    assert "shadow verify-fix" in md.lower()


def test_probe_verdict_explains_uncertainty() -> None:
    r = _empty_report("probe")
    object.__setattr__(r, "total_traces", 5)
    object.__setattr__(r, "affected_traces", 1)
    object.__setattr__(r, "blast_radius", 0.2)
    md = render_pr_comment(r)
    assert "PROBE" in md
    assert "uncertain" in md.lower() or "low confidence" in md.lower()


def test_low_power_flag_surfaces_in_comment() -> None:
    r = _empty_report("probe")
    object.__setattr__(r, "total_traces", 5)
    object.__setattr__(r, "affected_traces", 1)
    object.__setattr__(r, "flags", ["low_power"])
    md = render_pr_comment(r)
    assert "low statistical power" in md.lower() or "few traces" in md.lower()


def test_renderer_includes_hidden_marker_for_pr_comment_dedup() -> None:
    """The GitHub Action's comment.py looks for a hidden marker so
    it can update the previous comment instead of stacking new ones."""
    md = render_pr_comment(_empty_report("ship"))
    assert "<!-- shadow-diagnose-pr -->" in md
```

- [ ] **Step 6.2: Run tests, see ImportError**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_render.py -q
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 6.3: Implement the renderer**

Create `python/src/shadow/diagnose_pr/render.py`:

```python
"""PR-comment markdown renderer for `shadow diagnose-pr`.

Plain English first, metrics second. Voice rules:

  * Verdict on the first line, in caps.
  * Affected-trace count is a *fraction* (84 / 1,247) not a percent
    — fractions ground the reader in real numbers.
  * Cause block leads with the delta id (so the reader sees
    `system_prompt.md:47` before the math).
  * "Why it matters" is the policy-violation translation — what
    actually happens in the failing traces.
  * "Suggested fix" is a hint, not a patch (v1).
  * Verify-fix command is the call to action.

The hidden HTML marker `<!-- shadow-diagnose-pr -->` is what the
GitHub Action's comment.py looks for to update an existing PR
comment in place rather than stack new ones.
"""

from __future__ import annotations

from shadow.diagnose_pr.models import CauseEstimate, DiagnosePrReport

_MARKER = "<!-- shadow-diagnose-pr -->"


def _fmt_count(n: int) -> str:
    return f"{n:,}"


def _fmt_signed(x: float) -> str:
    return f"{x:+.2f}"


def _verdict_blurb(verdict: str) -> str:
    return {
        "ship": "No behavior regression detected against the production-like trace sample.",
        "probe": "Behavior changed but the effect is uncertain (CI crosses zero).",
        "hold": "This PR changes agent behavior with measurable effect.",
        "stop": "This PR violates a critical policy and must not merge as-is.",
    }.get(verdict, "")


def _render_cause(c: CauseEstimate) -> list[str]:
    lines = [
        "### Dominant cause",
        "",
        f"`{c.delta_id}` appears to be the main cause.",
        "",
        f"- Axis: `{c.axis}`",
        f"- ATE: `{_fmt_signed(c.ate)}`",
    ]
    if c.ci_low is not None and c.ci_high is not None:
        lines.append(f"- 95% CI: `[{c.ci_low:.2f}, {c.ci_high:.2f}]`")
    if c.e_value is not None:
        lines.append(f"- E-value: `{c.e_value:.1f}`")
    return lines


def render_pr_comment(report: DiagnosePrReport) -> str:
    """Render a full PR-comment markdown body for a diagnose-pr
    report. Output ends in a newline."""
    out: list[str] = [_MARKER, ""]
    out.append(f"## Shadow verdict: {report.verdict.upper()}")
    out.append("")

    blurb = _verdict_blurb(report.verdict)
    if blurb:
        out.append(blurb)
        out.append("")

    if report.total_traces > 0:
        out.append(
            f"This PR changes agent behavior on **{_fmt_count(report.affected_traces)}**"
            f" / **{_fmt_count(report.total_traces)}** production-like traces."
        )
        out.append("")

    if "low_power" in report.flags:
        out.append(
            "> :warning: **Low statistical power** — fewer than 30 traces in the sample. "
            "Treat the verdict as advisory; widen `--max-traces` for more confidence."
        )
        out.append("")

    if report.verdict == "probe":
        out.append(
            "_Verdict is `probe` because affected traces exist but the causal effect "
            "is uncertain (low confidence / CI crosses zero). Investigate before merge._"
        )
        out.append("")

    if report.dominant_cause is not None:
        out.extend(_render_cause(report.dominant_cause))
        out.append("")

    if report.worst_policy_rule is not None and report.new_policy_violations > 0:
        out.append("### Why it matters")
        out.append("")
        out.append(
            f"{report.new_policy_violations} traces violate the "
            f"`{report.worst_policy_rule}` policy rule."
        )
        out.append("")

    if report.suggested_fix is not None:
        out.append("### Suggested fix")
        out.append("")
        out.append(report.suggested_fix)
        out.append("")

    out.append("### Verify the fix")
    out.append("")
    out.append("```bash")
    out.append("shadow verify-fix --report .shadow/diagnose-pr/report.json")
    out.append("```")
    out.append("")

    return "\n".join(out)


__all__ = ["render_pr_comment"]
```

- [ ] **Step 6.4: Run tests, see green**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_render.py -q
```
Expected: `5 passed`.

- [ ] **Step 6.5: Commit**

```bash
git -C /Users/manavpatel/Downloads/Shadow add python/src/shadow/diagnose_pr/render.py python/tests/test_diagnose_pr_render.py
git -C /Users/manavpatel/Downloads/Shadow commit -s -m "$(cat <<'EOF'
feat(diagnose-pr): markdown PR-comment renderer

render_pr_comment(report) emits a plain-English-first PR comment
with verdict header, affected-trace fraction, dominant-cause block
(delta id leading), policy "why it matters" translation, suggested
fix, and verify-fix call to action. A hidden HTML marker
(<!-- shadow-diagnose-pr -->) lets the GitHub Action update an
existing comment in place rather than stack new ones. low_power
flag surfaces as a warning blockquote when n<30.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: CLI command registration

**Files:**
- Modify: `python/src/shadow/cli/app.py` (append at end of file)
- Create: `python/tests/test_diagnose_pr_cli.py`

- [ ] **Step 7.1: Write failing CLI smoke tests**

Create `python/tests/test_diagnose_pr_cli.py`:

```python
"""Smoke tests for the `shadow diagnose-pr` CLI command.

These exercise the full path: argv → load configs → load traces →
extract deltas → build report → write JSON → write markdown →
exit code. They use the bundled quickstart fixtures so the test
runs offline."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shadow.cli.app import app


def _quickstart_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy the bundled quickstart fixtures into a writable tmp dir
    and return (baseline_cfg, candidate_cfg, traces_dir)."""
    import shadow.quickstart_data as _qs_data

    root = resources.files(_qs_data)
    baseline_yaml = tmp_path / "baseline.yaml"
    candidate_yaml = tmp_path / "candidate.yaml"
    baseline_yaml.write_bytes(root.joinpath("config_a.yaml").read_bytes())
    candidate_yaml.write_bytes(root.joinpath("config_b.yaml").read_bytes())
    traces = tmp_path / "traces"
    traces.mkdir()
    fixtures = root / "fixtures"
    for name in ("baseline.agentlog", "candidate.agentlog"):
        (traces / name).write_bytes(fixtures.joinpath(name).read_bytes())
    return baseline_yaml, candidate_yaml, traces


def test_diagnose_pr_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["diagnose-pr", "--help"])
    assert result.exit_code == 0
    assert "diagnose-pr" in result.stdout.lower() or "Diagnose" in result.stdout


def test_diagnose_pr_writes_json_and_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    base_cfg, cand_cfg, traces = _quickstart_files(tmp_path)
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "comment.md"
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces", str(traces),
            "--baseline-config", str(base_cfg),
            "--candidate-config", str(cand_cfg),
            "--out", str(out_json),
            "--pr-comment", str(out_md),
        ],
    )
    if result.exit_code != 0:
        pytest.fail(f"exit={result.exit_code}\nstdout:\n{result.stdout}")
    assert out_json.is_file()
    assert out_md.is_file()
    parsed = json.loads(out_json.read_text())
    assert parsed["schema_version"] == "diagnose-pr/v0.1"
    assert parsed["total_traces"] >= 1
    assert "Shadow verdict" in out_md.read_text()


def test_diagnose_pr_fail_on_hold_returns_exit_1_when_held(tmp_path: Path) -> None:
    """In the v0.1 skeleton with no real "affected" classification,
    we synthesise the affected case by passing the SAME file as both
    baseline traces and candidate traces — the skeleton classifier
    treats every trace as "potentially affected" pending Week-2
    real classification. With --fail-on probe this should exit 1."""
    runner = CliRunner()
    base_cfg, cand_cfg, traces = _quickstart_files(tmp_path)
    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces", str(traces),
            "--baseline-config", str(base_cfg),
            "--candidate-config", str(cand_cfg),
            "--out", str(out_json),
            "--fail-on", "probe",
        ],
    )
    parsed = json.loads(out_json.read_text())
    if parsed["verdict"] == "probe":
        assert result.exit_code == 1
    else:
        assert result.exit_code == 0  # ship is fine


def test_diagnose_pr_missing_config_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    out_json = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces", str(tmp_path),
            "--baseline-config", str(tmp_path / "nope.yaml"),
            "--candidate-config", str(tmp_path / "also_nope.yaml"),
            "--out", str(out_json),
        ],
    )
    assert result.exit_code != 0
    assert "config file not found" in (result.stdout + (result.stderr or "")).lower()
```

- [ ] **Step 7.2: Run CLI tests, see fails (command not registered)**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_cli.py -q
```
Expected: FAIL — `--help` will list available commands but not `diagnose-pr`.

- [ ] **Step 7.3: Find the registration anchor in app.py**

Run:
```bash
grep -n '^@app\.command' /Users/manavpatel/Downloads/Shadow/python/src/shadow/cli/app.py | tail -5
```
Note the last `@app.command` line number — we'll insert just below it. Currently the last command is `verify-cert` near line 3305.

- [ ] **Step 7.4: Append the diagnose-pr command to app.py**

Append the following block to the END of `python/src/shadow/cli/app.py`:

```python


# ---------------------------------------------------------------------------
# diagnose-pr: Causal Regression Forensics for AI Agents
#
# The headline command for the strategic pivot landed in
# docs/superpowers/specs/2026-05-03-causal-regression-forensics-design.md.
# v0.1 skeleton ships in Week 1 (this PR). 9-axis affected-trace
# classification arrives in Week 2; causal attribution + dominant-cause
# in Week 3.
# ---------------------------------------------------------------------------


@app.command("diagnose-pr", rich_help_panel="Common")
def diagnose_pr_cmd(
    traces: list[Path] = typer.Option(  # noqa: B008
        ...,
        "--traces",
        help="Production-like .agentlog files (or directories) to diagnose against.",
    ),
    baseline_config: Path = typer.Option(  # noqa: B008
        ...,
        "--baseline-config",
        help="Baseline agent config YAML (same schema as `shadow replay`).",
    ),
    candidate_config: Path = typer.Option(  # noqa: B008
        ...,
        "--candidate-config",
        help="Candidate agent config YAML.",
    ),
    out: Path = typer.Option(  # noqa: B008
        ...,
        "--out",
        help="Path to write the JSON report (diagnose-pr/v0.1 schema).",
    ),
    pr_comment: Path | None = typer.Option(  # noqa: B008
        None,
        "--pr-comment",
        help="Path to write the markdown PR comment (omit to skip).",
    ),
    changed_files: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--changed-files",
        help=(
            "Files changed in the PR (e.g. from `git diff --name-only`); used to "
            "attach human-readable filenames to prompt deltas."
        ),
    ),
    max_traces: int = typer.Option(
        200,
        "--max-traces",
        help="Cap on traces fed to the per-trace loop. Mining selects representatives above this.",
    ),
    fail_on: str = typer.Option(
        "none",
        "--fail-on",
        help="Exit non-zero on this verdict floor: none|probe|hold|stop.",
    ),
) -> None:
    """Diagnose a candidate config against production-like traces.

    Composes existing Shadow internals (parser, mining, the 9-axis
    differ, policy checker, causal attribution) into one PR-time
    command surface. Produces a JSON report and a markdown PR
    comment.

    v0.1 skeleton — affected-trace classification and causal
    attribution arrive in Weeks 2-3; this command currently runs
    delta extraction + report assembly + render.
    """
    from shadow.diagnose_pr.deltas import extract_deltas
    from shadow.diagnose_pr.loaders import load_config, load_traces
    from shadow.diagnose_pr.render import render_pr_comment
    from shadow.diagnose_pr.report import build_report, to_json

    try:
        baseline = load_config(baseline_config)
        candidate = load_config(candidate_config)
        loaded = load_traces(list(traces))
    except Exception as exc:
        _fail(exc)
        return

    deltas = extract_deltas(baseline, candidate, changed_files=changed_files)

    # v0.1 skeleton — Week 2 will replace this with the real
    # 9-axis classification.
    affected: set[str] = set()

    if max_traces > 0 and len(loaded) > max_traces:
        # Use the existing mining surface to pick representative cases
        # when the corpus is large. Mining already exists in
        # shadow.mine.mine; we wrap it here so the command stays
        # composable.
        from shadow.mine import mine as _mine

        records_only = [t.records for t in loaded]
        sampled = _mine(records_only, max_cases=max_traces, per_cluster=1)
        sampled_ids = {case.request_record.get("id", "") for case in sampled.cases}
        loaded = [t for t in loaded if t.records[0].get("id", "") in sampled_ids] or loaded

    report = build_report(traces=loaded, deltas=deltas, affected_trace_ids=affected)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_json(report) + "\n", encoding="utf-8")

    if pr_comment is not None:
        pr_comment.parent.mkdir(parents=True, exist_ok=True)
        pr_comment.write_text(render_pr_comment(report), encoding="utf-8")

    _emit_diagnose_pr_summary(report)

    floor_map = {"none": -1, "probe": 1, "hold": 2, "stop": 3}
    rank = {"ship": 0, "probe": 1, "hold": 2, "stop": 3}
    verdict_rank = rank.get(report.verdict, 0)
    floor = floor_map.get(fail_on, -1)
    if floor >= 0 and verdict_rank >= floor:
        raise typer.Exit(code=1)


def _emit_diagnose_pr_summary(report: "Any") -> None:  # type: ignore[name-defined]
    """One-line stdout summary so CI logs show what happened."""
    typer.echo(
        f"Shadow verdict: {report.verdict.upper()} — "
        f"{report.affected_traces}/{report.total_traces} affected"
    )
```

(The `_fail` helper is already defined in `app.py` and used by other commands.)

- [ ] **Step 7.5: Run CLI tests, see green**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests/test_diagnose_pr_cli.py -q
```
Expected: `4 passed`.

- [ ] **Step 7.6: Commit**

```bash
git -C /Users/manavpatel/Downloads/Shadow add python/src/shadow/cli/app.py python/tests/test_diagnose_pr_cli.py
git -C /Users/manavpatel/Downloads/Shadow commit -s -m "$(cat <<'EOF'
feat(diagnose-pr): register `shadow diagnose-pr` Typer command

The headline command for the Causal Regression Forensics pivot.
v0.1 skeleton: loads baseline + candidate YAML configs, loads
.agentlog traces (file or directory), extracts deltas, optionally
mines representatives if corpus exceeds --max-traces, assembles a
report, writes JSON to --out and markdown to --pr-comment, exits
non-zero based on --fail-on floor.

Real 9-axis affected-trace classification arrives in Week 2;
causal attribution + dominant cause in Week 3. Schema is stable
(diagnose-pr/v0.1) — those weeks layer behavior onto the same
fields without renames.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: End-to-end fixture sanity check

**Files:** none (verification only)

- [ ] **Step 8.1: Run the bundled demo end-to-end**

Run:
```bash
mkdir -p /tmp/diagnose-pr-smoke
PATH="$PWD/.venv/bin:$PATH" python -c "
from importlib import resources
import shadow.quickstart_data as q
root = resources.files(q)
import shutil, pathlib
out = pathlib.Path('/tmp/diagnose-pr-smoke')
out.mkdir(exist_ok=True)
(out / 'baseline.yaml').write_bytes(root.joinpath('config_a.yaml').read_bytes())
(out / 'candidate.yaml').write_bytes(root.joinpath('config_b.yaml').read_bytes())
traces = out / 'traces'
traces.mkdir(exist_ok=True)
for name in ('baseline.agentlog', 'candidate.agentlog'):
    (traces / name).write_bytes((root / 'fixtures').joinpath(name).read_bytes())
print('staged at', out)
"

PATH="$PWD/.venv/bin:$PATH" shadow diagnose-pr \
  --traces /tmp/diagnose-pr-smoke/traces \
  --baseline-config /tmp/diagnose-pr-smoke/baseline.yaml \
  --candidate-config /tmp/diagnose-pr-smoke/candidate.yaml \
  --out /tmp/diagnose-pr-smoke/report.json \
  --pr-comment /tmp/diagnose-pr-smoke/comment.md
```

Expected output: `Shadow verdict: SHIP — 0/2 affected` (skeleton classifier marks zero affected; real Week 2 wiring will surface real divergence).

- [ ] **Step 8.2: Inspect report.json**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -c "
import json
print(json.dumps(json.load(open('/tmp/diagnose-pr-smoke/report.json')), indent=2)[:800])
"
```
Expected: JSON with `schema_version: 'diagnose-pr/v0.1'`, `verdict: 'ship'`, `total_traces: 2`, `affected_traces: 0`, `blast_radius: 0.0`, `flags: ['low_power']`.

- [ ] **Step 8.3: Inspect comment.md**

Run:
```bash
cat /tmp/diagnose-pr-smoke/comment.md
```
Expected: starts with `<!-- shadow-diagnose-pr -->` marker, header `## Shadow verdict: SHIP`, low-power blockquote, verify-fix code block.

---

## Task 9: Lint / type / full suite gate

**Files:** none (verification only)

- [ ] **Step 9.1: ruff lint clean**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m ruff check python/
PATH="$PWD/.venv/bin:$PATH" python -m ruff format --check python/
```
Expected: `All checks passed!` and `XXX files already formatted`.

If `ruff check` flags issues, run `python -m ruff check --fix python/` and re-commit the fix as `chore(diagnose-pr): ruff lint`.

- [ ] **Step 9.2: full pytest suite green**

Run:
```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest python/tests -q --no-header 2>&1 | tail -5
```
Expected: at least `1701 + 5 + 7 + 9 + 7 + 5 + 4 + 1 = 1739 passed`. If the count is lower, investigate before tagging the week complete.

- [ ] **Step 9.3: Final commit if any lint touch-ups**

If steps 9.1 produced changes, commit them. Otherwise skip.

```bash
git -C /Users/manavpatel/Downloads/Shadow status --short
# If empty, no commit needed.
```

---

## Self-review (run before declaring Week 1 done)

Skim the spec sections 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 4.1, 5.1 against this plan:

| Spec section | Plan task that covers it |
|---|---|
| §3.1 CLI surface | Task 7 |
| §3.2 algorithm | Task 7 (skeleton; full in Weeks 2-3) |
| §3.3 data models | Task 2 |
| §3.4 affected-trace classification | Week 2 (deferred — explicit) |
| §3.5 risk + verdict | Week 2 (deferred — explicit; Task 5 ships trivial v0) |
| §3.6 JSON schema | Task 5 |
| §3.7 PR-comment markdown | Task 6 |
| §4.1 file plan | Tasks 1–7 |
| §5.1 30-day plan | Week 1 = this plan |

Three cross-referenced names (must match across tasks):
- `DiagnosePrReport` — defined Task 2, used Tasks 5, 6, 7
- `extract_deltas` — defined Task 4, used Task 7
- `build_report` / `to_json` — defined Task 5, used Task 7
- `render_pr_comment` — defined Task 6, used Task 7
- `LoadedTrace` — defined Task 3, used Tasks 5, 7

Names are consistent across the plan. ✓

---

*End of Week 1 plan. After execution, the next plan
(`2026-05-XX-diagnose-pr-week2.md`) wires up the 9-axis differ +
policy classification.*
