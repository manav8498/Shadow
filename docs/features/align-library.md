# `shadow.align` — standalone alignment library

> **Phase 6 of the Causal Regression Forensics roadmap.** Shadow's alignment
> engine, packaged as a reusable category primitive. Used by external tools
> that want trace alignment + divergence detection WITHOUT pulling in the
> diagnose-pr / 9-axis differ / policy machinery.

## Surface (v0.1)

Five public functions, one stable import:

```python
from shadow.align import (
    align_traces,         # full per-turn pairing
    first_divergence,     # find the FIRST meaningful divergence
    top_k_divergences,    # ranked top-K
    trajectory_distance,  # Levenshtein on flat tool-name sequences
    tool_arg_delta,       # structural diff of two JSON values
)
```

Plus four dataclasses: `Alignment`, `AlignedTurn`, `Divergence`, `ArgDelta`.

## Quick start

```python
from shadow import _core
from shadow.align import first_divergence, top_k_divergences, trajectory_distance

baseline = _core.parse_agentlog(open("baseline.agentlog", "rb").read())
candidate = _core.parse_agentlog(open("candidate.agentlog", "rb").read())

fd = first_divergence(baseline, candidate)
print(f"First divergence: {fd.kind} on {fd.primary_axis} at turn {fd.baseline_turn}")
# → First divergence: structural_drift on trajectory at turn 0

for d in top_k_divergences(baseline, candidate, k=3):
    print(f"  [{d.primary_axis}] {d.explanation}")

# Trajectory-only path (no .agentlog records needed):
print(trajectory_distance(["search", "summarize"], ["search", "edit"]))
# → 0.5
```

## When to reach for `shadow.align`

* **You have an eval framework** (RAGAS, TruLens, DeepEval, LangSmith) and
  want a stable trace-alignment primitive. `align_traces` returns the per-turn
  pairing your existing axes can layer on top of.
* **You're building an observability platform** and want to surface "the FIRST
  thing that diverged between this run and last week's golden trace."
  `first_divergence` answers in one call.
* **You're writing a regression test harness** with stored tool-call sequences
  per scenario. `trajectory_distance` lets you assert "the tool path is at
  most 30% different from baseline."
* **You have a tool-schema linter** and want to detect added/removed/changed
  argument fields between two recorded calls. `tool_arg_delta` returns
  typed deltas keyed by JSON path.

## What it doesn't do

* It doesn't classify *severity* — that's the 9-axis differ's job
  (`shadow._core.compute_diff_report`).
* It doesn't compute causal attribution — that's `shadow.diagnose_pr.attribution`.
* It doesn't enforce policy — that's `shadow.hierarchical`.

For all three of those, use `shadow diagnose-pr` end-to-end.

## API reference

### `align_traces(baseline, candidate) -> Alignment`

Pair every baseline chat turn to its best-match candidate turn (and vice
versa). Returns an `Alignment` whose `turns` list contains one `AlignedTurn`
per pair, with optional `None` indices for inserted/dropped turns.

### `first_divergence(baseline, candidate) -> Divergence | None`

Find the FIRST point at which the two traces meaningfully differ in
alignment order. Returns `None` when traces agree end-to-end.

### `top_k_divergences(baseline, candidate, k=5) -> list[Divergence]`

Top-K ranked by severity × downstream impact (the underlying Rust differ's
own ranking). Returns `[]` for identical traces. Raises `ValueError` for
`k < 1`.

### `trajectory_distance(baseline_tools, candidate_tools) -> float`

Levenshtein edit distance between two flat tool-name sequences, normalised
to `[0.0, 1.0]` by the longer length. `0.0` = identical, `1.0` = pure
substitution. Pure Python — no Rust dependency.

### `tool_arg_delta(a, b) -> list[ArgDelta]`

Structural diff between two JSON values. Walks dicts, lists, and scalars;
produces typed deltas (`added` / `removed` / `changed` / `type_changed`)
keyed by slash-separated JSON-pointer paths. Pure Python.

## Implementation notes

* Three of five functions (`align_traces`, `first_divergence`,
  `top_k_divergences`) delegate to `shadow._core.compute_diff_report`'s
  internal alignment, so the per-turn pairing matches what the 9-axis
  differ already does. There's exactly one canonical alignment algorithm
  in the project, not two.
* `trajectory_distance` and `tool_arg_delta` don't themselves call
  Rust, but importing `shadow.align` still loads the parent `shadow`
  package which pulls in the Rust extension via `shadow/__init__.py`.
  So the v0.1 align surface still requires the Rust extension to be
  installed. The spec-literal top-level `shadow_align` package (no
  parent dependency) is deferred to v0.2.

## Asymmetric corpora

When one side has chat pairs and the other is empty, `first_divergence`
returns a `Divergence` with `kind="structural_drift_full"` and
`primary_axis="trajectory"` — the underlying Rust differ surfaces
no per-turn divergence in this case (its alignment loop never fires
on a zero-length side), so the Python wrapper detects the asymmetry
up front and stamps an explicit kind. `align_traces` still returns
zero turns for the same case; the asymmetry is a structural-corpus
fact, not a per-turn one.

When both sides are empty, `first_divergence` returns `None` — there
is nothing to diverge.

## Performance

`trajectory_distance` is O(n²) DP Levenshtein. 1000-tool sequences
run in ~0.12 s; 2000-tool sequences in ~0.51 s. For very long
sequences (>5 K), use a streaming approximation upstream.

## Stability

This API is `v0.1`. The dataclass field names and function signatures
are stable; future versions add fields without renaming or removing
existing ones.

## Three-language ports

The same surface ships in three languages. Same algorithms, byte-identical
results on shared inputs (verified by parity tests in each port).

| Language | Package | `trajectory_distance` | `tool_arg_delta` | `align_traces` / `first_divergence` / `top_k_divergences` |
|---|---|:---:|:---:|---|
| Python | `shadow.align` (PyPI: `shadow-diff`) | ✓ pure Python | ✓ pure Python | ✓ via `shadow._core` |
| TypeScript | `shadow-diff/align` (npm: `shadow-diff`) | ✓ pure TS | ✓ pure TS | v0.1 stubs (v0.2 napi-rs) |
| Rust | `shadow-align` (crates.io) | ✓ pure Rust | ✓ pure Rust | v0.1 stubs (v0.2 re-export from shadow-core) |

The cross-language parity tests are committed in:

* `python/tests/test_align_library.py` (Python)
* `typescript/test/align.test.ts` (TypeScript)
* `crates/shadow-align/tests/cross_language_parity.rs` (Rust)

If a number drifts between any two ports, fix it in all three.

## Standalone-tool example

See [`examples/standalone-align/compare_tool_trajectories.py`](https://github.com/manav8498/Shadow/blob/main/examples/standalone-align/compare_tool_trajectories.py)
for a runnable demo: a non-Shadow tool calls only `shadow.align`, produces
real trajectory distances + tool-arg deltas + first-divergence output. Zero
imports from `shadow.diagnose_pr`, `shadow.causal`, or `shadow.cli`.
