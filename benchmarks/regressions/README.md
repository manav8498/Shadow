# Shadow Regression Benchmark

Twenty synthetic-but-realistic baseline/candidate trace pairs, one per
regression class that matters in production LLM agents. A permanent
test bed for Shadow: if a code change drops the catch-rate, the CI
script fails.

## What it measures

For each case, the runner:
1. Generates a baseline trace and a candidate trace with a known
   regression embedded.
2. Runs `_core.compute_diff_report(...)` on them.
3. Asserts the expected axis reached at least the expected minimum
   severity.

Cases are grouped into six regression classes:

| # | Class | Cases |
|---|-------|-------|
| 1-5 | Tool-call regressions | reorder, rename, skip, schema drift, arg swap |
| 6-10 | Text-output regressions | verbosity blowup, truncation, over-apology, language drift, formatting loss |
| 11-13 | Safety regressions | over-refusal, missed-refusal, hedging |
| 14-16 | Format-conformance regressions | JSON→prose, missing field, extra preamble |
| 17-19 | Latency/cost regressions | 5x slowdown, model swap, token blowup |
| 20 | Control | identical baseline and candidate |

## Current catch-rate: 19/20

Case `10_formatting_loss` is marked as a **known limitation** — the
BM25 lexical semantic axis cannot distinguish markdown formatting
from prose when token overlap is high. Catching it honestly requires
either the `[embeddings]` extra (real sentence-transformer
similarity) or a format-aware custom Judge rubric.

The runner returns exit 0 as long as every non-known-limit case is
caught. If a case marked `known_limit: True` starts being caught, the
runner prints a "consider un-marking" hint — a green light to tighten
the benchmark.

## Running

```bash
python benchmarks/regressions/run.py           # human-readable table
python benchmarks/regressions/run.py --json    # machine-readable
```

## Adding a case

1. Append a new `case_NN_short_name()` function to `cases.py`.
2. Return `(baseline_records, candidate_records, expected)` where
   `expected` has `axis`, `min_severity`, `description`, and
   optionally `known_limit: True`.
3. Add the case name + function to the `CASES` list at the bottom of
   `cases.py`.

For cases that Shadow genuinely cannot catch with its current axes,
mark `known_limit: True` and document why. Don't weaken the
assertion; document the limitation.
