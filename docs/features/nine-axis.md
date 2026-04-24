# Nine-axis diff

Shadow's core report is a nine-row table with a statistically-rigorous
delta + 95% CI on each axis. The axes cover what actually matters
about an agent's behaviour, not what's easy to measure.

## The nine axes

| Axis | What it measures | Unit |
|---|---|---|
| **semantic** | Final-response text similarity (TF-IDF cosine by default; sentence-transformer embeddings with the `[embeddings]` extra) | 0–1 (1 = identical) |
| **trajectory** | Tool-call sequence edit distance | 0–1 (0 = identical) |
| **safety** | Refusal rate | 0–1 |
| **verbosity** | Response length in output tokens | tokens |
| **latency** | End-to-end wall-clock | ms |
| **cost** | Per-response USD spend | $ |
| **reasoning** | Reasoning / thinking token depth | tokens |
| **judge** | LLM-as-judge score (empty unless `--judge` is set) | 0–1 |
| **conformance** | Schema / JSON parseability rate | 0–1 |

## Statistical guarantees

- **Bootstrap 95% CIs**: 1000 paired resamples per axis, percentile
  method. CI bounds are emitted even on small samples, the
  `low_power` flag fires automatically when n < 5.
- **Severity tiers**: `none / minor / moderate / severe` computed
  from both the effect size and the CI bracket. A delta whose CI
  crosses zero is capped at `minor`, regardless of point estimate.
- **No hidden coercion**: raw units per axis. No "normalised score"
  that hides magnitude.

## Output format

### Terminal

```
Shadow diff, 5 response pair(s)
baseline : sha256:8fc9f133…
candidate: sha256:11a5b3a2…

┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━┓
┃axis       ┃ baseline ┃ candidate┃     delta ┃   95% CI ┃ severity ┃ flags ┃ n┃
┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━┩
│semantic   │    1.000 │    0.061 │    -0.939 │ [-0.95, -0.85] │ severe │       │ 5│
│trajectory │    0.000 │    1.000 │    +1.000 │ [+1.00, +1.00] │ severe │       │ 5│
│conformance│    1.000 │    0.000 │    -1.000 │ [-1.00, -1.00] │ severe │       │ 5│
│...
└───────────┴──────────┴──────────┴───────────┴──────────┴──────────┴───────┴──┘

worst severity: severe
```

### Markdown (PR comment)

Same data as a GitHub-flavoured markdown table, with emoji severity
indicators:

```
| axis | baseline | candidate | delta | 95% CI | severity | n |
|------|---------:|----------:|------:|:-------|:---------|---:|
| semantic | 1.000 | 0.061 | -0.939 | [-0.95, -0.85] | 🔴 severe | 5 |
| trajectory | 0.000 | 1.000 | +1.000 | [+1.00, +1.00] | 🔴 severe | 5 |
```

### JSON (machine-readable)

```bash
shadow diff baseline.agentlog candidate.agentlog --output-json diff.json
```

```json
{
  "rows": [
    {"axis": "semantic", "baseline_median": 1.0, "candidate_median": 0.061,
     "delta": -0.939, "ci95_low": -0.95, "ci95_high": -0.85,
     "severity": "severe", "flags": [], "n": 5}
  ],
  "drill_down": [...],
  "first_divergence": {...},
  "divergences": [...],
  "recommendations": [...]
}
```

## Reading the report

- **Worst severity** appears at the top. If it's `severe`, stop and
  read the "What this means" paragraph first.
- **Low n warning**: at n < 5, bootstrap CIs are unreliable. Record
  more pairs (10+ is a conservative floor).
- **Top divergences** lists the specific turn(s) where the candidate
  diverged from the baseline. Structural > decision > style drift in
  priority order.
- **Recommendations**: prescriptive one-line fixes, with severity
  tier and action kind (restore / remove / revert / review / verify).
- **Drill-down**: ranks the most regressive pair(s) with per-axis
  normalised scores. Use this to click into the worst specific turn.

## See also

- [Judges](judges.md): populating the `judge` axis
- [Causal bisection](bisect.md): attribute regressions to specific
  config deltas
