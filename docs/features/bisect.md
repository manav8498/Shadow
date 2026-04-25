# Causal bisection

When a PR changes multiple things at once, prompt edit, model swap,
tool-schema rename, param change, Shadow's `bisect` command tells
you which specific delta moved which axis.

Under the hood: LASSO regression over a full-factorial or
Plackett-Burman design matrix across the config deltas, fit per axis.
The hardened variant (v0.2+) adds pairwise interaction effects and
emits 95% bootstrap CIs on every attribution weight.

## Basic usage

```bash
shadow bisect config_a.yaml config_b.yaml \
  --traces baseline.agentlog --candidate-traces candidate.agentlog
```

Output: per-axis attribution of the movement to specific config
deltas (`model`, `params.temperature`, `prompt.system`, each `tools[i]`).

## Hardened mode (live backend)

Pass `--backend anthropic|openai` to replay each corner of the
design matrix through a real LLM:

```bash
shadow bisect config_a.yaml config_b.yaml \
  --traces baseline.agentlog \
  --backend anthropic
```

This is the "LASSO-over-corners" mode: 2^k corner replays (k = number
of differing config categories), LASSO fit per axis with
Meinshausen-Bühlmann stability selection, and honest bootstrap CIs.

## Output format

`shadow bisect` defaults to a terminal renderer that hedges the
language. Without sandboxed counterfactual replay, attribution is
**correlational, not causally proven** — LASSO + stability selection
narrows the field, but two correlated deltas can still split a single
axis's true cause and look equally significant. The renderer leads
with this caveat:

```
Bisect attribution (estimated, correlational). Confirm with
`shadow replay --backend <provider>` for causal proof.

semantic:
  prompt.system          est. 74.9%   95% CI [71.0%, 89.2%]  sel_freq=1.00  (stable, CI excludes 0)

latency:
  model_id               est. 61.3%   95% CI [59.2%, 68.0%]  sel_freq=1.00  (stable, CI excludes 0)
  tools                  est. 19.7%   95% CI [15.3%, 22.4%]  sel_freq=0.94  (stable, CI excludes 0)
```

Reading the signal:

- `est. 74.9%` is the estimated attributed share of the axis's total
  delta. The `est.` prefix is intentional — this is a regression
  coefficient, not a proof of causation.
- `95% CI [a%, b%]` is the 95% bootstrap percentile CI on the share.
- `sel_freq=1.00` is how often the delta survived across bootstrap
  resamples (Meinshausen-Bühlmann stability).
- The trailing qualifier spells out which conditions a row passes:
  `(stable, CI excludes 0)` means selection frequency ≥ 0.6 *and* the
  CI excludes zero. Rows that pass screening but fail one say so
  explicitly (`screening only`, `CI crosses 0`, `weak signal`).

Use `--format json` to get the raw attribution dict for scripting,
or `--format markdown` for PR comments.

## When to use

- Multi-delta PRs where the nine-axis diff shows movement but the
  cause is ambiguous
- Picking between rollback candidates ("is the regression from the
  prompt or the model?")
- Post-incident retros, attributes the behaviour change to the
  specific commit's specific field

See the paper references in the module docstring for the statistical
foundation: Chatterjee & Lahiri 2011 (residual bootstrap), Lim &
Hastie 2015 (strong hierarchy via `glinternet`), Efron 2014 (CI
width under re-tuning), Meinshausen & Bühlmann 2010 (stability
selection).
