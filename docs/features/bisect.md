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

```
semantic:
  prompt.system            74.9% [71.0%, 89.2%]  sel_freq=1.00  ✓
  model_id x prompt.system 13.8% [3.8%, 17.5%]   sel_freq=0.89  ✓

latency:
  model_id          61.3% [59.2%, 68.0%]  sel_freq=1.00  ✓
  tools             19.7% [15.3%, 22.4%]  sel_freq=0.94  ✓
  model_id x tools  16.6% [12.0%, 19.4%]  sel_freq=0.96  ✓
```

Reading the signal:

- `78%` = attributed share of that axis's total delta
- `[71%, 89%]` = 95% bootstrap percentile CI
- `sel_freq=1.00` = how often the delta was selected across bootstrap
  resamples (Meinshausen-Bühlmann stability)
- `✓` = conjunction-significant (selection_frequency ≥ 0.6 AND
  CI excludes 0)
- `x` = pairwise interaction term

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
