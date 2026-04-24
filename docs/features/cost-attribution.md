# Cost attribution

When a PR changes cost, the question isn't just "did it?" — it's
"*why*, and by how much per user-facing session?" Cost attribution
decomposes the per-session cost delta into three independent sources.

## The decomposition

```
total_delta = model_swap + token_movement + mix_residual
```

- **model_swap**: how much of the delta is the candidate's
  price-per-token vs the baseline's, holding token counts constant
  at candidate levels
- **token_movement**: how much is the token-count change, holding
  price constant at baseline
- **mix_residual**: non-additive interaction (simultaneous model
  swap + token change)

When `|residual| > 10% of |total_delta|` the decomposition is
flagged as less trustworthy — the two-factor story is incomplete.

## Output

```
cost attribution (per session):
  session #0: $0.0870 → $0.0174 (Δ $-0.0696, -80.0%)
    model swap claude-opus-4-7→claude-sonnet-4-6: $-0.0696 (+100%)
    token movement:                                $+0.0000 (-0%)
  total: $0.0870 → $0.0174 (Δ $-0.0696)
```

## In the markdown / PR comment

```
| session | baseline | candidate | Δ | model swap | token move | mix |
|--------:|---------:|----------:|--:|-----------:|-----------:|----:|
| #0      | `$0.0870` | `$0.0174` | `$-0.0696` | `$-0.0696` (+100%) | `$+0.0000` (-0%) | `$+0.0000` (—) |
```

## When it surfaces

Attribution always runs on `shadow diff`, but only prints when the
total cost delta is non-zero. Trace pairs with no pricing data (or
`--pricing` unset) won't produce attribution output.

## Pricing table

```bash
shadow diff baseline.agentlog candidate.agentlog --pricing pricing.json
```

With `pricing.json`:

```json
{
  "claude-opus-4-7":   {"input": 1.5e-5, "output": 7.5e-5},
  "claude-sonnet-4-6": {"input": 3e-6,   "output": 1.5e-5, "cached_input": 3e-7}
}
```

Rich-dict pricing supports `input` / `output` / `cached_input` /
`cached_write_5m` / `cached_write_1h` / `reasoning` / `batch_discount`.
Legacy tuple `[input, output]` also accepted.

Unknown models contribute $0 (no crash).
