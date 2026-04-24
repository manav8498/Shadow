# Pricing table

The `--pricing` flag on `shadow diff` and `shadow bisect` accepts a
JSON file mapping model name → pricing details.

## Rich dict form (recommended)

```json
{
  "claude-opus-4-7": {
    "input": 15e-6,
    "output": 75e-6,
    "cached_input": 1.5e-6,
    "cached_write_5m": 18.75e-6,
    "cached_write_1h": 30e-6,
    "reasoning": 75e-6,
    "batch_discount": 0.5
  },
  "claude-sonnet-4-6": {
    "input": 3e-6,
    "output": 15e-6
  }
}
```

All rates are USD per token (note the `e-6`, per-token, not per-million-tokens).

- `input` / `output`: required
- `cached_input`: rate for cached input tokens (falls back to `input`)
- `cached_write_5m` / `cached_write_1h`: rates for writing to the
  short / long cache tiers
- `reasoning`: rate for thinking tokens (falls back to `output`)
- `batch_discount`: multiplier applied to batch-mode invocations

## Legacy tuple form

Accepted for backward compatibility:

```json
{
  "claude-opus-4-7": [15e-6, 75e-6]
}
```

## Unknown models

Models not present in the pricing table contribute $0 to cost axis
and cost attribution. No crash, no warning, Shadow won't guess
pricing.

## Example

Ship a `pricing.json` alongside your fixtures:

```bash
shadow diff fixtures/baseline.agentlog fixtures/candidate.agentlog \
  --pricing pricing.json \
  --output-json diff.json
```
