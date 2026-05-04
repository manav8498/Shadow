<!-- shadow-diagnose-pr -->

## Shadow verdict: STOP

This PR violates a critical policy and must not merge as-is.

This PR changes agent behavior on **3** / **3** production-like traces.

> :warning: **Low statistical power** — fewer than 30 traces in the sample. Treat the verdict as advisory; widen `--max-traces` for more confidence.

> :information_source: **Synthetic mock backend.** Cause magnitudes below come from a deterministic per-delta heuristic (not real LLM behavior). Re-run with `--backend live` for a grounded estimate.

### Dominant cause

`prompt.system` appears to be the main cause.

- Axis: `trajectory`
- ATE: `+0.60`
- 95% CI: `[0.60, 0.60]`
- E-value: `6.7`

### Why it matters

3 traces violate the `confirm-before-refund` policy rule.

### Suggested fix

Review the prompt change at `prompts/candidate.md` — restore the instruction or constraint it removed.

### Verify the fix

```bash
shadow verify-fix --report .shadow/diagnose-pr/report.json
```
