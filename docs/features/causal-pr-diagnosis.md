# Causal PR Diagnosis

> **Shadow tells you which prompt, model, tool schema, or config change broke
> your AI agent — proven against production-like traces before merge.**

`shadow diagnose-pr` is the headline product. Three commands close the loop
from regression detection through fix verification:

| Step | Command | Output |
|---|---|---|
| 1. Diagnose | `shadow diagnose-pr` | verdict + dominant cause + PR-comment markdown |
| 2. Verify | `shadow verify-fix` | pass/fail on the fix's effectiveness |
| 3. Gate (CI) | `shadow gate-pr` | exit code mapped to verdict (0/1/2/3) |

---

## How it works

1. **Load** baseline + candidate configs and `.agentlog` traces.
2. **Mine** representative cases when corpora exceed `--max-traces` (default 200).
3. **Diff** each paired trace via the 9-axis Rust differ — semantic, trajectory,
   safety, verbosity, latency, cost, reasoning, judge, conformance.
4. **Check policy** if `--policy` is supplied: detect new violations (regressions)
   on rules like `must_call_before(confirm_refund_amount, issue_refund)`.
5. **Attribute causes** via one of three backends:
   - `recorded` (default): observed deltas with confidence ≤ 1.0; no causal CI.
   - `mock`: deterministic per-delta intervention with bootstrap CI + E-value.
     Surfaces a "synthetic mock backend" disclosure in the PR comment.
   - `live`: per-baseline-trace OpenAI replay; corpus-mean divergence; `--max-cost`
     USD cap aborts before runaway spend.
6. **Verdict** — ship / probe / hold / stop — from
   `risk.classify_verdict(affected, has_dangerous_violation, has_severe_axis)`.
7. **Render** a JSON report (schema `diagnose-pr/v0.1`) and a Markdown PR comment.

---

## The PR comment

Plain English first, metrics second:

```markdown
## Shadow verdict: STOP

This PR violates a critical policy and must not merge as-is.
This PR changes agent behavior on **3** / **3** production-like traces.

### Dominant cause
`prompt.system` appears to be the main cause.
- Axis: `trajectory`
- ATE: `+0.60`
- 95% CI: `[0.60, 0.60]`
- E-value: `6.7`

### Why it matters
3 traces violate the `confirm-before-refund` policy rule.

### Suggested fix
Review the prompt change at `prompts/candidate.md` —
restore the instruction or constraint it removed.

### Verify the fix
```bash
shadow verify-fix --report .shadow/diagnose-pr/report.json
```
```

---

## Verdict matrix

| Verdict | Condition | gate-pr exit |
|---|---|---|
| `ship` | no affected traces, no dangerous violation | 0 |
| `probe` | (reserved for Week-5 CI distinction) | 1 |
| `hold` | affected traces, no severe axis, no dangerous violation | 1 |
| `stop` | severe axis regression OR dangerous-tool policy violation | 2 |
| (internal error) | tooling/pipeline failure | 3 |

"Dangerous tool" is a `must_call_before` / `must_call_once` rule with
`severity: error|critical` AND either `tags: [dangerous]` OR a v1 keyword
(refund, pay, transfer, wire, delete, drop, shutdown, revoke, grant, escalate).

---

## Three backends

### `recorded` (default — offline, free)

No replay. The runner enumerates the `extract_deltas` output and reports
each as a candidate cause with `confidence=0.5` (multi-delta) or `1.0`
(single-delta + observed divergence). No bootstrap CI.

Use for fast PR-time runs against pre-recorded traces, where the verdict
+ blast radius + policy violations are the real signal.

### `mock` (synthetic, deterministic, free)

Builds a synthetic per-delta replay function from the extracted deltas:

| DeltaKind | Axis | Synthetic ATE |
|---|---|---:|
| `prompt` | trajectory | 0.6 |
| `policy` | safety | 0.7 |
| `tool_schema` | trajectory | 0.5 |
| `retriever` | semantic | 0.5 |
| `model` | verbosity | 0.4 |
| `temperature` | verbosity | 0.3 |
| `unknown` | semantic | 0.2 |

The PR comment surfaces a **synthetic_mock** disclosure
(`> :information_source: Synthetic mock backend...`) so readers don't
mistake the synthetic numbers for real LLM behavior. Output structure
(bootstrap CI, E-value) matches `--backend live` exactly — the demo
runs deterministically without spending money.

### `live` (real OpenAI replay)

Each baseline trace is anchored to its own `OpenAIReplayer` (one
`(user_message, response_text)` pair per trace). For each
single-delta perturbation of the baseline config, the runner sends
the perturbed config to OpenAI **once per trace** and means the
divergence vectors. `--max-cost USD` caps total spend; `OPENAI_API_KEY`
is read from the env (never accepted as a parameter).

Pricing (USD per million tokens, conservative-high):

| Model | Input | Output |
|---|---:|---:|
| gpt-4o-mini | 0.150 | 0.600 |
| gpt-4o | 2.50 | 10.00 |
| gpt-4.1-mini | 0.40 | 1.60 |
| gpt-4.1 | 2.00 | 8.00 |
| (other) | 0.50 | 2.00 |

---

## CI integration

The `shadow-diagnose-pr` GitHub Action wraps the command and posts the
PR comment via the existing dedup-by-marker `comment.py`:

```yaml
- name: Shadow diagnose-pr
  uses: manav8498/Shadow/.github/actions/shadow-diagnose-pr@main
  with:
    baseline-traces: prod-traces/
    candidate-traces: candidate-traces/
    baseline-config: baseline.yaml
    candidate-config: candidate.yaml
    policy: shadow-policy.yaml
    backend: recorded
    fail-on-hold: 'true'
  env:
    # Only needed for `backend: live`.
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Verify-fix

After landing a fix, re-record traces against the fixed candidate and run:

```bash
shadow verify-fix \
  --report      .shadow/diagnose-pr/report.json \
  --traces      baseline_traces/ \
  --fixed-traces fixed_candidate_traces/ \
  --policy      shadow-policy.yaml \
  --out         .shadow/diagnose-pr/verify.json
```

Pass criteria (defaults):

- `affected_reversed_rate >= 0.90` (90% of regressed traces now match baseline)
- `safe_trace_regression_rate <= 0.02` (no more than 2% collateral damage)
- `new_policy_violations == 0` (when `--policy` supplied)

Exit 0 on pass, 1 on fail with explicit `fail_reasons` in stdout + JSON.

---

## Performance

Measured on a 2024 MacBook Pro, M-series, single Python 3.11:

| Workload | Duration | Memory |
|---|---:|---:|
| 100 paired pairs / `recorded` | 0.21s | 40 MB |
| 1,000 paired pairs / `recorded` | 0.85s | 74 MB |
| 5,000 single-side traces / `recorded` (mining) | 0.54s | 80 MB |
| 1,000 paired pairs / `mock` + bootstrap (500) | 0.95s | 85 MB |

Above 16 paired traces, per-pair diff + policy check runs in a
ThreadPoolExecutor (Rust differ + regex policy both release the GIL).

---

## See also

- The runnable demo: [`examples/refund-causal-diagnosis/`](../../examples/refund-causal-diagnosis/)
- The design spec: [`docs/superpowers/specs/2026-05-03-causal-regression-forensics-design.md`](../superpowers/specs/2026-05-03-causal-regression-forensics-design.md)
- Implementation plans (Weeks 1–4): [`docs/superpowers/plans/`](../superpowers/plans/)
