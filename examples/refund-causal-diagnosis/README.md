# Refund-agent causal diagnosis

The wedge demo for `shadow diagnose-pr`. Shows the full
diagnose → fix → verify loop end-to-end against a refund-processing
agent where the candidate config drops a critical "always confirm
before refunding" instruction from the system prompt.

## What this proves

Given:

- **Baseline** config + traces — agent confirms then issues refund.
- **Candidate** config — system prompt edited; "always confirm" is gone.
- **Policy** — `must_call_before(confirm_refund_amount, issue_refund)`.

Shadow's `diagnose-pr` answers in one PR comment:

1. **Did behavior change?** Yes — 3/3 production-like traces affected.
2. **Which exact change caused it?** `prompt.system` (axis: trajectory).
3. **How confident?** ATE +0.60, 95% CI [0.60, 0.60], E-value 6.7.
4. **What policy is violated?** `confirm-before-refund`.
5. **What fix should we verify?** Restore the prompt instruction.

Then `verify-fix` re-runs the affected traces against a fixed
candidate and confirms the regression is reversed: PASS, 3/3.

## Layout

```
refund-causal-diagnosis/
├── baseline.yaml              compliant agent config
├── candidate.yaml             config with prompt instruction removed
├── policy.yaml                must_call_before policy rule
├── prompts/                   the prompt files cited in PR comments
│   ├── baseline.md
│   └── candidate.md
├── baseline_traces/           three pre-recorded compliant traces
├── candidate_traces/          three traces showing the regression
├── demo.sh                    one-line runner
└── README.md                  this file
```

## Running it

```bash
./demo.sh
```

Default backend is `mock` (deterministic, free, offline) — perfect
for CI and contributor onboarding. The mock backend produces real
bootstrap CIs from a synthetic per-delta intervention; numbers
aren't grounded in real-world data but the structure of the
output is exactly what `--backend live` produces.

To run against the real OpenAI API:

```bash
export OPENAI_API_KEY=sk-...
SHADOW_DEMO_BACKEND=live ./demo.sh
```

## Verify the fix

```bash
# In this demo, the "fixed candidate" is just the baseline itself
# (compliant) — verify-fix confirms the regression reverses.
shadow verify-fix \
  --report .shadow/diagnose-pr/report.json \
  --traces baseline_traces \
  --fixed-traces baseline_traces \
  --policy policy.yaml \
  --out .shadow/diagnose-pr/verify.json
```

Expected output:

```
Shadow verify-fix: PASS — 3/3 affected reversed, 0/0 safe regressed
```

## CI use

The thinnest CI gate is `shadow gate-pr` — verdict-mapped exit
codes (0 ship, 1 hold/probe, 2 stop, 3 internal error):

```yaml
- name: Shadow gate
  run: |
    shadow gate-pr \
      --traces           baseline_traces/ \
      --candidate-traces candidate_traces/ \
      --baseline-config  baseline.yaml \
      --candidate-config candidate.yaml \
      --policy           policy.yaml \
      --pr-comment       shadow-comment.md
```
