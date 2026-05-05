# Wire into CI

Get a Shadow PR comment that names the exact change that broke the agent — in about ten minutes from a clean repo. Two paths: the composite GitHub Action (recommended), or a hand-rolled workflow if you need to customise.

## The one-line setup

```bash
shadow init --github-action
```

Drops `.github/workflows/shadow-diagnose-pr.yml` into your repo. The generated workflow uses env vars at the top so you can edit the trace + config paths in one place:

```yaml
env:
  BASELINE_TRACES:  fixtures/baseline_traces
  CANDIDATE_TRACES: fixtures/candidate_traces
  BASELINE_CONFIG:  configs/baseline.yaml
  CANDIDATE_CONFIG: configs/candidate.yaml
  POLICY:           configs/shadow-policy.yaml
  BACKEND:          recorded   # `mock` for demos · `live` for real ATE + CI + E-value
  MAX_COST_USD:     "5.00"     # only used when BACKEND=live
```

Edit those to match the layout you commit, push the PR, and every PR gets a `shadow diagnose-pr` comment with the verdict, the dominant cause, and the suggested fix.

## Tighter alternative — the composite action

If you'd rather keep the workflow short, use the composite action directly. It handles the awkward parts: marker-dedup so re-runs edit the existing comment instead of stacking new ones, forked-PR fallback that writes to the workflow summary when the token is read-only, exit-code mapping, and a baseline-ref discovered automatically from the PR.

```yaml
name: shadow diagnose-pr

on:
  pull_request:

permissions:
  pull-requests: write
  contents: read

jobs:
  diagnose:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0   # diagnose-pr uses git blame to cite prompt-file changes
      - uses: manav8498/Shadow/.github/actions/shadow-diagnose-pr@main
        with:
          baseline-traces:  fixtures/baseline_traces
          candidate-traces: fixtures/candidate_traces
          baseline-config:  configs/baseline.yaml
          candidate-config: configs/candidate.yaml
          policy:           configs/shadow-policy.yaml
          backend:          recorded   # offline; no API spend
          fail-on-hold:    'true'      # block merge on hold/probe/stop
        env:
          # Only when backend: live
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

`fail-on-hold: true` blocks merge on `hold`/`probe`/`stop`; the default lets only `stop` block.

## Three backends

| `backend:` | Use when | Spend |
|---|---|---|
| `recorded` (default) | Pre-recorded baseline + candidate traces committed to the repo | $0 |
| `mock` | Demos and contributor onboarding — deterministic synthetic ATEs that match `live` output structure | $0 |
| `live` | You want real intervention-based ATE + bootstrap CI + E-value on the dominant cause | OpenAI billing; cap with `max-cost` |

## Recording the fixtures

You need two `.agentlog` corpora committed: a **baseline** (compliant agent) and a **candidate** (the change under review). The baseline gets recorded once on `main`; the candidate gets re-recorded on the PR branch.

```bash
shadow record -o fixtures/baseline_traces/run-001.agentlog -- python your_agent.py
git add fixtures/baseline_traces/
git commit -m "fixtures: baseline trace corpus for diagnose-pr"
```

`shadow record` auto-instruments the OpenAI / Anthropic SDKs, redacts secrets by default, and writes content-addressed `.agentlog` files. Same flow for Node:

```bash
shadow record -o fixtures/baseline_traces/run-001.agentlog -- node my-agent.js
```

For PR-time recording, run the same command on the candidate branch and write into `fixtures/candidate_traces/`. Two patterns work:

1. **Pre-record both sides locally.** Commit baseline + candidate trace dirs. Workflow just diffs committed files. Deterministic CI; no API keys needed.
2. **Re-record candidate on every push.** Add a recording step in CI that runs your agent against the PR branch's config, then `diagnose-pr` against the freshly written candidate corpus. Needs `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` as a secret.

## A minimal hand-rolled workflow

Skip the composite action when you need to embed `diagnose-pr` inside a larger CI job:

```yaml
- run: pip install --upgrade shadow-diff
- run: |
    shadow diagnose-pr \
      --traces           fixtures/baseline_traces \
      --candidate-traces fixtures/candidate_traces \
      --baseline-config  configs/baseline.yaml \
      --candidate-config configs/candidate.yaml \
      --policy           configs/shadow-policy.yaml \
      --pr-comment       comment.md \
      --out              report.json
- env:
    GH_TOKEN: ${{ github.token }}
  run: gh pr comment "${{ github.event.pull_request.number }}" --body-file comment.md
```

Or use `shadow gate-pr` when you only need the verdict-mapped exit code:

```yaml
- run: |
    shadow gate-pr \
      --traces           fixtures/baseline_traces \
      --candidate-traces fixtures/candidate_traces \
      --baseline-config  configs/baseline.yaml \
      --candidate-config configs/candidate.yaml \
      --policy           configs/shadow-policy.yaml \
      --pr-comment       comment.md
```

`gate-pr` exit codes: `0` ship · `1` hold/probe · `2` stop · `3` internal error.

## What lands on the PR

```
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
Review the prompt change at `prompts/refund.md:17` —
restore the instruction or constraint it removed.

### Verify the fix
shadow verify-fix --report .shadow/diagnose-pr/report.json
```

Verdict-first; metrics second; fix command at the bottom. Reviewers don't need to learn Shadow vocabulary to act on it.

See [`docs/sample-pr-comment.md`](../sample-pr-comment.md) for a full real example.

## Closing the loop with `verify-fix`

After the engineer pushes a fix, re-run with the patched config:

```bash
shadow verify-fix \
  --report       .shadow/diagnose-pr/report.json \
  --traces       fixtures/baseline_traces \
  --fixed-traces fixtures/fixed_candidate_traces \
  --policy       configs/shadow-policy.yaml \
  --out          .shadow/diagnose-pr/verify.json
```

Pass criteria (defaults): `affected_reversed_rate ≥ 0.90`, `safe_trace_regression_rate ≤ 0.02`, `new_policy_violations == 0`. Exit 0 on pass, 1 on fail with explicit `fail_reasons`.

## Legacy: `shadow diff` only

If you don't want causal attribution and just want the nine-axis diff posted on every PR (the pre-3.0 flow), run with `--legacy-diff`:

```bash
shadow init --github-action --legacy-diff
```

This drops the older `shadow-diff.yml` workflow that runs `shadow diff` + `shadow report --format github-pr`. Same trace format, same content-addressing — just no causal layer.

## Next

- [Behavior policy](../features/policy.md) — the twelve rule kinds and `when:` gating that drive `--policy`
- [Causal PR diagnosis](../features/causal-pr-diagnosis.md) — how attribution, verdicts, and the three backends work
- [Nine-axis diff](../features/nine-axis.md) — what each axis actually measures
- [Release certificate](../features/certificate.md) — `shadow certify` as a release gate
