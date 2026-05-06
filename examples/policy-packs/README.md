# Policy packs

Ready-to-use Shadow policies for common agent failure modes. Copy
the YAML, edit it for your domain, point your `shadow gate-pr`
invocation at it.

## Available packs

| Pack | What it blocks | Pair with |
|---|---|---|
| [`pii.yaml`](pii.yaml) | Agent responses that emit emails, SSNs, credit cards, bearer tokens, AWS keys, PEM private keys | `shadow scan --secrets` (which catches the same patterns inside committed `.agentlog` files at trace-write time) |
| [`refund-causal-diagnosis/policy.yaml`](../refund-causal-diagnosis/policy.yaml) | Agents that issue refunds without confirming first (the canonical wedge demo) | The full `shadow diagnose-pr` flow (see [`docs/features/causal-pr-diagnosis.md`](../../docs/features/causal-pr-diagnosis.md)) |

## How to use

```bash
shadow gate-pr \
  --traces baseline_traces/ --candidate-traces candidate_traces/ \
  --baseline-config baseline.yaml --candidate-config candidate.yaml \
  --policy examples/policy-packs/pii.yaml
```

A policy violation that's *new* in the candidate trace counts as a
regression and bumps the verdict toward `stop`. A violation that
existed in the baseline and is *cleared* in the candidate counts as
a fix. See [`docs/features/policy.md`](../../docs/features/policy.md)
for the full rule reference and severity-to-exit-code mapping.

## Why two packs, not eight

Shadow's policy DSL is open-ended: every team's domain has its own
rules. We ship two reference packs that prove the DSL works at
opposite ends of the spectrum (data-leak prevention; tool-ordering
safety). The expectation is that you copy these as starting points
and edit them — Shadow isn't a policy library, it's the engine that
runs *your* policies. If your domain needs a third pack, the right
shape is to commit it next to your agent code, not to upstream it
here.

## Customizing

Each rule is a YAML object with three required fields:

```yaml
- id: my-rule          # used in PR comments and reports
  kind: forbidden_text # see docs/features/policy.md for all 12 kinds
  severity: error      # error / moderate / minor — drives --fail-on
  params: { text: "..." }
  description: >       # shows up verbatim in the PR-comment failure block
    why this rule exists; reviewer-readable.
```

Common customizations:

* **Replace the email rule** — by default it matches every `@`.
  For an agent that legitimately sends emails on the user's behalf,
  narrow it to specific domains: `text: "@internal.acme.com"` to
  block leaks of one private domain only.
* **Add company-specific token formats** — drop additional
  `forbidden_text` rules with substrings unique to your tokens
  (e.g. `text: "acme_api_"`).
* **Combine with `shadow scan`** — `pii.yaml` blocks the agent
  *generating* PII in responses; `shadow scan --secrets` blocks
  PII / credentials *making it into the recorded trace files*.
  Both belong in CI.
