# Example judge configs

Each YAML in this directory is a `--judge-config` payload for
`shadow diff --judge <kind>`. Every file keys into exactly one judge
kind, see the top comment of each for usage.

Run any of them against the devops-agent fixtures like so:

```bash
shadow diff \
  --judge procedure \
  --judge-config examples/judges/procedure_devops.yaml \
  --judge-backend anthropic \
  examples/devops-agent/fixtures/baseline.agentlog \
  examples/devops-agent/fixtures/candidate.agentlog
```

The `mock` backend returns a neutral score per pair (it has no recorded
judge responses); use `anthropic` or `openai` for real judgment.

| File | Judge kind | Target scenario |
|---|---|---|
| `procedure_devops.yaml` | `procedure` | devops-agent's dropped backup-before-migration protocol |
| `schema_customer_support.yaml` | `schema` | customer-support bot structured-output conformance |
| `factuality_acme.yaml` | `factuality` | catching hallucinated Acme policies |
| `refusal_policy.yaml` | `refusal` | over-/under-refusal against a domain-restriction policy |
| `tone_concise.yaml` | `tone` | verbosity drift against a concise-persona target |
| `llm_generic.yaml` | `llm` | templated `LlmJudge` with a three-tier scale |
