# Shadow quickstart

This directory is exactly what `shadow quickstart` drops into your
working directory, a minimal, runnable Shadow scenario you can
diff in under 60 seconds without writing any code.

## What's here

| File | What |
|---|---|
| `agent.py` | A toy agent with 3 LLM calls. Uses a canned mock backend so it runs without API keys. |
| `config_a.yaml` | Baseline agent config. |
| `config_b.yaml` | Candidate config, differs on 3 known axes (system prompt, temperature, tool schema). |
| `fixtures/baseline.agentlog` | Baseline trace, pre-recorded. |
| `fixtures/candidate.agentlog` | Candidate trace, pre-recorded. |

## Run the diff (no API keys required)

```bash
shadow diff fixtures/baseline.agentlog fixtures/candidate.agentlog
```

You'll see a nine-axis diff table, a first-divergence row, a
top-K divergence list, prescriptive fix recommendations, and a
per-pair drill-down, all on committed trace data.

## Try it on your own agent

```bash
# Record your agent (any Python script; no code change needed):
shadow record -o my-baseline.agentlog -- python your_agent.py

# Make a change (new prompt, new model, new tool), re-record:
shadow record -o my-candidate.agentlog -- python your_agent.py

# Diff:
shadow diff my-baseline.agentlog my-candidate.agentlog
```

## Learn more

- Full docs: https://github.com/manav8498/Shadow
- Spec: https://github.com/manav8498/Shadow/blob/main/SPEC.md
- PyPI: `pip install shadow-diff`
