# Shadow quickstart

This is a runnable Shadow scenario you can diff in under 60 seconds — no API keys, no agent code to write. `shadow quickstart` dropped these files into your working directory; everything below works offline against the committed fixtures.

## What's here

| File | What |
|---|---|
| `agent.py` | A toy agent with 3 LLM calls. Uses a canned mock backend so it runs without API keys. |
| `config_a.yaml` | Baseline agent config. |
| `config_b.yaml` | Candidate config — differs on 3 known axes (system prompt, temperature, tool schema). |
| `fixtures/baseline.agentlog` | Baseline trace, pre-recorded. |
| `fixtures/candidate.agentlog` | Candidate trace, pre-recorded. |

## Run the diff

```bash
shadow diff fixtures/baseline.agentlog fixtures/candidate.agentlog
```

You'll see a nine-axis diff table, the first-divergence row, a top-K divergence list, prescriptive fix recommendations, and a per-pair drill-down — all on committed trace data.

## Or run the full diagnose-pr flow

The headline command names the exact change that broke the agent:

```bash
shadow diagnose-pr \
  --traces           fixtures/baseline.agentlog \
  --candidate-traces fixtures/candidate.agentlog \
  --baseline-config  config_a.yaml \
  --candidate-config config_b.yaml \
  --pr-comment       comment.md
```

Open `comment.md` to see the verdict + dominant cause + suggested fix as it would land on a real PR.

## Try it on your own agent

```bash
# Record your agent (any Python script, no code change needed):
shadow record -o my-baseline.agentlog -- python your_agent.py

# Make a change (new prompt, new model, new tool), re-record:
shadow record -o my-candidate.agentlog -- python your_agent.py

# Diff:
shadow diff my-baseline.agentlog my-candidate.agentlog
```

## Learn more

- Full docs: https://github.com/manav8498/Shadow
- Wire diagnose-pr into CI: https://github.com/manav8498/Shadow/blob/main/docs/quickstart/ci.md
- The `.agentlog` spec: https://github.com/manav8498/Shadow/blob/main/SPEC.md
- PyPI: `pip install shadow-diff`
