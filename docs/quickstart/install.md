# Install and first diff

This page walks you through the three commands that take you from
"just heard about Shadow" to "looking at a real behavioural diff."
No API keys, no agent code changes, no CI.

## 1. Install

```bash
pip install shadow-diff
```

- Ships prebuilt wheels for Linux (manylinux_2_34 x86_64), macOS
  (arm64), and Windows (x86_64).
- Requires Python 3.11 or newer.
- Installs the `shadow` CLI script alongside the `shadow` Python
  package. Import and CLI names are `shadow`; PyPI distribution
  name is `shadow-diff` because the bare `shadow` name is taken on
  PyPI by an unrelated 2015 utility.

Verify:

```bash
shadow version
# → shadow 1.x.y (spec 0.1)
```

## 2. Scaffold a working scenario

```bash
shadow quickstart
```

Drops a `shadow-quickstart/` directory containing:

- `agent.py` — a toy agent with three LLM calls
- `config_a.yaml` / `config_b.yaml` — baseline + candidate configs
  that differ on three known axes
- `fixtures/baseline.agentlog` + `fixtures/candidate.agentlog` —
  pre-recorded traces (no API keys needed to run the diff)
- `QUICKSTART.md` — next-step instructions

Use `shadow quickstart path/to/dir` to scaffold elsewhere, or
`--force` to overwrite existing files.

## 3. Run the diff

```bash
cd shadow-quickstart
shadow diff fixtures/baseline.agentlog fixtures/candidate.agentlog
```

You'll see:

- A **nine-axis** table (semantic, trajectory, safety, verbosity,
  latency, cost, reasoning, judge, conformance) with deltas and 95%
  confidence intervals.
- A **low-n banner** if fewer than 5 paired responses.
- A **top-K divergences** list with the Needleman-Wunsch-aligned
  first points of divergence.
- **Recommendations**: prescriptive one-line fixes.
- **Per-pair drill-down**: which specific turn drove the regression.
- A **"What this means"** paragraph in plain English.

## What's next

- Run Shadow on your own agent: [Record your own agent](record.md)
- Wire it into every PR: [Wire into CI](ci.md)
- Understand each axis: [Nine-axis diff](../features/nine-axis.md)
