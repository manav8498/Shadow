# Shadow

**Find the exact change that broke your AI agent.**

Your teammate opens a PR that tweaks the system prompt, swaps GPT-4o for a cheaper model, or adjusts a tool schema. Code review looks fine. Unit tests pass. You merge. A week later a customer reports the refund agent started issuing refunds without confirming. The PR landed days ago. Nobody saw it coming because the code looked harmless.

That's the bug class Shadow exists to catch. `shadow diagnose-pr` answers — in one PR comment — which exact prompt, model, tool-schema, or config change caused the regression, with bootstrap CI + E-value when run with `--backend live`. Then `shadow verify-fix` closes the loop by re-running only the affected traces against your fix.

## Sixty-second adoption

```bash
pip install shadow-diff
shadow quickstart
shadow diff shadow-quickstart/fixtures/baseline.agentlog \
            shadow-quickstart/fixtures/candidate.agentlog
```

That's a real nine-axis diff on pre-recorded `.agentlog` fixtures. No API keys, no agent code. See [Install and first diff](quickstart/install.md) for the full walkthrough, or jump to [Wire into CI](quickstart/ci.md) to see Shadow comment on every PR in ten minutes.

## What Shadow does, in one screen

1. **What changed?** A nine-axis diff (semantic, trajectory, safety, verbosity, latency, cost, reasoning, judge, format) with bootstrap 95% CIs and severity bands.
2. **Why?** Causal attribution names the specific config delta most likely responsible for each regression. Pearl-style ATE + back-door adjustment + E-value sensitivity (opt-in) live alongside the stable LASSO bisection.
3. **Is it safe to merge?** A YAML policy declares behavioral rules (tool ordering, output shape, forbidden text). The same policy enforces at PR time *and* at runtime — a runtime override can't ship something CI rejected.

The report lands in the PR comment. No dashboard, no separate login, no trace upload. Traces stay on your disk.

## Highlights

- **Nine axes**: semantic, trajectory, safety, verbosity, latency, cost, reasoning, LLM-judge, format conformance.
- **Ten built-in judges**, including a configurable `LlmJudge`. Verified against live Claude Haiku 4.5 and GPT-4o-mini.
- **Causal bisection** with pairwise interactions and bootstrap 95% CIs on attribution weights.
- **Session-cost attribution** that decomposes cost delta into model-swap, token-movement, and mix-residual.
- **Six-level hierarchical diff**: trace, session, turn, span, token, policy.
- **Eight importers** to bridge from existing observability tooling: Langfuse, Braintrust, LangSmith, OpenAI Evals, OTLP, MCP, Vercel AI SDK, PydanticAI.
- **Partial replay**: lock a baseline prefix, replay only the suffix through a live backend.
- **Zero-config instrumentation**: `shadow record -- python agent.py` (or `node agent.js`). No code changes.
- **Cross-OS**: Ubuntu, macOS, and Windows on Python 3.11, 3.12, and 3.13.

## Where Shadow fits

Shadow is a CI/repo-native tool. **It does not replace your LLM observability platform — it complements one.** If you want a hosted dashboard for your traces, use whichever platform you already have. If you want behavior changes blocked in your PR before they merge — and the same rule enforced at runtime — that's what Shadow ships as a single command. See the [comparison matrix](comparison.md) for an honest read against EvalView, Microsoft AGT, Preloop, AgentEvals, and Speedscale.

## Where to next

- New to Shadow? [Install and first diff](quickstart/install.md).
- Understanding the output? [Nine-axis diff](features/nine-axis.md).
- Wiring Shadow into CI? [Wire into CI](quickstart/ci.md).
- Looking for a specific feature? See the feature index in the sidebar.

## Links

- **PyPI**: [shadow-diff](https://pypi.org/project/shadow-diff/)
- **Source**: [manav8498/Shadow](https://github.com/manav8498/Shadow)
- **Spec**: [.agentlog v0.1](https://github.com/manav8498/Shadow/blob/main/SPEC.md)
- **License**: Apache-2.0
