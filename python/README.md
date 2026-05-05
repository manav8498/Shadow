# shadow-diff

**Find the exact change that broke your AI agent.**

Shadow is a CI-native regression-forensics tool for LLM agents. One command on the PR — `shadow diagnose-pr` — answers:

1. Did agent behavior change?
2. How many traces are affected?
3. **Which exact prompt / model / tool / config change caused it?**
4. With what confidence (ATE + bootstrap CI + E-value when run with `--backend live`)?
5. What fix should `verify-fix` confirm before merge?

The PyPI distribution is `shadow-diff`. The Python import path is `shadow`. The CLI is `shadow`.

## Install

```bash
pip install shadow-diff
```

Requires Python 3.11+. Pre-built wheels ship for Linux x86_64, macOS arm64, and Windows x86_64; other platforms build from source (Rust required).

Optional extras:

```bash
pip install 'shadow-diff[anthropic]'   # if your agent uses Claude
pip install 'shadow-diff[openai]'      # if your agent uses GPT
pip install 'shadow-diff[embeddings]'  # paraphrase-robust semantic diff
pip install 'shadow-diff[all]'         # everything
```

## 60-second tour

```bash
shadow demo                  # nine-axis diff on bundled fixtures, no API key
shadow quickstart            # writable copy of a runnable scenario
```

Then run `diff` against the writable scenario:

```bash
cd shadow-quickstart
shadow diff fixtures/baseline.agentlog fixtures/candidate.agentlog
```

For the full `diagnose-pr` flow against your own agent, see [`docs/features/causal-pr-diagnosis.md`](https://github.com/manav8498/Shadow/blob/main/docs/features/causal-pr-diagnosis.md) and the runnable [`refund-causal-diagnosis`](https://github.com/manav8498/Shadow/tree/main/examples/refund-causal-diagnosis) demo.

## Record your own agent

```python
from shadow.sdk import Session

with Session(output_path="trace.agentlog"):
    # Your existing Anthropic / OpenAI code, unchanged.
    client.messages.create(model="claude-sonnet-4-6", messages=[...])
```

Shadow auto-instruments the Anthropic and OpenAI SDKs and writes content-addressed `.agentlog` files. Secrets are redacted by default. Or skip the code change entirely:

```bash
shadow record -o trace.agentlog -- python your_agent.py
```

## Daily workflow — Shadow as `pytest` for agent behavior

```bash
shadow inspect trace.agentlog                  # debug a single trace
shadow scan baseline_traces/                   # block secret leaks
shadow baseline create baseline_traces/        # pin the gold standard
shadow gate-pr ...                             # gate every PR
```

## Full docs

The canonical README, the `.agentlog` spec, runnable examples, and the comparison against adjacent agent-eval and runtime-governance tools all live at **https://github.com/manav8498/Shadow**.

## License

Apache-2.0. See `LICENSE-APACHE` in this distribution. The `.agentlog` spec is independently published under Apache-2.0.
