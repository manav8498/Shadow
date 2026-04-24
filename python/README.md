# shadow

**Catch AI-agent regressions before they hit production.**

Shadow is a PR-native diff tool for LLM agents, it records your
agent's calls, replays them under a new config, and tells you what
changed across nine behavioural dimensions.

## Install

```bash
pip install shadow

# With Anthropic support:
pip install 'shadow[anthropic]'

# With OpenAI support:
pip install 'shadow[openai]'

# With both + embeddings:
pip install 'shadow[anthropic,openai,embeddings]'
```

Requires Python 3.11 or newer.

## Quickstart

```python
from shadow.sdk import Session

with Session(output_path="trace.agentlog"):
    # Your existing Anthropic / OpenAI code, unchanged.
    client.messages.create(model="claude-sonnet-4-6", messages=[...])
```

Shadow automatically patches the Anthropic and OpenAI SDKs to capture
every request/response. Secrets are redacted by default.

Then in CI:

```bash
shadow replay new-config.yaml --baseline trace.agentlog
shadow diff trace.agentlog candidate.agentlog
shadow bisect old-config.yaml new-config.yaml --traces trace.agentlog
```

## Full docs

The canonical README, examples, the `.agentlog` spec, and the project
roadmap live at **https://github.com/manav8498/Shadow**.

## License

Dual-licensed under **MIT OR Apache-2.0**. See `LICENSE-MIT` and
`LICENSE-APACHE` in this distribution, or the project repository.
