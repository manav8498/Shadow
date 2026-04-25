# Wire into CI

The one-command GitHub Actions integration:

```bash
shadow init --github-action
```

Drops `.github/workflows/shadow-diff.yml` into your repo. Edit the
`BASELINE` / `CANDIDATE` paths to point at fixtures you commit, then
push. Every PR gets a nine-axis diff comment.

## What the workflow does

```yaml
name: shadow diff

on:
  pull_request:
    paths:
      - "configs/**"
      - "fixtures/**.agentlog"
      - ".github/workflows/shadow-diff.yml"

permissions:
  pull-requests: write
  contents: read

jobs:
  diff:
    runs-on: ubuntu-latest
    env:
      BASELINE: fixtures/baseline.agentlog
      CANDIDATE: fixtures/candidate.agentlog
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install --upgrade "shadow-diff>=1.2,<2"
      - run: shadow diff "$BASELINE" "$CANDIDATE" --output-json diff.json || true
      - run: shadow report diff.json --format github-pr > comment.md
      - env:
          GH_TOKEN: ${{ github.token }}
        run: gh pr comment "${{ github.event.pull_request.number }}" --body-file comment.md
```

## Gating the merge on regressions

The default workflow posts a comment but never fails the check. To
turn Shadow into a required CI gate, add a separate step that
re-runs `shadow diff` with `--fail-on`:

```yaml
      - name: Block merge on severe regression
        run: shadow diff "$BASELINE" "$CANDIDATE" --fail-on severe
```

The PR comment is already posted by the previous step, so a blocked
PR still has the diff visible. `--fail-on` levels: `minor`,
`moderate`, `severe`. Policy regressions (added with
`--policy <file>`) also count toward the gate. See
[Behavior policy](../features/policy.md) for the policy language.

## Committing fixtures

Your baseline `.agentlog` is committed to the repo. Produce it once
by running your agent through `shadow record`, then commit the result:

```bash
shadow record -o fixtures/baseline.agentlog -- python your_agent.py
git add fixtures/baseline.agentlog
git commit -m "fixtures: baseline trace for nightly PR gate"
```

The candidate is produced on each PR by replaying the baseline through
the PR's modified config. You can either:

1. **Record on every push**, the workflow above replays the baseline
   through the PR branch's config and diffs the result.
2. **Commit both sides**, pre-record both baseline and candidate
   locally and commit both. Workflow just diffs the committed files.
   Good for deterministic CI without requiring API keys.

## Advanced: `--judge auto` in CI

If you set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` as a GitHub
secret, add `--judge auto` to the diff step:

```yaml
- env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: shadow diff "$BASELINE" "$CANDIDATE" --judge auto --output-json diff.json || true
```

`--judge auto` picks a sanity judge against whichever backend key is
set (Anthropic preferred, it's cheaper). Axis 8 turns from an empty
row into a real signal. Budget: ~$0.0003 per diff run.

## Cost-attribution in the PR comment

When the diff JSON has a non-zero cost delta, the markdown renderer
inserts a `## Cost attribution` section automatically. Reviewers see
not just "cost changed" but *why*:

```
| session | baseline | candidate | Δ | model swap | token move | mix |
|--------:|---------:|----------:|--:|-----------:|-----------:|----:|
| #0      | $0.0870  | $0.0174   | $-0.0696 | $-0.0696 (+100%) | $+0.0000 (-0%) | $+0.0000 (-) |
```

## Next

- [Nine-axis diff](../features/nine-axis.md): what each column means
- [Behavior policy](../features/policy.md): the twelve rule kinds and
  conditional `when:` gating that drive `shadow diff --policy`
- [Release certificate](../features/certificate.md): generate an
  Agent Behavior Certificate with `shadow certify` and verify it as
  a release gate
- [Causal bisection](../features/bisect.md): attribute regressions to
  specific config deltas
