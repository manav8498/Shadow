# Real-world walkthrough — Acme Widgets support bot

A concrete scenario showing Shadow catching a real regression before it
merges. Everything here runs offline against committed fixtures.

## The setup

Acme Widgets runs a customer support bot on Claude Opus 4.7. It handles
three kinds of interactions every day:

1. **Refund requests** — "I want to return order #12345"
2. **Order status checks** — "Where's order #67890?"
3. **Structured-data requests** — "Show me my last 3 orders as JSON"

The current production config is [`config_a.yaml`](config_a.yaml). It
has a careful system prompt that requires confirming refund amounts
with the customer before issuing them, and demands JSON-only output
when the customer asks for structured data.

## The PR

Engineer X opens a PR that changes the system prompt "to make the bot
feel more friendly" and renames a tool parameter "for clarity". The
new config is [`config_b.yaml`](config_b.yaml).

At a glance the diff looks small and benign:

```diff
- You are a customer support assistant for Acme Widgets.
-
- When a customer asks about an order, use `lookup_order(order_id)`...
- [detailed refund-confirmation protocol...]
- When the customer asks for structured data, respond with valid JSON only.
- Keep responses under 3 sentences.

+ You are Acme Widgets' customer support assistant. Be helpful and
+ efficient. Use tools when appropriate and be thorough in your
+ explanations so customers feel heard.

- properties: { order_id: { type: string } }
+ properties:
+   id: { type: string }
+   include_shipping: { type: boolean }
```

Three "harmless" tweaks. Two days ago this would have just been
code-reviewed ("LGTM, approved") and shipped.

## Shadow runs on the PR

Shadow replays 3 representative production traces against the candidate
config and posts this PR comment:

```
Shadow diff — 3 response pair(s)
axis          baseline   candidate  delta        95% CI           severity
semantic      1.000      0.698     -0.302    [-0.35, -0.27]       🔴 severe
trajectory    0.000      1.000     +1.000    [+0.00, +1.00]       🔴 severe
safety        0.000      0.000     +0.000    [+0.00,  +0.00]         none
verbosity    42.000    168.000   +126.000    [+78.00, +132.00]    🔴 severe
latency     540.000   1180.000   +640.000    [+550.00, +800.00]   🔴 severe
cost          0.000      0.000     +0.000    [+0.00,   +0.00]        none
reasoning     0.000      0.000     +0.000    [+0.00,   +0.00]        none
judge         0.000      0.000     +0.000    [+0.00,   +0.00]        none
conformance   0.000      0.000     +0.000    [+0.00,   +0.00]        none

worst severity: severe
```

**Four of nine axes flagged severe.**

## What each severe axis actually means

### `semantic` severe — answers are 30% less similar

Baseline answers and candidate answers aren't saying the same things
anymore. Candidate's responses are wandering into a different semantic
space. Example turn 3 ("Show me my last 3 orders as JSON"):

- **Baseline output:** `[{"id":"12345","total":89.99,"status":"delivered"},...]` (pure JSON)
- **Candidate output:** `"Of course! Here are your 3 most recent orders. I've included the order ID, total, and current status for each..."` (prose, no JSON)

The candidate *dropped the JSON-only directive*, so it's now returning
prose when downstream systems expect structured output.

### `trajectory` severe — 100% tool-call divergence

This one is the real bombshell. Inspecting turn 1 (the refund request)
directly:

```
BASELINE turn 1 (refund request) — tool calls:
  → lookup_order({'order_id': '12345'})

CANDIDATE turn 1 (refund request) — tool calls:
  → lookup_order({'id': '12345', 'include_shipping': False})
  → refund_order({'order_id': '12345', 'amount': 89.99})
```

Two separate regressions on the same turn:

1. **The tool schema changed under the model's feet.** Baseline calls
   `lookup_order(order_id='12345')`. Candidate calls
   `lookup_order(id='12345', include_shipping=False)` because the
   schema renamed `order_id` → `id`. If Engineer X's downstream code
   still reads `order_id` from the tool call, it silently breaks.
2. **The candidate issues the refund without confirming with the
   customer first.** Baseline stops after `lookup_order` and would
   return a message asking the customer to confirm. Candidate just
   *does the refund* — because the PR deleted the three-step protocol
   from the system prompt. That's real money leaving Acme's bank
   account without a human saying "yes, refund me."

This is the kind of bug that's invisible in unit tests (the tool
contracts still typecheck!) but catastrophic in production.

### `verbosity` severe — 4× more tokens per response

Baseline median: 42 output tokens. Candidate median: 168. The "be
thorough in your explanations" clause in the new prompt bloated every
response.

### `latency` severe — responses 2× slower

Baseline p50: 540 ms. Candidate p50: 1180 ms. Purely a consequence of
the verbosity increase — more tokens to generate means more wall-clock.

## Bisection

`shadow bisect config_a.yaml config_b.yaml` identifies **7 atomic
deltas** between the two configs:

```
1. prompt.system                              (whole system prompt rewritten)
2. tools[0].description                       (shortened)
3. tools[0].input_schema.properties.id.type   (added)
4. tools[0].input_schema.properties.include_shipping.type (added)
5. tools[0].input_schema.properties.order_id.type (removed)
6. tools[0].input_schema.required[0]          (changed from "order_id" to "id")
7. tools[1].description                       (dropped the "ONLY use after
                                              customer confirmation" sentence)
```

With 7 deltas, Shadow uses an 8-run full-factorial design (2³ would
normally cover 3 factors; here we'd use Plackett-Burman for a larger
k, which stops at 12 runs instead of 128). The LASSO-attribution step
is a v0.1 skeleton — the per-corner scoring wires up to live-replay
in v0.2 — but the delta enumeration itself already lets the reviewer
see *exactly* which lines of the PR are under attribution.

## What the team does with this

They don't merge. They look at the trajectory divergence, realize the
candidate prompt lost the refund-confirmation protocol, and revert that
part of the diff. They keep the `include_shipping` addition. They
re-run Shadow, the diff drops to "minor" across the board, they merge.

Total elapsed: ~30 seconds of CI time saved them from:
- An undetermined number of incorrectly-issued refunds.
- A tool-schema change that silently broke downstream consumers.
- 2× compute spend they weren't budgeted for.

## Limitations this example surfaces (documented for v0.2)

- **`safety` axis didn't flag the unconfirmed-refund bug.** The
  built-in safety axis detects *refusals* ("I can't help with that"),
  not the opposite failure mode of *acting without confirmation*. The
  intended home for "did the bot follow my rubric?" is the `judge`
  axis — but that axis requires a user-supplied Judge protocol
  implementation, which v0.1 doesn't ship a default for.
- **`conformance` axis shows `n=0`** even though candidate failed to
  produce JSON when baseline did. The axis only counts pairs where
  *both* sides have JSON intent (heuristic: text starts with `{` or
  `[`). A "lost JSON intent" detector belongs in v0.2.
- **Bisection attribution weights are 0.** The delta enumeration is
  correct; the per-corner replay-and-score step is placeholder in
  v0.1 (awaits live-replay wiring in v0.2).

The four axes that *did* fire — trajectory, semantic, verbosity,
latency — were enough to block the bad PR.

## Reproduce this locally

```bash
# From the repo root, after `just setup`:
.venv/bin/python examples/customer-support/generate_fixtures.py

.venv/bin/shadow diff \
  examples/customer-support/fixtures/baseline.agentlog \
  examples/customer-support/fixtures/candidate.agentlog \
  --output-json /tmp/cs_report.json

.venv/bin/shadow report /tmp/cs_report.json --format github-pr

.venv/bin/shadow bisect \
  examples/customer-support/config_a.yaml \
  examples/customer-support/config_b.yaml \
  --traces examples/customer-support/fixtures/baseline.agentlog
```

Runtime: ~1 second total, no network.
