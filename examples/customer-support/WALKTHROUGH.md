# Real-world walkthrough, Acme Widgets support bot

A concrete scenario showing Shadow catching a real regression before it
merges. Everything here runs offline against committed fixtures.

## The setup

Acme Widgets runs a customer support bot on Claude Opus 4.7. It handles
three kinds of interactions every day:

1. **Refund requests**, "I want to return order #12345"
2. **Order status checks**, "Where's order #67890?"
3. **Structured-data requests**, "Show me my last 3 orders as JSON"

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
Shadow diff, 3 response pair(s)
axis          baseline   candidate  delta        95% CI           severity
semantic      1.000      0.698     -0.302    [-0.35, -0.27]       🔴 severe
trajectory    0.000      1.000     +1.000    [+0.00, +1.00]       🔴 severe
safety        0.000      0.000     +0.000    [+0.00, +0.00]          none
verbosity    42.000    168.000   +126.000    [+78.00, +132.00]    🔴 severe
latency     540.000   1180.000   +640.000    [+550.00, +800.00]   🔴 severe
cost          0.000      0.000     +0.000    [+0.00,  +0.00]         none
reasoning     0.000      0.000     +0.000    [+0.00,  +0.00]         none
judge         0.000      0.000     +0.000    [+0.00,  +0.00]         none
conformance   1.000      0.000     -1.000    [-1.00, -1.00]       🔴 severe

worst severity: severe
```

**Five of nine axes flagged severe.** The four `none` readings are
correct by the axis definitions:

- `safety` measures the model's own refusal / content-filter rate.
  Neither side refused here, so correctly `none`. (The unconfirmed
  refund is a *trajectory* divergence, candidate made a tool call
  baseline didn't, and it's caught on that axis at `+1.000 severe`.)
- `cost` needs a pricing table (not supplied on this run).
- `reasoning`, no thinking tokens in this scenario.
- `judge`, no user-supplied rubric.

## What each severe axis actually means

### `semantic` severe, answers are 30% less similar

Baseline answers and candidate answers aren't saying the same things
anymore. Candidate's responses are wandering into a different semantic
space. Example turn 3 ("Show me my last 3 orders as JSON"):

- **Baseline output:** `[{"id":"12345","total":89.99,"status":"delivered"}...]` (pure JSON)
- **Candidate output:** `"Of course! Here are your 3 most recent orders. I've included the order ID, total, and current status for each..."` (prose, no JSON)

The candidate *dropped the JSON-only directive*, so it's now returning
prose when downstream systems expect structured output.

### `trajectory` severe, 100% tool-call divergence

This one is the real bombshell. Inspecting turn 1 (the refund request)
directly:

```
BASELINE turn 1 (refund request), tool calls:
  → lookup_order({'order_id': '12345'})

CANDIDATE turn 1 (refund request), tool calls:
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
   *does the refund*, because the PR deleted the three-step protocol
   from the system prompt. That's real money leaving Acme's bank
   account without a human saying "yes, refund me."

This is the kind of bug that's invisible in unit tests (the tool
contracts still typecheck!) but catastrophic in production.

### `verbosity` severe, 4× more tokens per response

Baseline median: 42 output tokens. Candidate median: 168. The "be
thorough in your explanations" clause in the new prompt bloated every
response.

### `latency` severe, responses 2× slower

Baseline p50: 540 ms. Candidate p50: 1180 ms. Purely a consequence of
the verbosity increase, more tokens to generate means more wall-clock.

## Bisection

`shadow bisect config_a.yaml config_b.yaml --traces fixtures/baseline.agentlog
--candidate-traces fixtures/candidate.agentlog` identifies 7 atomic
deltas and allocates each axis's divergence across the deltas that could
plausibly have caused it:

```
axis          attribution
-----------   -----------
semantic      100% prompt.system
trajectory    17% each across the six tools[0].* deltas
safety        14% each across prompt.system + six tool deltas
verbosity     100% prompt.system
latency       14% each across prompt.system + six tool deltas (downstream)
conformance   100% prompt.system
```

Reading this: **the prompt edit drove the semantic, verbosity, and
conformance regressions on its own** (those axes can't be explained by
the tool schema change, so they land entirely on the prompt).
**The trajectory regression is purely tool-schema** (it can't be
explained by prompt edits). **Safety and latency are shared**.
either cause is plausible for both.

What the reviewer learns: revert the prompt edit and you recover three
severe axes (semantic, verbosity, conformance) and at least share of
safety/latency. Keep the tool-schema change only if you ALSO fix the
safety contract that used to live in the tool description.

The allocator is a heuristic (docs: `shadow.bisect.runner.DELTA_KIND_AFFECTS`):
prompt deltas map to generation-level axes, tool-schema deltas map to
trajectory/safety, params map to verbosity/reasoning, model swaps
touch everything. A live-LLM-driven LASSO scorer over every corner of
the Plackett-Burman design is v0.2, this allocator is what works
today with just two recorded traces.

## What the team does with this

They don't merge. They look at the trajectory divergence, realize the
candidate prompt lost the refund-confirmation protocol, and revert that
part of the diff. They keep the `include_shipping` addition. They
re-run Shadow, the diff drops to "minor" across the board, they merge.

Total elapsed: ~30 seconds of CI time saved them from:
- An undetermined number of incorrectly-issued refunds.
- A tool-schema change that silently broke downstream consumers.
- 2× compute spend they weren't budgeted for.

## Honest limitations remaining in v0.1

The axes are all principled and domain-free, no hardcoded tool
prefixes for "risky" or "safe," no assumptions about customer-support
specifically. The trajectory axis catches tool-call divergence in any
domain; the conformance axis catches structural-output regressions in
any domain; etc.

Remaining scope:

- **`judge` axis ships as a trait without a default implementation.**
  Teams that want to express a domain rubric, "assistant MUST ask for
  confirmation before calling `refund_order`"; "ESI-1 cardiac
  presentations MUST page physician", plug in their own `Judge`. The
  Protocol is defined and stable. Shipping a default LLM-judge was out
  of scope for v0.1 because calling an LLM-judge from Rust is a
  Python-layer concern, and defaulting to any particular judge would
  be the domain hardcoding we're specifically avoiding.
- **Bisect attribution is a principled allocator, not causal
  inference.** It maps each delta-kind to the axes it can plausibly
  affect based on the axes' own definitions (e.g. `tools.*` can affect
  trajectory / verbosity / latency / cost but not safety, because
  safety measures the model's own refusal rate). Within a category
  (e.g. "prompt.system changed"), all content candidates get equal
  weight. Teasing apart *which sentence* of a prompt drove a specific
  axis needs the live-LLM LASSO-over-corners scorer scoped for v0.2.
- **`cost` requires a user-supplied pricing table.** Not a bug, the
  default-zero behaviour is intentional (we don't ship assumptions
  about what models cost your team). Pass `--pricing pricing.json` to
  activate.

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
  --traces examples/customer-support/fixtures/baseline.agentlog \
  --candidate-traces examples/customer-support/fixtures/candidate.agentlog
```

Runtime: ~1 second total, no network.
