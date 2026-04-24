# Changelog

All notable changes to Shadow are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

## [0.3.0] - 2026-04-24

### Added — zero-friction adoption

A full week on the one thing that was harder than it should have been:
making Shadow trivially usable for a new user who just ran
`pip install shadow-diff`.

- **Auto-instrumentation via PYTHONPATH + sitecustomize.**
  `shadow record -- python your_agent.py` now records an agent's
  LLM calls **with zero code changes** to the agent itself. A shim
  `sitecustomize.py` prepended to the child's PYTHONPATH fires at
  interpreter startup, checks for `SHADOW_SESSION_OUTPUT`, and if
  set, constructs and enters a `Session` whose atexit handler
  flushes the trace on process shutdown. All existing Session
  features (anthropic/openai monkey-patching, redaction, trace
  propagation) apply automatically. New `--tags` flag on
  `shadow record` propagates to the metadata record. New
  `--no-auto-instrument` opts out cleanly if the agent already
  manages its own Session.

- **`shadow quickstart` scaffolder.** Drops a working demo
  (`agent.py`, `config_a.yaml`, `config_b.yaml`, two pre-recorded
  `.agentlog` fixtures, `QUICKSTART.md`) into a directory in one
  command. A brand-new user goes from `pip install shadow-diff` to
  seeing a real nine-axis diff in under 60 seconds, no API keys
  required. `--force` overwrites existing files; otherwise existing
  content is preserved.

- **`shadow init --github-action`.** Extends the existing `init`
  command with a flag that drops a ready-to-commit
  `.github/workflows/shadow-diff.yml` into the user's repo. Wires
  up `pip install shadow-diff`, runs `shadow diff` on every PR,
  renders the report as PR-comment markdown, and posts via the
  `gh` CLI. Edit two env vars (BASELINE / CANDIDATE paths),
  commit, and every PR gets a behavioural-diff comment with no
  further setup.

- **README rewrite.** The "Try it" section is now "5-minute
  adoption" and leads with the three-command adoption path:
  `pip install shadow-diff` → `shadow quickstart` → `shadow diff`.
  The "Instrument your own agent" section now documents the
  zero-config `shadow record` path as the recommended option, with
  the explicit `Session` pattern as the secondary choice.

### Fixed

- **Typer extra-args forwarding on `shadow record`.** Typer 0.24
  (bumped for the `--help` fix in 0.2.x) changed how the `--`
  separator is handled at the command level; `shadow record`
  needed its own `context_settings={"allow_extra_args": True,
  "ignore_unknown_options": True}` for child args after `--` to
  be forwarded to `ctx.args`. Previously only set at the app
  level, which no longer sufficed.

### Tests

- 10 new `test_autostart.py` tests covering env-var handling,
  tag parsing, empty-command rejection, exit-code propagation,
  `--no-auto-instrument` behaviour, and the PYTHONPATH shim contents.
- 12 new `test_quickstart.py` tests covering file scaffolding, valid
  agentlog output, `shadow diff` on the scaffolded fixtures,
  `--force` behaviour, `--github-action` workflow generation, and
  the composed quickstart + init flow.
- Hero harness extended from 34 to **47 end-to-end assertions**
  covering the new adoption path on committed fixtures.

## [0.2.2] - 2026-04-23

### Changed

- **PyPI distribution name renamed from `shadow` to `shadow-diff`.**
  The short `shadow` name was already registered on PyPI by an
  unrelated 2015 btrfs-snapshot utility, so the project is now
  published as `pip install shadow-diff`. The Python import path
  (`import shadow`), the installed CLI command (`shadow`), and the
  GitHub repo slug (`manav8498/Shadow`) are unchanged — only the
  PyPI distribution name differs.

### Fixed

- **Release pipeline** — `cargo cyclonedx --output-pattern package`
  rejected the `--output-pattern` flag on `cargo-cyclonedx` 0.5.7
  (upstream removed it). Dropped the flag and rely on the default
  per-crate output path; the schema-valid minimal-SBOM fallback
  still catches any future upstream path drift.

## [0.2.1] - 2026-04-23

### Fixed

- **Release pipeline** — `cargo package -p shadow-core` failed in
  the v0.2.0 release run because the crate's `include` allowlist
  matched only `src/**/*.rs`; `src/store/schema.sql`, which is read
  via `include_str!()`, was excluded and the verify-build inside the
  published tarball broke. Added `src/**/*.sql` to the allowlist.
- **Release pipeline** — the Python SBOM step wrote to `dist/` at
  the repo root (which didn't exist) while wheels landed in
  `python/dist/`; redirected SBOM output to match.

### Added

- **PyPI publish job** (`publish-pypi` in `release.yml`) using OIDC
  Trusted Publisher — no API token required. Bound to a `pypi`
  GitHub Environment so the trust link is (repo, workflow,
  environment)-scoped. See `docs/PYPI-PUBLISHING.md` for the
  one-time setup (pypi.org pending publisher + GitHub Environment
  creation).

## [0.2.0] - 2026-04-23

### Fixed

- **`shadow <cmd> --help` now renders**. Bumped `typer` from pinned
  `0.13.0` to `>=0.15,<1.0`; the old pin was incompatible with
  `click 8.2+` (breaking `TyperArgument.make_metavar()` signature).
  All twelve subcommands print their help pages again.

### Added

- **Live-LLM judge tests** — new `python/tests/test_judge_live.py`
  exercises every judge against real Anthropic and OpenAI backends.
  Gated by `SHADOW_RUN_NETWORK_TESTS=1` plus `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY`; auto-skips otherwise. Each test picks a scenario
  where the correct verdict is unambiguous so a real LLM's behaviour
  can be asserted directly. ~$0.01 token budget per backend per full
  run.

- **Scale benchmark for drill-down** — new
  `benchmarks/scale_drill_down.py` runs the full nine-axis diff
  plus drill-down ranking on synthetic traces at `N ∈ {100, 500,
  1000}`. Asserts correctness (ranking invariants, dominant axis,
  `pair_count` match) and a 60-second wall-time cap at N=1000.
  Current numbers on a local laptop: 0.18 s @ N=100, 4.5 s @ N=500,
  18.1 s @ N=1000.

- **Every optional extra now exercised in CI.** New `serve` and
  `otel` extras in `pyproject.toml`, plus a new CI job
  `python-full-extras` that installs `dev`, `anthropic`, `openai`,
  `otel`, `serve`, and `embeddings` and runs the full test suite —
  no more silently-skipped tests. Test count went from 260 passed /
  18 skipped to 278 passed / 0 skipped (excluding the network-gated
  `test_judge_live.py` module, which correctly stays skipped without
  API keys).

- **Per-pair drill-down** — `DiffReport` gains a `drill_down` field
  ranking the top-K most-regressive response pairs in a trace set
  by an aggregate regression score, with a per-axis breakdown for
  each. Surfaces *which* specific turns drove each aggregate axis
  delta, so a reviewer scanning a PR with many paired traces can
  click-in to the single worst pair instead of hand-auditing them
  all.

  Each row carries `pair_index`, `baseline_turn`, `candidate_turn`,
  `regression_score`, `dominant_axis`, and an `axis_scores` list of
  8 `PairAxisScore` entries (Judge excluded — the Rust core never
  populates it). Per-axis `normalized_delta` is `|delta| /
  axis_scale` clamped to `[0, 4]`; scales are calibrated against the
  existing `Severity::Severe` thresholds so a value of 1.0
  corresponds to one severity-severe-sized movement on that axis
  and scores sum coherently across axes.

  On the real devops-agent fixture: top pair is pair #1 with
  regression score 9.32 — verbosity collapsed (402→128 output
  tokens) and trajectory flipped (baseline had 0 tool-divergence,
  candidate had 100%). A reviewer spots the regression without
  opening any raw `.agentlog` bytes.

  Rendered by every report format: terminal (one line per pair with
  the top 2 contributing axes indented), markdown / github-pr
  (`### Top regressive pairs` section with the top 3 inline and the
  rest under a collapsible `<details>` block).

  Tests: 8 new Rust unit tests in `diff::drill_down::tests` + 7
  Python renderer/integration tests. Hero harness extended to
  assert `drill_down` surfaces ≥ 3 regressive pairs on the real
  devops-agent scenario (now 34/34 end-to-end assertions).

  Closes the last open v0.2 ROADMAP item ("Per-pair drill-down in
  the diff report").

- **Hero end-to-end real-world scenario.** New harness
  `benchmarks/hero_devops_scenario.py` that exercises every Shadow
  feature (schema-watch, nine-axis diff, first-divergence,
  top-K divergences, recommendations, hardened bisection, and all
  ten judges) end-to-end against **committed `.agentlog` trace
  fixtures** — not hand-crafted verdicts. Inputs are the real YAML
  configs and real recorded agent traces under
  `examples/devops-agent/` and `examples/customer-support/`. Every
  feature must independently surface a symptom of the PR's regression
  for the run to pass: 30/30 assertions currently green across two
  domains. This answers the integration question the per-feature
  `validate_*.py` harnesses can't: do the features compose on real
  trace data?

- **Judge axis defaults — 10 ready-made judges.** Previously axis 8
  was effectively a no-op without a user-written rubric. The package
  now ships six new Judge classes on top of the existing four
  (`SanityJudge`, `PairwiseJudge`, `CorrectnessJudge`, `FormatJudge`):

  - **`LlmJudge`** — generic, user-configurable LLM-as-judge. Caller
    supplies a rubric string (may reference `{task}`, `{baseline}`,
    `{candidate}`) plus a `score_map` of verdict strings to
    `[0, 1]` scores. Construction-time validation rejects unknown
    placeholders so mistakes surface before the first judge call.
    Defaults (`temperature=0`, `max_tokens=512`) match published
    LLM-as-judge best practice (Zheng et al. 2024).
  - **`ProcedureAdherenceJudge`** — flags candidates that skip steps
    from a required procedure. Catches the devops-agent pattern
    where a prompt rewrite silently drops
    `backup_database → run_migration` ordering.
  - **`SchemaConformanceJudge`** — semantic schema review (shape +
    meaning). Complements `FormatJudge`'s mechanical JSON-schema
    validation.
  - **`FactualityJudge`** — flags candidates whose claims contradict
    a known-fact set.
  - **`RefusalAppropriateJudge`** — catches over- AND under-refusals
    against an explicit policy.
  - **`ToneJudge`** — tone / persona drift against a target.

  Shared judge helpers consolidated in `shadow.judge._common`
  (response-text extraction, JSON-object extraction from prose, NaN-
  safe clamping, uniform error verdicts) so breaking changes to
  judge-response parsing now land in one file.

  CLI: `shadow diff --judge <kind>` extended from `none|sanity` to
  nine values (`none`, `sanity`, `pairwise`, `llm`, `procedure`,
  `schema`, `factuality`, `refusal`, `tone`). New
  `--judge-config <file.yaml>` option loads rubric data for the
  domain judges. Six example config templates shipped under
  `examples/judges/` covering the devops-agent procedure, the
  customer-support schema, Acme factuality, a domain-restriction
  refusal policy, a concise-persona tone target, and a generic
  three-tier `LlmJudge`.

  Tests: 23 new ground-truth unit tests in `test_llm_judge.py`
  (placeholder validation, custom score maps, error paths,
  out-of-range confidence clamping, each of the five domain
  judges). Real-world validation harness
  (`benchmarks/alignment/validate_judges.py`) passes 14/14
  assertions exercising the full rubric → render → parse → score
  pipeline via a deterministic backend.

- **`shadow schema-watch`** — proactive tool-schema change detection
  that runs *before* replay. Classifies each change between two
  configs into one of four severity tiers (breaking / risky /
  additive / neutral) across eleven change kinds (tool added/removed,
  param added/removed/renamed, type changed, required flipped,
  enum narrowed/broadened, description edited). Rename detection
  pairs a removed and added param on the same tool by matching type
  and required-status, catching the most common breaking tool-schema
  edit in practice (see the `devops-agent` example where every tool
  silently renames `database` to `db`).

  Exposed as a new CLI command with three output formats — terminal
  (rich markup), markdown (GitHub table + expandable rationale), and
  json — and a `--fail-on` flag that controls exit-code behaviour in
  CI. Example output on the committed `customer-support` fixture:

  ```
  ✖ BREAKING  lookup_order: parameter renamed `order_id` → `id`
  ! RISKY     refund_order: description rewritten — imperative verbs removed
  + ADDITIVE  lookup_order: parameter `include_shipping` added (optional)
  · NEUTRAL   lookup_order: description rewritten
  ```

  Intended to run first in CI so PRs get a fast schema-breakage
  signal before the full nine-axis diff runs. 24 ground-truth unit
  tests plus a real-world validation harness
  (`benchmarks/alignment/validate_schema_watch.py`) that passes
  14/14 assertions against the `devops-agent` (8-tool database→db
  rename) and `customer-support` (order_id→id rename + optional
  param + risky description edit) fixtures.

- **Hardened causal bisection** — new
  `shadow.bisect.attribution.rank_attributions_with_interactions`.
  Fits **pairwise interaction effects** (delta A x delta B) in addition
  to main effects, and emits honest **bootstrap 95% confidence
  intervals** on every attribution weight. Example output on a
  realistic 4-delta config PR:

  ```
  semantic:
    prompt.system            74.9% [71.0%, 89.2%]  sel_freq=1.00  ✓
    model_id x prompt.system 13.8% [3.8%, 17.5%]   sel_freq=0.89  ✓

  latency:
    model_id          61.3% [59.2%, 68.0%]  sel_freq=1.00  ✓
    tools             19.7% [15.3%, 22.4%]  sel_freq=0.94  ✓
    model_id x tools  16.6% [12.0%, 19.4%]  sel_freq=0.96  ✓
  ```

  Implementation follows the research brief's explicit guidance:
  `PolynomialFeatures(interaction_only=True)` for the augmented
  design; **residual bootstrap** (Chatterjee & Lahiri 2011) instead
  of pairs bootstrap to avoid LASSO's point-mass-at-zero pathology;
  `alpha` fixed via outer `LassoCV` once on the original data (not
  re-tuned per resample — Efron 2014 shows that inflates CI width);
  per-resample normalisation **before** percentile (not normalise-
  then-bootstrap, which breaks CI independence); and a strong-
  hierarchy post-filter that drops A x B interactions where neither
  main effect survived stability selection (Lim & Hastie 2015
  *glinternet*).

  The `significant` flag is now a conjunction — selection frequency
  ≥ 0.6 AND CI excludes zero — which is honestly framed as
  screening + magnitude, not a multiplicity-adjusted p-value. Lex-
  sorted interaction pair labels eliminate the `(A,B)` vs `(B,A)`
  ambiguity. Output available via `run_bisect(... backend=...)`
  under the new `attributions_with_interactions` key. 7 new Rust/
  Python ground-truth tests plus a real-world validation harness
  (8/8 passing) in `benchmarks/alignment/validate_hardened_bisect.py`.

- **Prescriptive fix recommendations** (`shadow.diff.recommendations`).
  Transforms the divergence list from "what changed" into "what to do
  about it": a ranked list of specific, imperative actions a PR
  reviewer can act on in under 30 seconds. Examples from the real-
  world 10-turn scenario:

  - `[error]   RESTORE`  turn 9 — *"Restore `send_confirmation_email`
    at turn 9."*
  - `[error]   REMOVE`   turn 8 — *"Remove duplicate invocation of
    `lookup_order(order_id)` at turn 8."*
  - `[error]   REVIEW`   turn 4 — *"Review refusal behaviour at
    turn 4: candidate may be over-refusing."*
  - `[warning] REVERT`   turn 6 — *"Revert `refund(amount)` at turn 6
    to the baseline value."*

  Three-tier severity (Error / Warning / Info) matching ESLint /
  SonarQube / Rustc conventions. Five action kinds (Restore / Remove
  / Revert / Review / Verify) covering the expected regression
  patterns. Pure deterministic rule engine — no LLM dependency;
  LLM-enriched recommendations can layer on later.

  Each recommendation carries `severity`, `action`, `turn`, a one-line
  `message`, a `rationale` line citing the signal that triggered it,
  and the primary `axis`. Sorted by severity × confidence, capped at
  8 entries so PR comments don't bloat.

  Exposed on `DiffReport` as the new `recommendations` field
  (documented by the `Recommendation` TypedDict in `_core.pyi`).
  Rendered in all three report formats: markdown / github-pr
  (bulleted `### Recommendations` section with severity icons),
  terminal (colour-coded by severity). 14 new Rust unit tests + 8
  new Python renderer tests; real-world benchmark harness passes
  18/18 including action-correctness assertions.

- **Top-K divergence ranking** on top of first-divergence detection.
  The diff report now carries a `divergences` list (up to
  `DEFAULT_K=5` entries) in addition to the backward-compatible
  `first_divergence` field. Divergences are sorted by importance:
  Structural > Decision > Style (by class), then by confidence
  within a class, with walk order as the stable tiebreaker.
  Renderers show the top 3 inline and collapse any extras (#4 and
  beyond) into a `<details>` section in markdown / github-pr, or a
  "+N more" line in terminal. `first_divergence` remains walk-order-
  first for back-compat; `divergences[0]` is severity-rank-first.
  Validated end-to-end with the benchmark harness at 27/27 cases
  (real committed fixtures + adversarial stress + top-K-specific
  coverage).
- **First-divergence detection** (`shadow.diff.alignment`). Given two
  traces, identifies the first turn at which the candidate diverged
  from the baseline and classifies the divergence as
  `style_drift` / `decision_drift` / `structural_drift`. Uses a
  Needleman-Wunsch global alignment with Gotoh affine gap penalties
  over the chat-response sequence; per-cell cost combines Jaccard
  distance on tool-shape, character-shingle text similarity, arg-
  value diff, and stop-reason mismatch. Surfaces in every report
  renderer (terminal / markdown / github-pr) as a one-line root-
  cause summary like
  *"tool set changed: removed `search(query)`, added `search(limit,query)`"*.
  Exposed on the `DiffReport` dict as the new `first_divergence` key
  (documented by the `FirstDivergence` TypedDict in `_core.pyi`);
  `None` when the traces agree end-to-end. 14 new Rust tests + 6 new
  Python renderer tests.



- Live LLM backends: `shadow.llm.AnthropicLLM` wraps
  `anthropic.AsyncAnthropic`, `shadow.llm.OpenAILLM` wraps
  `openai.AsyncOpenAI`. Both implement the `LlmBackend` Protocol and
  lazy-import their SDK so `shadow` still runs without the extras.
  `shadow.llm.get_backend(name, **kwargs)` factory dispatches by name.
- LASSO-over-corners bisection scorer (`shadow.bisect.corner_scorer`).
  Given a live `LlmBackend`, builds a 2^k full-factorial over the
  differing config categories (`{model, prompt, params, tools}`),
  replays the baseline through the backend at each corner, computes
  the nine-axis divergence per corner, and fits LASSO per axis.
  Ground-truth test recovers `latency → model` and `verbosity → prompt`
  with > 70% weight.
- CLI: `shadow bisect --backend {anthropic,openai,positional}` wires
  the live-replay scorer. Without `--backend`, falls back to the
  heuristic kind-based allocator when `--candidate-traces` is supplied,
  or zero-placeholder otherwise.
- `run_bisect` three-mode dispatch: `lasso_over_corners` (with backend)
  → `heuristic_kind_allocator` (only a candidate trace) →
  `lasso_placeholder_zero` (neither). The `mode` field names which ran.
- 11 new tests covering the backends (fake SDK stubs, no network) and
  the corner scorer (deterministic `FakeBackend`).
- OSS governance / community files: `SUPPORT.md`, `GOVERNANCE.md`,
  `MAINTAINERS.md`, `TRADEMARK.md`, `CITATION.cff`,
  `.github/FUNDING.yml`, `.github/dependabot.yml`.
- `AxisStat` and `DiffReport` `TypedDict`s in `shadow/_core.pyi` so
  downstream Python consumers get real types for the Rust-extension
  return shapes.

### Changed

- Dual-license the implementation under **MIT OR Apache-2.0** (Rust
  community default). `SPEC.md` stays Apache-2.0 only so re-implementers
  get an explicit patent grant for the format.
- README rewritten in plain English (~200 lines), one quickstart,
  sample output matches real `just demo` output.
- Project metadata on `Cargo.toml` and `pyproject.toml`:
  keywords, categories, URLs, `Typing :: Typed`, explicit `include`
  allowlist for the Rust crate.
- `.gitignore` hardened: added `node_modules/`, JS tool caches,
  `.env.*.local` variants, agent-state dirs, `*.tsbuildinfo`,
  Jupyter checkpoints, SBOM outputs.
- `shadow/__init__.py` wraps the abi3-mismatch ImportError with a
  "requires Python 3.11+" hint. `AnthropicLLM.__init__` fails fast
  with a branded error if no API key is set, instead of deferring to
  the opaque HTTP-layer error from the SDK.
- Removed the `CLAUDE.md` internal coding-agent instruction file from
  the repository (not appropriate in a public OSS project).

### Fixed

- Severity classifier false negative on rate-bounded axes: a CI like
  `[0.0, 1.0]` was treated as straddling zero and downgraded Severe to
  Minor. Now `ci_straddles_zero` is strict (`ci_low < -epsilon &&
  ci_high > +epsilon`). A unanimous `+1.0` trajectory delta now
  correctly classifies as Severe.
- Conformance axis was dead for tool-use-only agents (`n=0`). It now
  also fires on `tool_use`-intent and scores by top-level key-set match.
- Delta extractor exploded a single tool-schema edit into dozens of
  leaf-level deltas, which over-determined the LASSO fit. Default
  `diff_configs(coalesce=True)` collapses each tool to a single delta
  keyed by tool name. Legacy leaf-level output still available via
  `coalesce=False`.
- Attribution row schema was inconsistent across bisect modes. All
  three modes now emit the same six keys (`delta`, `weight`,
  `ci95_low`, `ci95_high`, `significant`, `selection_frequency`).
- `DELTA_KIND_AFFECTS["tools"]` was missing `conformance` — added, so
  tool-schema edits can be attributed to the conformance axis.
- 11 pre-existing `ruff` lints (all pattern/idiom nits such as
  `class X(str, Enum)` → `class X(StrEnum)` for Python 3.11+).

### Notes

- This release has **no production users yet**. All "real-world"
  validation references the project's own example scenarios, not
  external deployments. Claims about behaviour should be read as
  "what the code does," not "what teams have confirmed in prod."

## [0.1.0] — 2026-04-22

First tagged release. Ships the Rust core, Python SDK + CLI, bisection
module, GitHub Action, end-to-end demo, and CI — see the per-phase
sections below for specifics.

### Summary

- 13 commits across 7 phases.
- **164 tests** total: 125 Rust (`cargo test -p shadow-core`) + 47 Python
  (`pytest python/tests`).
- **Rust coverage: 97.63% line / 98.93% function** on `shadow-core`.
- **Python coverage: 88.07%** across `shadow.*` packages.
- `cargo clippy --all-targets --all-features -- -D warnings` clean.
- `cargo fmt --check` clean.
- `mypy --strict` clean across 26 Python source files.
- `ruff check` + `ruff format --check` clean.
- `bash examples/demo/demo.sh` runs in **1.14 s** on an M-series laptop
  (target ≤ 10 s).



### Phase 0 — Scaffold

#### Decisions

- **PyO3 build system: maturin** (D1 in the plan). Using `abi3-py311` so one wheel
  supports Python 3.11+. Alternative considered: setuptools-rust — rejected
  because maturin's `develop` loop is the same-day ergonomics we want for TDD.
- **Workspace-level Rust pinned to 1.83.0** (stable at planning time). Individual
  dep versions pinned exactly in `Cargo.toml` per the user's "no bleeding-edge
  churn" constraint.
- **Clippy `unwrap_used`/`expect_used`/`panic` denied in non-test code** via
  inner attributes in `lib.rs` (not workspace-wide) so tests can still use
  `unwrap()`.
- **Python deps pinned to exact versions** in `python/pyproject.toml`. Real LLM
  SDKs (`anthropic`, `openai`) are optional extras — Shadow can run without
  them via `MockLLM`.
- **Cargo `crate-type = ["cdylib", "rlib"]`** on shadow-core so the same crate
  builds both as a normal Rust library (for `cargo test`) and as a PyO3
  extension (for `maturin develop`). The `python` feature gates the PyO3
  module; non-Python consumers get a clean rlib.
- **CLI entrypoint: `shadow = "shadow.cli.app:main"`** (Python `console_script`),
  not a Rust binary. This keeps the CLI logic testable with `typer.testing.CliRunner`
  and lets the Rust core remain pure-library.

#### Dead ends

- _(none yet — Phase 0 ran cleanly through file scaffolding)_

#### Blockers surfaced and resolved

- **Rust toolchain install** — `cargo` / `rustc` were absent; user-local
  rustup curl install authorized post-Phase-1 per `<interaction_policy>`
  checkpoint 1, installed into `~/.cargo` / `~/.rustup` (no sudo, no brew,
  no system-wide changes).
- **Rust toolchain version** — initially pinned to `1.83.0` (stable at
  original-plan time, late 2024); the release drop to `v0.1.0` then
  bumped to **Rust 1.95.0** (stable 2026-04-14) because the ecosystem had
  already moved past 1.83 — `indexmap 2.14+`, `proptest 1.11`, latest
  `just` / `maturin` all require edition2024 (Rust ≥ 1.85). The interim
  1.83 workaround (direct-pin `indexmap = "=2.6.0"` so `serde_json`'s
  transitive dep couldn't resolve to edition2024) was removed on the
  bump; Cargo.lock regenerated; all direct deps updated to the versions
  Cargo picks cleanly on 1.95.
- **Companion tool versions** — `just 1.50.0`, `maturin 1.13.1`,
  `cargo-llvm-cov 0.8.5`. All installed via `cargo install --locked`.
- **PyO3 feature split** — The original single `python` feature used
  `pyo3/extension-module`, which omits libpython link directives. That is
  correct for a maturin-built `.so` but makes `cargo test --features python`
  fail to link. Split into two features: `python` (pyo3 types available,
  libpython linked, `abi3-py311`) and `extension` (adds extension-module,
  used only by maturin). `cargo test --workspace` runs pure-Rust tests
  without pulling pyo3 at all; the PyO3 bindings are tested from Python
  via pytest after `maturin develop`.
- **Rust 1.95 clippy tightening** — the toolchain bump surfaced three new
  lints: `doc_overindented_list_items` (fixed by de-indenting continuation
  lines in the replay engine's doc comment), `useless_conversion` firing
  on PyO3's idiomatic `?`-in-`PyResult` patterns (addressed with a
  module-level `#![allow]` in `src/python.rs` with a comment explaining
  why), and a `clone`→`slice::from_ref` suggestion in a parser test.

### Phase 1 — SPEC.md

#### Decisions

- **Content-address payload only, not the envelope.** `id = sha256(canonical_json(payload))`
  so two identical requests dedupe to the same blob. Envelope (`ts`, `parent`) is
  not hashed. Alternative considered: hash the whole envelope — rejected because
  it defeats dedup and makes MockLLM replay lookups harder (you'd need to reconstruct
  the envelope to look up a response).
- **RFC 8785 (JCS) for canonical JSON**, with two application clarifications
  (§5.2 — Unicode NFC normalization on strings and keys; §5.4 — no
  `Decimal`/`NaN`/`Infinity`). Picking an existing RFC instead of inventing
  our own rules means any JCS library is most of the way there; the NFC
  addition covers a gap in JCS where visually-identical strings encoded
  differently would hash differently.
- **Known-vector lives in §5.6 as a "Conformance test case"** (moved on
  review from §6.2, which now points back to §5.6). The vector covers both
  canonicalization bytes and the resulting content id, so a fresh
  implementer can verify both at once.
- **Known-vector hash pinned in §6.2:** `{"hello":"world"}` →
  `sha256:93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588`.
  Verified locally with `python3 -c 'import hashlib; print(hashlib.sha256(b"{\"hello\":\"world\"}").hexdigest())'`.
  Phase 2's `agentlog::hash` test suite pins this vector.
- **One trace = one file.** Concatenating two `.agentlog` files does NOT
  produce a valid `.agentlog` — simpler invariant than allowing multi-trace
  files, and forces the SQLite index to be the "set" abstraction.
- **First record is always `metadata` with `parent: null`.** A trace is
  identified by its root's content id, so we don't need a separate trace_id
  field in the envelope.
- **Redaction before canonicalization.** The hash reflects redacted content,
  not raw. Means a post-hoc audit can't trivially reconstruct the original
  — that's intentional.
- **Streaming responses = one record** with an optional `stream_timings`
  array, not a record per token. Per-token records would explode storage
  and make `shadow diff` much slower. Timing preserved, token stream
  content aggregated.

#### Dead ends

- _(none yet — spec came together in one pass)_

### Phase 2 — shadow-core (Rust)

Nine commits land the full Rust core. Final tree: 125 unit tests, `cargo
clippy --all-features -- -D warnings` clean, `cargo fmt --check` clean,
**97.63% line coverage** on `shadow-core` (target ≥85%) measured via
`cargo llvm-cov --workspace` (98.93% function coverage).

#### Decisions

- **Payload-only hashing (SPEC §6.1).** Record envelope (`ts`, `parent`)
  is not in the hash; only `canonical_json(payload)`. Two identical
  requests dedupe to the same id.
- **Records hold payloads as `serde_json::Value`, not typed structs.**
  Typed payload structs come later alongside the consumers that need
  them (Python SDK). Keeps the core storage path provider-agnostic.
- **Atomic writes for the FS store** via write-tmp + rename. A crash
  mid-write leaves a `.tmp` orphan, not a corrupt real trace.
- **`include_str!` SQLite schema.** Schema lives in `store/schema.sql`;
  bundled into the binary so there's no runtime file resolution and the
  schema can be versioned alongside the code that consumes it.
- **`PyO3` feature split: `python` vs `extension`.** The base `python`
  feature pulls pyo3 with `abi3-py311` but NOT `extension-module`, so
  `cargo test --features python` links libpython normally. `extension`
  adds `extension-module` for maturin. This avoids the "cargo test
  link-fails because libpython isn't linked" footgun.
- **`cargo test --workspace` runs NO features by default.** PyO3
  bindings are tested from Python after `maturin develop`. Keeps the
  Rust test loop zero-config (no need for python3.11 on PATH).
- **Backend trait takes `&Value`, returns `Value`.** Envelope ownership
  stays with the engine; backends are tiny adapters (a live Anthropic
  backend is ~30 lines). Matches SPEC §10's replay algorithm cleanly.
- **Replay errors → `Error` records, not engine panics.** A baseline
  with some-missing responses still produces a complete candidate
  trace, with error counts in the `replay_summary`.
- **Clock abstraction** in `replay::engine` so tests can pin
  timestamps. Ships a `FixedClock` for fixtures; production uses the
  system clock (Phase 3).
- **Semantic axis: hash-surrogate embedding in Rust.** Clearly labelled
  test-only in the module docstring. The production embedding
  (`sentence-transformers/all-MiniLM-L6-v2`) is plugged in by the
  Python layer per CONTRIBUTING.md. This kept an ML dep out of the Rust
  crate.
- **Axis nodes take `pairs: &[(&Record, &Record)]`.** Pair extraction
  lives in `diff/mod.rs::extract_response_pairs`. Uneven counts (e.g.
  candidate missed some) truncate to the shorter side; callers that
  want to flag count-mismatch consult the replay_summary directly.
- **Severity thresholds:** `None` (CI crosses zero and
  delta is tiny), `Minor` (<10% relative), `Moderate` (<30%), `Severe`
  (≥30% or CI clearly excludes zero with large magnitude).
- **Bootstrap defaults to 1000 iterations** with seeded RNG for
  reproducibility; callers can override both.
- **Test-only `assert_eq!` / `unwrap()`** allowed; clippy's
  unwrap_used/panic lints are denied only in non-test code via
  `#![cfg_attr(not(test), ...)]`.

#### Dead ends

- Attempted to write `assert_eq!` in `paired_ci` for length-mismatch
  precondition — tripped clippy's `panic` lint which applies to
  `panic!()` expansions of `assert!` macros. Resolved with a narrow
  `#[allow(clippy::panic)]` block around an explicit `panic!()` —
  cleaner than restructuring to return a Result for a programmer-error
  guard.
- First pass at the `meta` omit-when-None test used `!wire.contains("meta")`,
  which accidentally matched the kind string `"metadata"`. Fix: match the
  quoted field name `"\"meta\""` instead. Small lesson: substring
  assertions on JSON are fragile; prefer explicit field-level checks.

### Phase 3 — Python SDK + CLI

#### Decisions

- **PyO3 bindings take/return dicts, not typed pyclass wrappers.** Simpler
  surface for Python users and no second type system to maintain. The
  serde_json::Value ↔ PyObject conversion goes through the `pythonize`
  crate (pinned `=0.22.0` to match `pyo3=0.22.6`).
- **Type stubs shipped in `python/src/shadow/_core.pyi`.** mypy --strict
  users see the PyO3 surface without importing the compiled extension.
- **Session is a manual recorder in v0.1.** Monkey-patch-based
  auto-instrumentation of `anthropic` / `openai` Python clients is
  deferred to v0.2 — their streaming surfaces are too divergent to
  unify cleanly in a first cut, and forcing users into
  `record_chat(req, resp)` is a small enough overhead that it's not
  blocking adoption.
- **Python-side `run_replay`.** The Rust replay engine exists but isn't
  exposed through PyO3 — the `LlmBackend` trait is async and calling
  back into Python from a Rust trait object needs PyO3 ceremony that
  wasn't worth the code for v0.1. The Python replay mirrors SPEC §10
  semantics exactly.
- **CLI uses typer.** Every subcommand has an end-to-end integration
  test via `typer.testing.CliRunner`. Machine-consumed JSON outputs go
  through `sys.stdout.write` (unstyled) to avoid Rich's ANSI escapes
  breaking `jq` pipelines.
- **`PositionalMockLLM` added** alongside `MockLLM`. Positional replay
  is the only sensible demo backend when baseline and reference traces
  were recorded with different configs (different request payloads →
  different content ids → MockLLM strict would miss every request).
  Clearly labelled as "for demos and integration tests, not production."

#### Dead ends

- First pass at `cargo test --features python` failed to link on
  macOS because the `extension-module` pyo3 feature omits libpython
  link directives. Resolved by splitting the feature into `python`
  (abi3-py311 only, links libpython) and `extension` (adds
  extension-module, used by maturin). `cargo test` takes neither by
  default — PyO3 bindings are tested from Python after `maturin
  develop` builds the `.so`.
- Initial pass at the `meta` omission test used a substring check
  against `"meta"` which accidentally matched `"metadata"` (the kind
  name). Fix: check for the quoted field name `"\"meta\""`.

### Phase 4 — Bisection (LASSO + Plackett-Burman)

#### Decisions

- **Per-axis LASSO with `alpha=0.01`.** scikit-learn handles the coord
  descent. Normalization: `|coef| / sum(|coef|)` so each axis's
  attributions sum to 1 (or 0 when the axis is invariant across
  corners).
- **Hadamard/Paley PB matrices tabulated** for runs ∈ {8, 12, 16, 20,
  24}. Runs > 24 error out in v0.1 (k ≤ 23). Full factorial capped at
  k=6 (64 runs). `choose_design` picks the right one automatically.
- **v0.1 runner emits placeholder-zero divergence.** Real per-corner
  replay scoring lands in v0.2 with live-LLM support. The plumbing
  (delta extraction, design, LASSO, attribution ranking) is correct
  and tested against a synthetic ground-truth recovery case
  (≥0.9 attribution to delta #2 on the trajectory axis; ≤0.05 on
  every other delta; zero attribution on every other axis since there
  is no signal).

### Phase 5 — GitHub Action

#### Decisions

- **Composite action, not a JavaScript action.** No Node build
  pipeline; the logic lives in `action.yml` shell steps plus a
  stdlib-only `comment.py`.
- **Hidden HTML marker** lets subsequent runs update the existing PR
  comment in place. One comment per PR, not a running log.
- **Step-summary write** means even fork PRs (where posting a comment
  is blocked) still surface the diff in the GitHub Actions log view.

### Phase 6 — Demo + README

#### Decisions

- **Demo fixtures are committed.** `examples/demo/fixtures/{baseline,
  candidate}.agentlog` are deterministic outputs of
  `generate_fixtures.py` (also committed). A fresh clone runs
  `just demo` in ≈1 s without touching a network. Regenerating the
  fixtures is reproducible.
- **`PositionalMockLLM` is the demo backend.** `MockLLM` (content-id
  lookup) wouldn't work because baseline/candidate differ in
  system-prompt wording — their request payloads have different ids.
- **README opens with "Why?"** (per review), then a four-column
  competitive-landscape table (Langfuse / Braintrust / LangSmith /
  Shadow). Gives readers a reason to keep scrolling before they hit
  install instructions.

### Phase 7 — CI + release

#### Decisions

- **Matrix: Ubuntu + macOS × Python 3.11 + 3.12.** Windows deferred
  (SDK is POSIX-path biased; `.venv/bin/python` assumptions are
  Windows-unfriendly without more plumbing than the v0.1 budget
  allows).
- **`taiki-e/install-action` for `cargo-llvm-cov`** so CI doesn't pay
  the ~45 s compile-from-source cost on every run.
- **Rust cache via `Swatinem/rust-cache`** shaves ~5 min off matrix
  legs.
- **Python tests gate at 85% coverage** (`--cov-fail-under=85`).
  Current: 88.07%.
- **Three jobs**: `rust` (fmt/clippy/test/coverage), `python` (ruff,
  mypy, pytest+coverage), `demo` (end-to-end). `python` depends on
  `rust`, `demo` on `python`.



---

## Conventions

- Each phase boundary lands a `### Phase N — <title>` section.
- `#### Decisions` for design choices, with the alternative that was
  considered and why it was rejected.
- `#### Dead ends` for approaches that were tried and backed out of, with
  the reason.
- `#### Blockers surfaced` for things the user needs to act on.
