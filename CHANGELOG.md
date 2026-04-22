# Changelog

All notable changes to Shadow are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

- **Live LLM backends.** `shadow.llm.AnthropicLLM` (wraps
  `anthropic.AsyncAnthropic`) and `shadow.llm.OpenAILLM` (wraps
  `openai.AsyncOpenAI`). Both implement the `LlmBackend` Protocol; both
  lazy-import their SDK so `shadow` still runs without the extras.
  `shadow.llm.get_backend(name, **kwargs)` factory dispatches by name.
- **LASSO-over-corners bisection scorer**
  (`shadow.bisect.corner_scorer`). Given a live `LlmBackend`, builds a
  2^k full-factorial over the differing config categories
  (`{model, prompt, params, tools}`), replays the baseline through the
  backend at each corner, computes the nine-axis divergence per corner,
  and fits LASSO per axis. Returns per-axis attribution weights that
  sum to 1 across the active categories. Ground-truth test recovers
  `latency → model` and `verbosity → prompt` with > 70 % weight.
- **CLI `shadow bisect --backend {anthropic,openai,positional}`** wires
  the live-replay scorer through the CLI. Without `--backend`, falls
  back to the heuristic kind-based allocator when `--candidate-traces`
  is supplied, or zero-placeholder otherwise.
- **`run_bisect` three-mode dispatch**:
  `lasso_over_corners` (best, with backend) → `heuristic_kind_allocator`
  (when only a candidate trace is available) → `lasso_placeholder_zero`
  (neither). The `mode` field in the output names which ran.
- **11 new tests** covering the backends (fake SDK stubs installed in
  `sys.modules` — no network) and the corner scorer (deterministic
  `FakeBackend` whose response metrics depend on which categories are
  active, letting LASSO recover ground-truth attributions).

### Changed

- README Limitations section updated — bisection is no longer
  described as "heuristic-only in v0.1". The heuristic remains as a
  no-credentials fallback, with the live-backend LASSO scorer as the
  primary path.

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
  Python layer per CLAUDE.md D5. This kept an ML dep out of the Rust
  crate.
- **Axis nodes take `pairs: &[(&Record, &Record)]`.** Pair extraction
  lives in `diff/mod.rs::extract_response_pairs`. Uneven counts (e.g.
  candidate missed some) truncate to the shorter side; callers that
  want to flag count-mismatch consult the replay_summary directly.
- **Severity thresholds (CLAUDE.md §4):** `None` (CI crosses zero and
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
