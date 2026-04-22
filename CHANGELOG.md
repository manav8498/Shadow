# Changelog

All notable changes to Shadow are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased] — v0.1.0 in progress

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
  checkpoint 1. Installed: Rust 1.83.0 + clippy + rustfmt + llvm-tools-preview
  into `~/.cargo` / `~/.rustup` (no sudo, no brew, no system-wide changes).
- **Ecosystem compatibility** — Rust 1.83.0 predates several recent deps
  that require Cargo's `edition2024` feature (stabilized in 1.85). Resolved
  by (a) tightening every direct dep in `crates/shadow-core/Cargo.toml` to
  exact `=x.y.z` pins per the user's "pin to exact versions" constraint,
  and (b) adding `indexmap = "=2.6.0"` as a direct dep so `serde_json`'s
  transitive `indexmap` can't resolve to the edition2024-requiring 2.14.0.
  Cargo.lock committed for reproducibility.
- **PyO3 feature split** — The original single `python` feature used
  `pyo3/extension-module`, which omits libpython link directives. That is
  correct for a maturin-built `.so` but makes `cargo test --features python`
  fail to link. Split into two features: `python` (pyo3 types available,
  libpython linked, `abi3-py311`) and `extension` (adds extension-module,
  used only by maturin). `cargo test --workspace` runs pure-Rust tests
  without pulling pyo3 at all; the PyO3 bindings are tested from Python
  via pytest after `maturin develop`.
- **Companion tool versions** — `just 1.46.0` and `maturin 1.10.2` (installed
  via `cargo install --version`). Latest upstream versions (`just 1.50`,
  `maturin 1.13`) require Rust 1.85 / 1.88 respectively; the older versions
  are the newest compatible with our pinned 1.83 toolchain.

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

### Phase 3+ — not started

---

## Conventions

- Each phase boundary lands a `### Phase N — <title>` section.
- `#### Decisions` for design choices, with the alternative that was
  considered and why it was rejected.
- `#### Dead ends` for approaches that were tried and backed out of, with
  the reason.
- `#### Blockers surfaced` for things the user needs to act on.
