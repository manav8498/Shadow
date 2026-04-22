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

_(in progress)_

### Phase 2+ — not started

---

## Conventions

- Each phase boundary lands a `### Phase N — <title>` section.
- `#### Decisions` for design choices, with the alternative that was
  considered and why it was rejected.
- `#### Dead ends` for approaches that were tried and backed out of, with
  the reason.
- `#### Blockers surfaced` for things the user needs to act on.
