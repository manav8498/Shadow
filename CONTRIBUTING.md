# Contributing to Shadow

Thanks for considering a contribution. This file tells you what the workflow
expects. If anything is unclear, open an issue or a draft PR — it's always
better to ask than to guess.

## Ground rules

1. **Small, reviewable PRs.** One logical change per PR.
2. **Conventional Commits** ([conventionalcommits.org](https://www.conventionalcommits.org/))
   — `feat(scope):`, `fix(scope):`, `docs:`, `test:`, `chore:`, `refactor:`,
   `spec:` (for `SPEC.md` changes).
3. **TDD preferred.** Write the failing test first, commit; implement, commit.
   `CLAUDE.md` §Workflow has the full protocol. Exception: typo fixes, docs-only
   changes, and trivial refactors can skip TDD.
4. **No behavior change without a test.** Every bug fix lands with a regression
   test that fails before the fix and passes after.
5. **No `unwrap()` / `expect()` / `panic!()` in non-test Rust.** Enforced by
   clippy lints inside `lib.rs`. Same rule for `# type: ignore` in Python —
   don't use it unless a `# TODO(v0.x):` comment explains why.
6. **Pin every new direct dependency exactly** (`=x.y.z`), update
   `CLAUDE.md` §Dependency policy, and justify in the PR description.

## Setup

```bash
git clone https://github.com/manav8498/Shadow
cd shadow
just setup        # installs user-local rustup if missing, creates .venv,
                  # installs pinned deps, builds the PyO3 extension
```

Requires: a POSIX shell, `curl` (for rustup), Python 3.11+ on PATH.
Windows isn't tested in v0.1.

## The inner loop

```bash
just test          # cargo test + pytest (fast, run on every save)
just lint          # fmt/clippy + ruff/mypy (run before every commit)
just demo          # end-to-end demo runs in ~1s (run before every PR)
just ci            # what CI runs — everything above + coverage gates
```

Before opening a PR, `just ci` must pass locally. CI will re-run the same
matrix across Ubuntu + macOS × Python 3.11 + 3.12.

## Working on each layer

### shadow-core (Rust)

```bash
cargo test -p shadow-core                 # unit tests
cargo test -p shadow-core <filter>        # one module / test
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all
cargo llvm-cov --workspace                # coverage (gate: ≥85% line)
```

### Python SDK / CLI / bisect

```bash
.venv/bin/python -m pytest python/tests       # unit + integration
.venv/bin/python -m ruff check python/
.venv/bin/python -m mypy --strict python/src  # gate: zero errors

# When you change the Rust side that PyO3 exposes, rebuild:
cd python && ../.venv/bin/maturin develop --features extension --release
```

### Demo / examples

```bash
bash examples/demo/demo.sh                    # under 10s, offline
python examples/edge-cases/probe.py           # 20-case adversarial probe
```

## Adding a new diff axis

1. Create `crates/shadow-core/src/diff/<axis>.rs` with a `compute(pairs,
   seed) -> AxisStat` function. Measure something **domain-free** and derivable
   from a pair of `Record`s — if your measurement requires a domain rubric,
   it belongs in the `Judge` axis instead.
2. Add a variant to `Axis` in `diff/axes.rs` and extend `Axis::all()` +
   `Axis::label()`.
3. Wire it into `diff/mod.rs::compute_report`.
4. Ship a test that exercises a realistic regression on this axis and asserts
   severity matches expectations.
5. Update `CLAUDE.md` §Nine axes (now ten) and `SPEC.md` if the axis exposes a
   new field to records.

## The `Judge` axis and domain rules

Shadow deliberately ships no default domain rubric — that's the whole point
of keeping the core axes generic. If your contribution feels like "please add
a detector for X specific thing my domain cares about," the right place is
almost always a `Judge` implementation (Python side), not a new hardcoded
pattern in a core axis. Examples:

- "Detect if a SQL query lacks a WHERE clause before DELETE." → Judge.
- "Score ESI level correctness for clinical triage." → Judge.
- "Flag unconfirmed refund_order tool calls." → Judge.

If we ship a family of example Judges over time (one per domain), they'll live
under `examples/judges/` and be documented clearly as opt-in.

## Writing a `SPEC.md` change

Touching `SPEC.md` is a bigger deal than touching code. Rules:

- Any change that affects the content hash (§5, §6) is a BREAKING change and
  requires a major-version bump (`1.0`). Discuss in an issue first.
- Additive changes (new record kinds, new optional fields) within `0.x.y` are
  fine and just need a SPEC section + a compatibility note in the CHANGELOG.
- Every SPEC change should come with a conformance test in
  `crates/shadow-core/tests/` — if we can't pin it, we can't claim it.

## How reviewers will review your PR

- **Correctness first.** Does it do what it says? Can a regression slip
  through?
- **Test shape.** Is the test asserting the BEHAVIOUR you changed, or is it
  over-specified (brittle) / under-specified (vacuous)?
- **Dead ends.** If you tried an approach that didn't work and backed out of
  it, note it in `CHANGELOG.md` under "Dead ends" so the next contributor
  doesn't hit the same wall.
- **Docstring drift.** If you changed the public surface, your doc comments and
  `CLAUDE.md` should say what the code now does.
- **No new domain hardcoding.** See "The Judge axis" above.

## Releasing (maintainers only)

- Version lives in three places: `Cargo.toml` workspace, `python/pyproject.toml`,
  and `python/src/shadow/__init__.py`. Keep them aligned.
- CHANGELOG.md: move the unreleased section under a new `[x.y.z] — YYYY-MM-DD`
  header.
- Tag with `git tag -a vX.Y.Z -m "..."` and push tags.
- `cargo publish -p shadow-core` (Rust) and `maturin publish` (Python).

## Questions?

Open a [GitHub Discussion](https://github.com/manav8498/Shadow/discussions) or
a draft PR. For security issues, see [`SECURITY.md`](SECURITY.md).
