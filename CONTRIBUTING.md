# Contributing to Shadow

Thanks for considering a contribution. This file tells you what the workflow
expects. If anything is unclear, open an issue or a draft PR, it's always
better to ask than to guess.

## Ground rules

1. **Small, reviewable PRs.** One logical change per PR.
2. **Conventional Commits** ([conventionalcommits.org](https://www.conventionalcommits.org/)) — **required** for release automation.
   The supported types are `feat(scope):`, `fix(scope):`, `perf:`, `docs:`, `test:`, `refactor:`, `build:`, `ci:`, `chore:`, `revert:`,
   plus the project-specific `spec:` for `SPEC.md` changes.
   Release-please reads commit types between releases to compute the
   next version: `feat:` → minor, `fix:` / `perf:` → patch, any with
   `!` after the type or a `BREAKING CHANGE:` footer → major.
   Commits without a recognised type still land but don't trigger a
   bump and don't appear in the auto-generated CHANGELOG section.
3. **TDD preferred.** Write the failing test first, commit; implement, commit.
   `CONTRIBUTING.md` §Workflow has the full protocol. Exception: typo fixes, docs-only
   changes, and trivial refactors can skip TDD.
4. **No behavior change without a test.** Every bug fix lands with a regression
   test that fails before the fix and passes after.
5. **No `unwrap()` / `expect()` / `panic!()` in non-test Rust.** Enforced by
   clippy lints inside `lib.rs`. Same rule for `# type: ignore` in Python -
   don't use it unless a `# TODO(v0.x):` comment explains why.
6. **Dependency policy.** Pin **dev / build / test** dependencies exactly
   (`=x.y.z`) so CI reruns are bit-reproducible. **Runtime** dependencies
   should use reviewed lower + upper bounds (e.g. `numpy>=2.2,<3`) so
   downstream applications can resolve compatible versions of their own.
   Update `CONTRIBUTING.md` §Dependency policy and justify in the PR
   description for any new direct dep.

## Setup

```bash
git clone https://github.com/manav8498/Shadow
cd shadow
just setup        # installs user-local rustup if missing, creates.venv,
                  # installs pinned deps, builds the PyO3 extension
```

Requires: a POSIX shell, `curl` (for rustup), Python 3.11+ on PATH.
Windows is covered in CI for the Python wheel and Rust core; some
shell-based helper scripts (e.g. parts of the `justfile`) still
assume POSIX, so day-to-day development on Windows is best done
under WSL2.

## The inner loop

```bash
just test          # cargo test + pytest (fast, run on every save)
just lint          # fmt/clippy + ruff/mypy (run before every commit)
just demo          # end-to-end demo runs in ~1s (run before every PR)
just ci            # what CI runs, everything above + coverage gates
```

Before opening a PR, `just ci` must pass locally. CI will re-run the same
matrix across Ubuntu + macOS × Windows × Python 3.11 + 3.12 + 3.13.

### Pre-commit hooks (recommended)

A `.pre-commit-config.yaml` is checked into the repo. It runs the same
ruff lint, ruff format, basic hygiene, and DCO sign-off checks that CI
enforces — locally, on every `git commit`. Install once:

```bash
pip install pre-commit            # or: pipx install pre-commit
pre-commit install                # installs the git hook in .git/hooks/
pre-commit install --hook-type commit-msg   # for the DCO sign-off check
```

After that, every commit auto-runs the hooks and blocks if any fail.
To run on the whole repo manually:

```bash
pre-commit run --all-files
```

mypy is intentionally not in pre-commit (too slow for the inner loop);
use `just lint` to get full strict mypy locally before pushing.

## Working on each layer

### shadow-diff core crate (Rust, source dir `crates/shadow-core/`)

```bash
cargo test -p shadow-diff                 # unit tests
cargo test -p shadow-diff <filter>        # one module / test
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all
cargo llvm-cov --workspace                # coverage (gate: ≥85% line)
```

### Python SDK / CLI / bisect

```bash.venv/bin/python -m pytest python/tests       # unit + integration.venv/bin/python -m ruff check python/
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
   from a pair of `Record`s, if your measurement requires a domain rubric,
   it belongs in the `Judge` axis instead.
2. Add a variant to `Axis` in `diff/axes.rs` and extend `Axis::all()` +
   `Axis::label()`.
3. Wire it into `diff/mod.rs::compute_report`.
4. Ship a test that exercises a realistic regression on this axis and asserts
   severity matches expectations.
5. Update `README.md` "The nine behavioral dimensions" section (rename to match
   the new count) and `SPEC.md` if the axis exposes a new field to records.

## The `Judge` axis and domain rules

Shadow deliberately ships no default domain rubric, that's the whole point
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
  `crates/shadow-core/tests/`, if we can't pin it, we can't claim it.

## How reviewers will review your PR

- **Correctness first.** Does it do what it says? Can a regression slip
  through?
- **Test shape.** Is the test asserting the BEHAVIOUR you changed, or is it
  over-specified (brittle) / under-specified (vacuous)?
- **Dead ends.** If you tried an approach that didn't work and backed out of
  it, note it in `CHANGELOG.md` under "Dead ends" so the next contributor
  doesn't hit the same wall.
- **Docstring drift.** If you changed the public surface, your doc comments and
  `CONTRIBUTING.md` should say what the code now does.
- **No new domain hardcoding.** See "The Judge axis" above.

## Releasing (maintainers only)

### API stability: SemVer commitment

Shadow follows [Semantic Versioning 2.0.0](https://semver.org/) starting
at v2.5.0. The public API in v2.x is stable: **no breaking changes
within a major version**. A future v3.0.0 may rework public API; until
then, anything documented in the public surface (CLI commands listed
in `shadow --help`, importable symbols in `shadow.<module>`,
`.agentlog` format §3-§4 of `SPEC.md`, ABOM certificate fields)
remains backward-compatible.

What "breaking" means in practice:

- Renaming or removing a CLI command, flag, or environment variable
- Removing a class/function/method from a public module
- Changing a function signature in a way that breaks existing callers
- Removing a record `kind`, payload field, or changing its semantics
- Removing a certificate field or changing its type
- Tightening a previously-permissive validator in a way that rejects
  existing valid inputs

What is NOT breaking and may change in any patch release:

- Internal modules prefixed with `_` (e.g. `shadow._core`, `shadow._telemetry`)
- Adding new optional parameters with defaults
- Adding new CLI flags or commands
- Adding new record kinds or optional payload fields
- Performance characteristics
- Error messages (format, wording)
- Deprecated APIs (these emit `DeprecationWarning` for at least one
  minor version before removal in the next major)

For format-level changes, see `SPEC.md` §13 (Versioning and forward/back
compatibility).



Shadow ships three artifacts that share a single version number:

- Python wheel (`shadow-diff` on PyPI) — `python/pyproject.toml` + `python/src/shadow/__init__.py`
- Rust crate (`shadow-diff` on crates.io, source in `crates/shadow-core/`) — `Cargo.toml` workspace
- TypeScript SDK (`shadow-diff` on npm) — `typescript/package.json`

**All three bump together on every release.** A v2.5.0 release means the
Python wheel, Rust crate, and TypeScript SDK are all at 2.5.0, even if a
specific component had no functional changes that release. This is the
Cargo workspace / Kubernetes release-train model: predictable for users,
no-confusion when filing issues.

The release pipeline (`.github/workflows/release.yml`) checks all three
versions match before tagging. CI fails if they drift.

### Release flow

The full release flow — including one-time secret setup, manual
override paths, troubleshooting known failures, and how to yank a
broken release — lives in [RELEASE.md](RELEASE.md). Short version:
land Conventional Commit-formatted PRs on `main`, merge the
long-running `chore: release X.Y.Z` PR that release-please
maintains, and the publish chain fires automatically.

## Developer Certificate of Origin (DCO)

Every commit to Shadow must carry a Developer Certificate of Origin sign-off.
This is a lightweight, no-paperwork way for contributors to certify that they
wrote the patch (or otherwise have the right to submit it under the project's
Apache 2.0 license). The full DCO text is at [developercertificate.org](https://developercertificate.org/).

By adding a `Signed-off-by` line to your commit message, you certify that:

> 1. The contribution was created in whole or in part by you and you have
>    the right to submit it under the open source license indicated; or
> 2. The contribution is based upon previous work that, to the best of your
>    knowledge, is covered under an appropriate open source license and you
>    have the right to submit that work with modifications under the same
>    open source license; or
> 3. The contribution was provided directly to you by some other person who
>    certified (1), (2), or (3) and you have not modified it.
> 4. You understand and agree that this project and the contribution are
>    public and that a record of the contribution (including all personal
>    information you submit with it) is maintained indefinitely.

### How to sign off

Use the `-s` (or `--signoff`) flag every time you commit:

```bash
git commit -s -m "feat(diff): add semantic similarity axis"
```

This appends a line like the following to your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

Configure git once and forget about it:

```bash
git config user.name  "Your Name"
git config user.email "your.email@example.com"
```

If you forget to sign off, amend the last commit:

```bash
git commit --amend --signoff
```

For a chain of unsigned commits on a feature branch, rebase and re-sign:

```bash
git rebase --signoff main
```

Pull requests with unsigned commits will fail the DCO check in CI and cannot
be merged until every commit carries a `Signed-off-by` line that matches the
commit's author.

## Questions?

Open a [GitHub Discussion](https://github.com/manav8498/Shadow/discussions) or
a draft PR. For security issues, see [`SECURITY.md`](SECURITY.md).
