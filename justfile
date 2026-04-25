# Shadow task runner. `just --list` shows everything.
# All recipes assume you're at the repo root.

# Default recipe just shows the menu.
default:
    @just --list

# One-shot bootstrap for a fresh clone: installs llvm-tools-preview, cargo-llvm-cov,
# creates the .venv, installs the Python package in editable mode, and builds the
# PyO3 extension with maturin. Idempotent.
setup:
    rustup component add llvm-tools-preview || true
    cargo install --locked cargo-llvm-cov || true
    uv venv --python 3.11
    uv pip install --python .venv/bin/python -e "python[dev]"
    cd python && ../.venv/bin/maturin develop --release

# Full test suite: Rust + Python.
test: test-rust test-python

test-rust:
    # No feature flags: pure-Rust tests only.
    #   * `--features extension` would enable pyo3/extension-module, which
    #     omits the libpython link directives → test binary fails to link.
    #   * `--features python` pulls pyo3 with abi3-py311 (no extension-module),
    #     whose build.rs insists on a system Python ≥ 3.11 — not a given on
    #     dev machines where the default `python3` is older.
    # The PyO3 bindings are tested from Python via `just test-python` after
    # `maturin develop` builds them. This keeps the Rust test loop zero-config.
    cargo test --workspace

test-python:
    uv run --python .venv/bin/python pytest python/tests -v

# Lint everything: fmt check + clippy (-D warnings) + ruff + mypy --strict.
lint: lint-rust lint-python

lint-rust:
    cargo fmt --all -- --check
    # clippy doesn't link, so --all-features is safe here and gives wider
    # coverage than --features python (picks up the `extension` feature's
    # code paths too).
    cargo clippy --workspace --all-targets --all-features -- -D warnings

lint-python:
    # Scope must match .github/workflows/ci.yml exactly. CI lints
    # `python/ examples/` and type-checks the demo entry points; if
    # this recipe drifts narrower, broken pushes slip past `just ci`.
    uv run --python .venv/bin/python ruff check python/ examples/
    uv run --python .venv/bin/python ruff format --check python/ examples/
    uv run --python .venv/bin/python mypy --config-file python/pyproject.toml --strict python/src examples/demo/agent.py examples/demo/generate_fixtures.py

# Run the end-to-end demo (uses MockLLM, must complete <10s).
demo:
    bash examples/demo/demo.sh

# What CI runs. Must pass locally before pushing.
ci: lint test
    # Coverage on the same feature set as tests (no pyo3 features — bindings
    # are tested from Python after maturin develop builds the extension).
    cargo llvm-cov --workspace --fail-under-lines 85
    uv run --python .venv/bin/python pytest python/tests --cov=shadow --cov-fail-under=85
    just demo

fmt:
    cargo fmt --all
    uv run --python .venv/bin/python ruff format python/ examples/

# Mirror the GitHub Actions matrix locally: the exact command set from
# .github/workflows/ci.yml run in the same order. Use this before
# pushing if `just ci` was green but you suspect a drift between local
# and CI environments. Catches the three classes of failure that have
# bitten previous releases:
#   1. ruff/mypy scope (CI lints examples/, local sometimes didn't)
#   2. optional-extras gating (mcp/serve modules need extras installed)
#   3. macOS clippy + windows shell quirks (still local-only here, but
#      this recipe at least flushes Linux-equivalent breakage early)
ci-local:
    @echo "==> rust: fmt + clippy + test + coverage"
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo test --workspace
    cargo llvm-cov --workspace --fail-under-lines 85 --summary-only
    @echo "==> python: ruff + mypy (CI scope)"
    .venv/bin/python -m ruff check python/ examples/
    .venv/bin/python -m ruff format --check python/ examples/
    .venv/bin/python -m mypy --config-file python/pyproject.toml --strict python/src examples/demo/agent.py examples/demo/generate_fixtures.py
    @echo "==> python: pytest with coverage gate"
    .venv/bin/python -m pytest python/tests --cov=shadow --cov-config=python/pyproject.toml --cov-fail-under=85 --cov-report=term-missing
    @echo "==> demo: end-to-end <10s"
    # GNU `timeout` is `gtimeout` on macOS (coreutils). CI uses Linux
    # where it's just `timeout`. Pick whichever is on PATH; if neither,
    # run unguarded — the demo targets <10s and a runaway is rare.
    bash -c 'if command -v timeout >/dev/null 2>&1; then timeout 30 bash examples/demo/demo.sh; elif command -v gtimeout >/dev/null 2>&1; then gtimeout 30 bash examples/demo/demo.sh; else bash examples/demo/demo.sh; fi'
    @just ci-local-extras
    @echo "==> ci-local: ALL GREEN"

# Replicate the `python-full-extras` CI job: install every optional
# extra, then run the entire pytest suite with no --ignore filter.
# Catches optional-extras gating bugs (e.g. an mcp test that imports at
# collection time and crashes a fresh-extras install). Idempotent —
# extras add to whatever's already in .venv.
ci-local-extras:
    @echo "==> ci-local-extras: install every optional extra"
    uv pip install --python .venv/bin/python -e "python[dev,anthropic,openai,otel,serve,mcp]"
    # `embeddings` pulls sentence-transformers (~500 MB with torch);
    # split into its own step so a flake here doesn't mask earlier
    # failures and the output stays legible.
    uv pip install --python .venv/bin/python -e "python[embeddings]"
    @echo "==> ci-local-extras: pytest with all extras (no --ignore)"
    .venv/bin/python -m pytest python/tests -v --ignore=python/tests/test_judge_live.py

# Regenerate demo fixtures (run after the demo stops producing a baseline).
regen-fixtures:
    uv run --python .venv/bin/python python examples/demo/agent.py --record-to examples/demo/fixtures/baseline.agentlog
