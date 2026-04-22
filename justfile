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
    uv run --python .venv/bin/python ruff check python/
    uv run --python .venv/bin/python ruff format --check python/
    uv run --python .venv/bin/python mypy --strict python/src

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
    uv run --python .venv/bin/python ruff format python/

# Regenerate demo fixtures (run after the demo stops producing a baseline).
regen-fixtures:
    uv run --python .venv/bin/python python examples/demo/agent.py --record-to examples/demo/fixtures/baseline.agentlog
