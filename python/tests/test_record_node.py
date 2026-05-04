"""Tests for `shadow record -- node ...` auto-instrumentation.

Mirrors `test_autostart.py` for Python agents — verifies the
Node-runtime detection logic and that `NODE_OPTIONS` is
injected when the wrapped command is a Node-family CLI.

End-to-end recording (actually running a Node agent) is
tested in `typescript/test/auto.test.ts`; this file focuses
on the Python-side wiring decisions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from shadow.cli.app import _is_node_runtime

# ---- _is_node_runtime() classifier ---------------------------------------


@pytest.mark.parametrize(
    "executable",
    [
        "node",
        "/usr/local/bin/node",
        "/opt/homebrew/bin/node",
        "node.exe",
        "C:\\Program Files\\nodejs\\node.exe",
        "nodejs",  # debian/ubuntu
        "node18",  # version-suffixed symlink
        "node20",
        "node22",
        "npx",
        "npx.cmd",
        "tsx",
        "ts-node",
        "ts-node.cmd",
    ],
)
def test_is_node_runtime_true_for_node_family(executable: str) -> None:
    """All Node-family CLIs that respect NODE_OPTIONS must be detected."""
    assert _is_node_runtime(executable), f"{executable!r} should be a Node runtime"


@pytest.mark.parametrize(
    "executable",
    [
        "python",
        "python3",
        "python3.11",
        "/usr/bin/python",
        "bash",
        "sh",
        # Bun and Deno are intentionally excluded — different preload
        # mechanism — so detector must NOT classify them.
        "bun",
        "deno",
        # Edge cases: arbitrary executables containing "node" but not
        # the runtime itself.
        "nodemon",  # popular Node dev tool, NOT the runtime
        "node-gyp",
        "/usr/local/bin/code",  # VS Code CLI
    ],
)
def test_is_node_runtime_false_for_other_executables(executable: str) -> None:
    """Non-Node executables — including Node-adjacent tools — must
    not trigger NODE_OPTIONS injection. False positives would set
    NODE_OPTIONS on commands that don't honor it."""
    assert not _is_node_runtime(executable), f"{executable!r} should NOT be a Node runtime"


# ---- end-to-end record subprocess ---------------------------------------


def _run_record(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke `shadow record` end-to-end via a subprocess. We can't
    just call the typer.Context-based command in-process because it
    raises typer.Exit, which is awkward to assert against."""
    return subprocess.run(
        [sys.executable, "-m", "shadow.cli.app", "record", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )


def test_record_does_not_inject_node_options_for_python_command(tmp_path: Path) -> None:
    """The Node injection path is gated on the runtime classifier;
    Python commands must NOT pick up NODE_OPTIONS=--import...

    We verify by running a tiny inline Python that checks its own
    NODE_OPTIONS env var. NODE_OPTIONS may legitimately be set by
    the parent environment, so we only assert the Shadow injection
    string is absent (not that the variable itself is absent).
    """
    out = tmp_path / "trace.agentlog"
    result = _run_record(
        tmp_path,
        "-o",
        str(out),
        "--",
        sys.executable,
        "-c",
        "import os; print('NODEOPTS=' + os.environ.get('NODE_OPTIONS', '<unset>'))",
    )
    assert result.returncode == 0, result.stderr
    assert "shadow-diff/auto" not in result.stdout
    assert "shadow-diff/auto" not in result.stderr


def test_record_injects_node_options_for_node_command(tmp_path: Path) -> None:
    """`shadow record -- node -e '...'` must inject
    `--import shadow-diff/auto` into NODE_OPTIONS so the Node
    runtime activates the TS SDK's auto entrypoint."""
    out = tmp_path / "trace.agentlog"
    # We use a fake `node` shim that just echoes its NODE_OPTIONS env
    # var. This avoids requiring real Node + a published shadow-diff
    # npm install in the test sandbox — we're only verifying that the
    # Python CLI sets the env correctly. End-to-end Node behaviour is
    # covered in typescript/test/auto.test.ts.
    fake_node = tmp_path / "node"
    fake_node.write_text('#!/usr/bin/env bash\necho "NODEOPTS=${NODE_OPTIONS:-<unset>}"\nexit 0\n')
    fake_node.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(
        [sys.executable, "-m", "shadow.cli.app", "record", "-o", str(out), "--", "node"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "--import shadow-diff/auto" in result.stdout
    ), f"expected NODE_OPTIONS injection in stdout, got:\n{result.stdout}"


def test_record_no_auto_instrument_disables_node_options_injection(tmp_path: Path) -> None:
    """`--no-auto-instrument` must disable both the Python sitecustomize
    shim AND the Node NODE_OPTIONS injection. The flag is the escape
    hatch for users who already opened their own Session in-process."""
    out = tmp_path / "trace.agentlog"
    fake_node = tmp_path / "node"
    fake_node.write_text('#!/usr/bin/env bash\necho "NODEOPTS=${NODE_OPTIONS:-<unset>}"\nexit 0\n')
    fake_node.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "record",
            "-o",
            str(out),
            "--no-auto-instrument",
            "--",
            "node",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "shadow-diff/auto" not in result.stdout
    ), f"expected NO injection under --no-auto-instrument, got:\n{result.stdout}"


def test_record_preserves_existing_node_options_when_injecting(tmp_path: Path) -> None:
    """If the user already has NODE_OPTIONS set (e.g. for memory
    limits), the Shadow injection must prepend, not clobber."""
    out = tmp_path / "trace.agentlog"
    fake_node = tmp_path / "node"
    fake_node.write_text('#!/usr/bin/env bash\necho "NODEOPTS=${NODE_OPTIONS:-<unset>}"\nexit 0\n')
    fake_node.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    env["NODE_OPTIONS"] = "--max-old-space-size=4096"
    result = subprocess.run(
        [sys.executable, "-m", "shadow.cli.app", "record", "-o", str(out), "--", "node"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--import shadow-diff/auto" in result.stdout
    assert (
        "--max-old-space-size=4096" in result.stdout
    ), "user's existing NODE_OPTIONS must be preserved alongside Shadow's injection"
