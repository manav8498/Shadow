"""Tests for `shadow record -- node ...` auto-instrumentation.

Mirrors `test_autostart.py` for Python agents — verifies the
Node-runtime detection logic and that `NODE_OPTIONS` is
injected when the wrapped command is a Node-family CLI.

End-to-end recording (actually running a Node agent) is
tested in `typescript/test/auto.test.ts`; this file focuses
on the Python-side wiring decisions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from shadow.cli.app import _is_bun_runtime, _is_node_runtime

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
        # Package managers — they spawn Node child processes that
        # inherit NODE_OPTIONS. The TS auto entrypoint detects the
        # wrapper itself via `process.argv[1]` and skips, so only
        # the grandchild user agent records.
        "npm",
        "npm.cmd",
        "pnpm",
        "pnpm.cmd",
        "yarn",
        "yarn.cmd",
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
        # Bun and Deno are NOT Node runtimes — bun has its own preload
        # mechanism (handled by _is_bun_runtime + the --preload argv
        # injection); deno doesn't currently have a parallel injection
        # path. The detector must NOT classify either as Node-family.
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


# ---- _is_bun_runtime() classifier ---------------------------------------


@pytest.mark.parametrize(
    "executable",
    [
        "bun",
        "/opt/homebrew/bin/bun",
        "/usr/local/bin/bun",
        "bun.exe",
        "C:\\Users\\me\\.bun\\bin\\bun.exe",
        "bunx",
        "bunx.exe",
    ],
)
def test_is_bun_runtime_true_for_bun(executable: str) -> None:
    """Every bun / bunx variant — including Windows .exe — must register."""
    assert _is_bun_runtime(executable), f"{executable!r} should be a Bun runtime"


@pytest.mark.parametrize(
    "executable",
    [
        "node",
        "python",
        "deno",
        # An executable named "bunny" or "bundler" must NOT trigger.
        "bunny",
        "bundler",
        "bun-extras",
    ],
)
def test_is_bun_runtime_false_for_others(executable: str) -> None:
    """Non-bun executables — even ones whose name starts with `bun` —
    must not trigger the bun preload-injection path."""
    assert not _is_bun_runtime(executable), f"{executable!r} should NOT be a Bun runtime"


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


def _invoke_record_with_mocked_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    out: Path,
    *extra_args: str,
    extra_env: dict[str, str] | None = None,
    wrapped_cmd: tuple[str, ...] = ("node", "user-agent.js"),
) -> dict[str, str]:
    """Backwards-compatible helper that returns the env dict only."""
    return _invoke_record_with_capture(
        monkeypatch, out, *extra_args, extra_env=extra_env, wrapped_cmd=wrapped_cmd
    )["env"]


def _invoke_record_with_capture(
    monkeypatch: pytest.MonkeyPatch,
    out: Path,
    *extra_args: str,
    extra_env: dict[str, str] | None = None,
    wrapped_cmd: tuple[str, ...] = ("node", "user-agent.js"),
) -> dict[str, list[str] | dict[str, str]]:
    """Run `shadow record` in-process via CliRunner with `subprocess.run`
    mocked. Returns the cmd argv list AND env dict the wrapped command
    would have seen.

    No real subprocess spawn — works identically on Linux, macOS, and
    Windows. The mock returns a CompletedProcess with returncode 0 so
    `record()`'s `typer.Exit(code=result.returncode)` propagates as
    exit_code 0 in the runner.
    """
    from typer.testing import CliRunner

    from shadow.cli import app as _app_module

    captured: dict[str, Any] = {}  # type: ignore[name-defined]

    def _fake_run(cmd, env, check):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_app_module.subprocess, "run", _fake_run)
    if extra_env:
        for k, v in extra_env.items():
            monkeypatch.setenv(k, v)

    runner = CliRunner()
    args = ["record", "-o", str(out), *extra_args, "--", *wrapped_cmd]
    result = runner.invoke(_app_module.app, args)
    assert result.exit_code == 0, f"record exited {result.exit_code}; output:\n{result.output}"
    return captured


def test_record_injects_node_options_for_node_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`shadow record -- node ...` must inject `--import shadow-diff/auto`
    into NODE_OPTIONS so the Node runtime activates the TS SDK's auto
    entrypoint. End-to-end Node behaviour lives in
    `typescript/test/auto.test.ts`; this test pins the Python-side
    env-wiring decision."""
    out = tmp_path / "trace.agentlog"
    env = _invoke_record_with_mocked_subprocess(monkeypatch, out)
    assert "--import shadow-diff/auto" in env.get(
        "NODE_OPTIONS", ""
    ), f"expected NODE_OPTIONS injection, got: {env.get('NODE_OPTIONS')!r}"
    assert env.get("SHADOW_SESSION_OUTPUT") == str(out.resolve())


def test_record_no_auto_instrument_disables_node_options_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-auto-instrument` must disable both the Python sitecustomize
    shim AND the Node NODE_OPTIONS injection. The flag is the escape
    hatch for users who already opened their own Session in-process."""
    out = tmp_path / "trace.agentlog"
    env = _invoke_record_with_mocked_subprocess(monkeypatch, out, "--no-auto-instrument")
    assert "shadow-diff/auto" not in env.get("NODE_OPTIONS", ""), (
        f"expected NO injection under --no-auto-instrument, " f"got: {env.get('NODE_OPTIONS')!r}"
    )


def test_record_preserves_existing_node_options_when_injecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the user already has NODE_OPTIONS set (e.g. for memory
    limits), the Shadow injection must prepend, not clobber."""
    out = tmp_path / "trace.agentlog"
    env = _invoke_record_with_mocked_subprocess(
        monkeypatch,
        out,
        extra_env={"NODE_OPTIONS": "--max-old-space-size=4096"},
    )
    seen = env.get("NODE_OPTIONS", "")
    assert "--import shadow-diff/auto" in seen, f"expected injection, got: {seen!r}"
    assert (
        "--max-old-space-size=4096" in seen
    ), f"existing NODE_OPTIONS must be preserved, got: {seen!r}"


def test_record_does_not_inject_for_npm_when_argv_is_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity: when the wrapped command is plainly Python, the
    classifier returns False and no NODE_OPTIONS injection happens."""
    out = tmp_path / "trace.agentlog"
    env = _invoke_record_with_mocked_subprocess(
        monkeypatch, out, wrapped_cmd=("python", "agent.py")
    )
    assert "shadow-diff/auto" not in env.get(
        "NODE_OPTIONS", ""
    ), f"python wrapper got an unexpected injection: {env.get('NODE_OPTIONS')!r}"


# ---- Bun argv-injection -------------------------------------------------


def test_record_injects_preload_for_bun_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`shadow record -- bun ...` must insert `--preload shadow-diff/auto`
    into argv right after the bun executable. This is the bun-equivalent
    of the Node `--import` injection — tests the regression that BrowserOS
    reported (no auto-instrumentation under bun)."""
    out = tmp_path / "trace.agentlog"
    captured = _invoke_record_with_capture(
        monkeypatch, out, wrapped_cmd=("bun", "run", "my-agent.ts")
    )
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "bun"
    # The preload flag must be inserted right after the executable.
    assert cmd[1] == "--preload"
    assert cmd[2] == "shadow-diff/auto"
    # User's original args follow.
    assert cmd[3:] == ["run", "my-agent.ts"]
    # SHADOW_SESSION_OUTPUT must still be set so the auto entrypoint
    # can read it on bun startup.
    env = captured["env"]
    assert env["SHADOW_SESSION_OUTPUT"] == str(out.resolve())


def test_record_injects_preload_for_bunx_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bunx` (the bun npx-equivalent) gets the same preload injection."""
    out = tmp_path / "trace.agentlog"
    captured = _invoke_record_with_capture(
        monkeypatch, out, wrapped_cmd=("bunx", "tsx", "agent.ts")
    )
    cmd = captured["cmd"]
    assert cmd[:3] == ["bunx", "--preload", "shadow-diff/auto"]


def test_record_no_auto_instrument_disables_bun_preload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-auto-instrument` must disable the bun --preload injection.

    The escape hatch for users who already opened their own Session
    should suppress the bun path in the same way it suppresses the
    Python sitecustomize and Node NODE_OPTIONS paths.
    """
    out = tmp_path / "trace.agentlog"
    captured = _invoke_record_with_capture(
        monkeypatch,
        out,
        "--no-auto-instrument",
        wrapped_cmd=("bun", "run", "my-agent.ts"),
    )
    cmd = captured["cmd"]
    # No --preload injection — argv is the user's original verbatim.
    assert "--preload" not in cmd
    assert cmd == ["bun", "run", "my-agent.ts"]


def test_record_does_not_inject_preload_for_node_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bun preload injection must not fire for Node — Node uses
    NODE_OPTIONS instead. Otherwise we'd both set NODE_OPTIONS AND
    inject --preload (which Node doesn't recognise) into argv."""
    out = tmp_path / "trace.agentlog"
    captured = _invoke_record_with_capture(monkeypatch, out, wrapped_cmd=("node", "agent.js"))
    cmd = captured["cmd"]
    assert "--preload" not in cmd
    assert cmd == ["node", "agent.js"]
    # NODE_OPTIONS injection is the Node path.
    env = captured["env"]
    assert "shadow-diff/auto" in env.get("NODE_OPTIONS", "")
