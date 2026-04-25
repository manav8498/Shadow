"""Tests for ``SandboxedToolBackend``.

Best-effort isolation, not a security boundary — we test that the
documented patches actually fire on the documented operations and
that successful calls flow through unchanged.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

from shadow.tools.base import ToolCall
from shadow.tools.sandbox import SandboxedToolBackend, SandboxViolation

# ---- success path -------------------------------------------------------


def test_sandbox_runs_a_pure_tool_function() -> None:
    async def add(args: dict[str, Any]) -> str:
        return f"result: {args['x'] + args['y']}"

    sb = SandboxedToolBackend({"add": add})
    r = asyncio.run(sb.execute(ToolCall("c", "add", {"x": 2, "y": 3})))
    assert r.output == "result: 5"
    assert not r.is_error


def test_sandbox_unknown_tool_returns_error_result() -> None:
    sb = SandboxedToolBackend({})
    r = asyncio.run(sb.execute(ToolCall("c", "ghost", {})))
    assert r.is_error
    assert "unknown tool" in str(r.output)


def test_sandbox_tool_exception_becomes_is_error_result() -> None:
    async def explodes(args: dict[str, Any]) -> str:
        raise ValueError("boom")

    sb = SandboxedToolBackend({"explodes": explodes})
    r = asyncio.run(sb.execute(ToolCall("c", "explodes", {})))
    assert r.is_error
    assert "ValueError" in str(r.output)


# ---- network blocking ---------------------------------------------------


def test_sandbox_blocks_socket_connect() -> None:
    async def phone_home(args: dict[str, Any]) -> str:
        s = socket.socket()
        s.connect(("1.1.1.1", 80))
        return "should not reach"

    sb = SandboxedToolBackend({"phone": phone_home})
    r = asyncio.run(sb.execute(ToolCall("c", "phone", {})))
    assert r.is_error
    assert "socket.connect" in str(r.output)


def test_sandbox_with_block_network_false_lets_socket_through() -> None:
    """When the user explicitly opts out of network blocking, the
    patch isn't installed. We can't prove a real connect succeeds in
    a hermetic test, but we can verify the flag at least disables
    the SandboxViolation on a no-op path."""

    async def harmless(args: dict[str, Any]) -> str:
        # Construct a socket but don't connect — proves the patch
        # isn't intercepting at the constructor level.
        s = socket.socket()
        s.close()
        return "ok"

    sb = SandboxedToolBackend({"x": harmless}, block_network=False)
    r = asyncio.run(sb.execute(ToolCall("c", "x", {})))
    assert r.output == "ok"


# ---- subprocess blocking -----------------------------------------------


def test_sandbox_blocks_subprocess_run() -> None:
    async def shell_out(args: dict[str, Any]) -> str:
        subprocess.run(["echo", "hi"], check=False)
        return "should not reach"

    sb = SandboxedToolBackend({"shell": shell_out})
    r = asyncio.run(sb.execute(ToolCall("c", "shell", {})))
    assert r.is_error
    assert "subprocess.run" in str(r.output)


def test_sandbox_blocks_os_system() -> None:
    async def runs_os_system(args: dict[str, Any]) -> str:
        os.system("echo hi")
        return "should not reach"

    sb = SandboxedToolBackend({"sys": runs_os_system})
    r = asyncio.run(sb.execute(ToolCall("c", "sys", {})))
    assert r.is_error
    assert "os.system" in str(r.output)


# ---- filesystem-write redirection --------------------------------------


def test_sandbox_redirects_writes_to_tmpdir(tmp_path: Path) -> None:
    """A tool that writes ``./out/foo.txt`` ends up under the
    sandbox's tmpdir, not on the user's real filesystem."""
    target = "shadow_sandbox_write_test.txt"

    async def writes(args: dict[str, Any]) -> str:
        with open(target, "w") as f:
            f.write("payload")
        return "wrote"

    sb_tmp = tmp_path / "sandbox"
    sb = SandboxedToolBackend({"write": writes}, redirect_writes_to=sb_tmp)
    r = asyncio.run(sb.execute(ToolCall("c", "write", {})))
    assert not r.is_error
    # The file landed under the sandbox tmpdir, not in cwd.
    assert (sb_tmp / target).exists()
    # And NOT in the actual cwd.
    assert not Path(target).exists()


def test_sandbox_passes_read_mode_opens_through(tmp_path: Path) -> None:
    """Read-mode opens are NEVER intercepted so config files etc.
    keep working."""
    config = tmp_path / "config.txt"
    config.write_text("hello")

    async def reads(args: dict[str, Any]) -> str:
        return open(args["path"]).read()

    sb = SandboxedToolBackend({"read": reads}, redirect_writes_to=tmp_path / "sb")
    r = asyncio.run(sb.execute(ToolCall("c", "read", {"path": str(config)})))
    assert r.output == "hello"


# ---- patch isolation ----------------------------------------------------


def test_sandbox_restores_patches_after_call() -> None:
    """The patches must restore so a subsequent call outside the
    sandbox isn't broken."""

    async def harmless(args: dict[str, Any]) -> str:
        return "ok"

    sb = SandboxedToolBackend({"x": harmless})
    asyncio.run(sb.execute(ToolCall("c", "x", {})))
    # After the sandboxed call, `subprocess.run` works again.
    completed = subprocess.run(["true"], capture_output=True, check=False)
    assert completed.returncode == 0


# ---- the SandboxViolation type itself ---------------------------------


def test_sandbox_violation_carries_operation_and_detail() -> None:
    err = SandboxViolation("socket.connect", "1.1.1.1:80")
    assert err.operation == "socket.connect"
    assert err.detail == "1.1.1.1:80"
    assert "socket.connect" in str(err)
    assert "1.1.1.1:80" in str(err)
