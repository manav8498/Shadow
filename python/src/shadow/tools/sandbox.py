"""``SandboxedToolBackend`` — wrap user tool functions, block side effects.

This backend is what makes "shadow deployment" actually shadow: the
candidate agent calls the user's *real* tool functions, but every
side-effect-producing primitive those functions might use is patched
out so the call never reaches production.

Scope of the sandbox
--------------------

Best-effort isolation, not a security boundary. The patch list is:

- **Network**:
  - ``socket.socket`` connect → blocked
  - ``http.client.HTTPConnection`` and ``HTTPSConnection`` → blocked
  - top-level ``httpx``, ``requests``, ``aiohttp`` clients → blocked
- **Filesystem writes**:
  - all opens with mode containing ``w``, ``a``, ``+``, or ``x`` are
    redirected to a tempdir (the file path under tempdir mirrors the
    original so relative-path code that reads-then-writes the same
    file still works)
- **Subprocess**:
  - ``subprocess.Popen``, ``subprocess.run``, ``os.system``,
    ``os.execvp`` → blocked
- **Time** (optional):
  - When ``freeze_time`` is supplied, ``time.time`` and
    ``datetime.datetime.utcnow`` are patched to return that fixed
    instant.

Each patch raises a clear :class:`SandboxViolation` so a tool that
secretly tries to phone home surfaces in the trace as an
``is_error=True`` ``tool_result`` rather than silently succeeding.
None of this stops a tool from doing CPU computation, calling another
tool, or returning fabricated results — those are the agent's
business.

Design notes
------------

- All patching is done inside an ``asyncio.Lock``-guarded context so
  two concurrent sandboxed sessions don't race on the global module
  patches. The engine drives sessions sequentially by default; the
  lock is the safety net for users who run them in parallel.
- The sandbox patches the *ambient runtime*, not the tool function
  itself. We don't introspect the tool function's source, just bound
  what it can reach during execution.
- Patching is idempotent: nesting two ``__aenter__`` calls is a
  no-op for the inner one.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from shadow.tools.base import ToolBackend, ToolCall, ToolResult


class SandboxViolation(RuntimeError):
    """A sandboxed tool tried to perform a blocked side effect.

    Carries the operation name (``connect``, ``Popen``, ``open(w)``)
    and any salient args so the caller can render an actionable
    ``tool_result`` payload.
    """

    def __init__(self, operation: str, detail: str = "") -> None:
        super().__init__(f"sandbox blocked {operation}{': ' + detail if detail else ''}")
        self.operation = operation
        self.detail = detail


# Module-level lock so simultaneous sandboxed sessions don't tear
# each other's patches. asyncio.Lock is the right primitive because
# the engine drives the loop in async code.
_SANDBOX_LOCK = asyncio.Lock()


class SandboxedToolBackend(ToolBackend):
    """Run real tool functions but block their side effects.

    Parameters
    ----------
    tool_registry
        Mapping ``tool_name → async callable``. The callable receives
        the call's argument dict and returns a string or JSON-serialisable
        object.
    block_network
        Patch out all network primitives. Default True.
    block_subprocess
        Patch out subprocess / os.system / os.execvp. Default True.
    redirect_writes_to
        Path to a directory where write-mode opens are redirected.
        Default is a freshly-created tempdir whose lifetime tracks the
        backend instance.
    freeze_time
        Optional fixed UTC datetime. When set, ``time.time`` and
        ``datetime.datetime.utcnow`` return that instant during tool
        execution. Helps reproducibility for time-dependent tools.
    backend_id
        Stable identifier surfaced as ``self.id``.
    """

    def __init__(
        self,
        tool_registry: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]],
        *,
        block_network: bool = True,
        block_subprocess: bool = True,
        redirect_writes_to: Path | str | None = None,
        freeze_time: datetime.datetime | None = None,
        backend_id: str = "sandbox",
    ) -> None:
        self._tools = tool_registry
        self._block_network = block_network
        self._block_subprocess = block_subprocess
        if redirect_writes_to is None:
            self._tmpdir = Path(tempfile.mkdtemp(prefix="shadow-sandbox-"))
            self._owns_tmpdir = True
        else:
            self._tmpdir = Path(redirect_writes_to)
            self._tmpdir.mkdir(parents=True, exist_ok=True)
            self._owns_tmpdir = False
        self._freeze_time = freeze_time
        self._id = backend_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def tmpdir(self) -> Path:
        return self._tmpdir

    def __del__(self) -> None:  # pragma: no cover - GC timing
        if getattr(self, "_owns_tmpdir", False):
            with contextlib.suppress(Exception):
                import shutil

                shutil.rmtree(self._tmpdir)

    # ---- ToolBackend.execute ------------------------------------------

    async def execute(self, call: ToolCall) -> ToolResult:
        fn = self._tools.get(call.name)
        if fn is None:
            return ToolResult(
                tool_call_id=call.id,
                output=f"<sandbox: unknown tool {call.name!r}>",
                is_error=True,
                latency_ms=0,
            )
        started = time.perf_counter()
        try:
            async with _SANDBOX_LOCK, self._patches():
                output = await fn(call.arguments)
        except SandboxViolation as exc:
            return ToolResult(
                tool_call_id=call.id,
                output=f"sandbox violation: {exc}",
                is_error=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                output=f"{type(exc).__name__}: {exc}",
                is_error=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ToolResult(
            tool_call_id=call.id,
            output=_normalize_output(output),
            is_error=False,
            latency_ms=latency_ms,
        )

    # ---- patch context -------------------------------------------------

    @contextlib.asynccontextmanager
    async def _patches(self) -> Any:
        """Apply all enabled patches for the duration of one tool call.

        Each patch is independent; a disabled flag (e.g.
        ``block_network=False``) leaves the underlying primitive
        untouched. Restore order is the inverse of install order so
        the runtime returns to baseline state cleanly.
        """
        restorers: list[Callable[[], None]] = []
        try:
            if self._block_network:
                restorers.extend(_install_network_block())
            if self._block_subprocess:
                restorers.extend(_install_subprocess_block())
            restorers.extend(_install_write_redirect(self._tmpdir))
            if self._freeze_time is not None:
                restorers.extend(_install_time_freeze(self._freeze_time))
            yield
        finally:
            for restore in reversed(restorers):
                with contextlib.suppress(Exception):
                    restore()


# ---- patch installers ----------------------------------------------------


def _install_network_block() -> list[Callable[[], None]]:
    """Block socket / http.client connects.

    ``httpx``, ``requests``, and ``aiohttp`` ride on top of either
    ``socket.socket`` or the asyncio loop's transport. Patching the
    socket layer is sufficient to break all three.
    """
    restorers: list[Callable[[], None]] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked_connect(self: socket.socket, address: Any) -> None:
        raise SandboxViolation("socket.connect", str(address))

    def blocked_connect_ex(self: socket.socket, address: Any) -> int:
        raise SandboxViolation("socket.connect_ex", str(address))

    socket.socket.connect = blocked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked_connect_ex  # type: ignore[method-assign]

    def restore() -> None:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]

    restorers.append(restore)
    return restorers


def _install_subprocess_block() -> list[Callable[[], None]]:
    """Block subprocess.run / Popen / os.system / os.execvp."""
    restorers: list[Callable[[], None]] = []

    original_popen_init = subprocess.Popen.__init__
    original_run = subprocess.run
    original_system = os.system
    original_execvp = os.execvp

    def blocked_popen_init(self: Any, *args: Any, **kwargs: Any) -> None:
        raise SandboxViolation("subprocess.Popen", str(args[0]) if args else "")

    def blocked_run(*args: Any, **kwargs: Any) -> Any:
        raise SandboxViolation("subprocess.run", str(args[0]) if args else "")

    def blocked_system(command: str) -> int:
        raise SandboxViolation("os.system", command)

    def blocked_execvp(file: str, args: Any) -> None:
        raise SandboxViolation("os.execvp", file)

    subprocess.Popen.__init__ = blocked_popen_init  # type: ignore[method-assign]
    subprocess.run = blocked_run  # type: ignore[assignment]
    os.system = blocked_system  # type: ignore[assignment]
    os.execvp = blocked_execvp  # type: ignore[assignment]

    def restore() -> None:
        subprocess.Popen.__init__ = original_popen_init  # type: ignore[method-assign]
        subprocess.run = original_run  # type: ignore[assignment]
        os.system = original_system  # type: ignore[assignment]
        os.execvp = original_execvp  # type: ignore[assignment]

    restorers.append(restore)
    return restorers


def _install_write_redirect(tmpdir: Path) -> list[Callable[[], None]]:
    """Redirect write-mode ``open()`` calls into a tempdir.

    The original path is preserved as a relative-path subtree inside
    the tempdir, so a tool that writes ``./out/foo.txt`` ends up at
    ``{tmpdir}/out/foo.txt``. Read-mode opens pass through unchanged
    so a tool can still read snapshot fixtures.

    Pure read-only opens (``r``, ``rb``) are NEVER intercepted, so
    config files, prompts, and anything else the tool reads at start
    keep working.
    """
    restorers: list[Callable[[], None]] = []
    builtins_mod = sys.modules.get("builtins")
    if builtins_mod is None:
        return restorers
    original_open = builtins_mod.open

    def patched_open(
        file: Any,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(mode, str) or not any(c in mode for c in ("w", "a", "+", "x")):
            return original_open(file, mode, *args, **kwargs)
        if not isinstance(file, str | bytes | os.PathLike):
            # Open on a fd or non-path object — let it through; that's
            # not a path the sandbox can meaningfully redirect.
            return original_open(file, mode, *args, **kwargs)
        target = Path(os.fsdecode(file))
        # Strip leading separators / drive letter so path joining lands
        # safely under tmpdir on every platform.
        try:
            relative = target.relative_to(target.anchor) if target.is_absolute() else target
        except ValueError:
            relative = Path(target.name)
        redirected = tmpdir / relative
        redirected.parent.mkdir(parents=True, exist_ok=True)
        return original_open(redirected, mode, *args, **kwargs)

    builtins_mod.open = patched_open  # type: ignore[assignment]

    def restore() -> None:
        builtins_mod.open = original_open  # type: ignore[assignment]

    restorers.append(restore)
    return restorers


def _install_time_freeze(frozen: datetime.datetime) -> list[Callable[[], None]]:
    """Pin ``time.time`` and ``datetime.utcnow`` to a fixed instant."""
    restorers: list[Callable[[], None]] = []
    epoch = frozen.timestamp() if frozen.tzinfo else frozen.replace(tzinfo=datetime.UTC).timestamp()
    original_time = time.time
    original_utcnow = datetime.datetime.utcnow

    def fixed_time() -> float:
        return epoch

    def fixed_utcnow() -> datetime.datetime:
        return frozen.replace(tzinfo=None) if frozen.tzinfo else frozen

    time.time = fixed_time  # type: ignore[assignment]
    datetime.datetime.utcnow = fixed_utcnow  # type: ignore[assignment]

    def restore() -> None:
        time.time = original_time  # type: ignore[assignment]
        datetime.datetime.utcnow = original_utcnow  # type: ignore[assignment]

    restorers.append(restore)
    return restorers


# ---- helpers ------------------------------------------------------------


def _normalize_output(value: Any) -> str | dict[str, Any]:
    """Coerce an arbitrary tool return into a Shadow-record-compatible value.

    Strings pass through. Dicts pass through. Lists and other
    JSON-serialisable types are wrapped under ``{"value": ...}`` so
    the record's ``output`` field stays a string-or-dict union.
    Anything else is ``str()``'d.
    """
    if isinstance(value, str | dict):
        return value
    if isinstance(value, list | tuple):
        return {"value": list(value)}
    if value is None:
        return ""
    return str(value)


__all__ = [
    "SandboxViolation",
    "SandboxedToolBackend",
]
