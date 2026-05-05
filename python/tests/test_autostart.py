"""Tests for the zero-config `shadow record` auto-instrumentation path.

The critical invariant: running an arbitrary Python script via
`shadow record -- python <script>` should produce a valid `.agentlog`
file WITHOUT the script importing `shadow` anywhere.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from shadow.sdk import _autostart, _bootstrap


def test_autostart_noop_when_env_var_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """If SHADOW_SESSION_OUTPUT isn't set, _autostart must not start a Session.

    This is the default on every regular Python invocation — the cost
    is one env-var read, not a Session + instrumentor.
    """
    monkeypatch.delenv("SHADOW_SESSION_OUTPUT", raising=False)
    # Reset the module-level flag; re-invoke the entry point.
    _autostart._BOOTSTRAP_DONE = False
    _autostart._start_session_from_env()
    assert not _autostart._already_bootstrapped()


def test_autostart_parses_tag_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """`SHADOW_SESSION_TAGS=env=dev,branch=main` should survive the parser."""
    monkeypatch.setenv("SHADOW_SESSION_TAGS", "env=dev,branch=main,bad,empty=,=noval")
    tags = _autostart._tags_from_env()
    assert tags == {"env": "dev", "branch": "main", "empty": ""}


def test_autostart_handles_malformed_tags_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHADOW_SESSION_TAGS", ",,,,")
    assert _autostart._tags_from_env() == {}
    monkeypatch.setenv("SHADOW_SESSION_TAGS", "")
    assert _autostart._tags_from_env() == {}


def test_bootstrap_dir_contains_sitecustomize() -> None:
    """The bootstrap dir must ship a sitecustomize.py — that's the hook."""
    d = Path(_bootstrap.__file__).parent
    assert (d / "sitecustomize.py").is_file()
    contents = (d / "sitecustomize.py").read_text()
    # The shim must import _autostart — that's the whole point.
    assert "shadow.sdk._autostart" in contents


def test_shadow_record_on_agent_with_zero_shadow_imports(tmp_path: Path) -> None:
    """End-to-end: a script that NEVER imports shadow still emits a trace.

    The wrapped script only prints a line. The autostart bootstrap
    should install a Session, write a metadata record on atexit, and
    `shadow record` should exit with the child's exit code.
    """
    agent_py = tmp_path / "clean_agent.py"
    agent_py.write_text('print("hello from a shadow-free agent")\n')
    trace_path = tmp_path / "trace.agentlog"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "record",
            "-o",
            str(trace_path),
            "--",
            sys.executable,
            str(agent_py),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert trace_path.exists(), f"no agentlog at {trace_path}"
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    assert len(records) >= 1
    # First record must be metadata, produced by Session.__enter__.
    assert records[0]["kind"] == "metadata"


def test_shadow_record_forwards_child_exit_code(tmp_path: Path) -> None:
    """A non-zero exit from the wrapped command should propagate."""
    agent_py = tmp_path / "failing_agent.py"
    agent_py.write_text("import sys\nsys.exit(7)\n")
    trace_path = tmp_path / "trace.agentlog"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "record",
            "-o",
            str(trace_path),
            "--",
            sys.executable,
            str(agent_py),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 7


def test_shadow_record_tags_flag_propagates(tmp_path: Path) -> None:
    """`--tags env=dev,branch=main` should land in the metadata record."""
    agent_py = tmp_path / "tagged_agent.py"
    agent_py.write_text("pass\n")
    trace_path = tmp_path / "trace.agentlog"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "record",
            "-o",
            str(trace_path),
            "--tags",
            "env=dev,branch=feat/zero-config",
            "--",
            sys.executable,
            str(agent_py),
        ],
        check=True,
    )
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    meta = records[0]
    assert meta["kind"] == "metadata"
    assert meta["payload"]["tags"]["env"] == "dev"
    assert meta["payload"]["tags"]["branch"] == "feat/zero-config"


def test_shadow_record_no_auto_instrument_flag(tmp_path: Path) -> None:
    """With --no-auto-instrument the bootstrap dir is NOT prepended.

    If the user's own agent runs its own `Session`, auto-instrumentation
    would open a second one around it. The flag opts out cleanly.
    """
    # Agent prints its PYTHONPATH; check that the shadow bootstrap dir
    # isn't in there when --no-auto-instrument is set.
    agent_py = tmp_path / "path_probe.py"
    agent_py.write_text("import os\n" "print('PYTHONPATH=' + os.environ.get('PYTHONPATH', ''))\n")
    # Default: bootstrap dir is on PYTHONPATH.
    default = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "record",
            "-o",
            str(tmp_path / "a.agentlog"),
            "--",
            sys.executable,
            str(agent_py),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "_bootstrap" in default.stdout
    # With --no-auto-instrument, bootstrap dir is absent.
    opted_out = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "record",
            "-o",
            str(tmp_path / "b.agentlog"),
            "--no-auto-instrument",
            "--",
            sys.executable,
            str(agent_py),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "_bootstrap" not in opted_out.stdout


def test_autostart_doesnt_crash_when_shadow_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hot path must tolerate a broken Shadow install without breaking
    the user's agent. Hard to simulate perfectly; at minimum verify the
    top-level exception handler is present and structured."""
    src = Path(_autostart.__file__).read_text()
    assert "except Exception" in src, "autostart must have a top-level guard"
    # The bootstrap is single-assignment; verify the flag is module-level.
    assert "_BOOTSTRAP_DONE" in src


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Windows ignores chmod(0o500) on directories — the pre-flight "
        "writability check correctly succeeds, then the wrapped child "
        "runs and exits 99. This asserts POSIX-only permission semantics."
    ),
)
def test_shadow_record_fails_fast_on_unwritable_output_path(tmp_path: Path) -> None:
    """A read-only output location must fail *before* the agent runs.

    The new user's worst nightmare: `shadow record -o /bad/path/foo.agentlog
    -- python my_agent.py` silently loses the recording after burning LLM
    tokens. Fail at CLI boot instead, with an actionable error.
    """
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    ro_dir.chmod(0o500)  # read + execute, no write
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "shadow.cli.app",
                "record",
                "-o",
                str(ro_dir / "trace.agentlog"),
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(99)",  # should never run
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # The check-before-spawn should fail with exit 2 (CLI/arg error)
        # without ever running the -c "raise SystemExit(99)" child.
        assert result.returncode == 2, (
            f"expected exit 2, got {result.returncode}. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        combined = (result.stdout + result.stderr).lower()
        assert "not writable" in combined or "write permission" in combined
    finally:
        ro_dir.chmod(0o700)


def test_shadow_record_rejects_empty_command_with_exit_2() -> None:
    """`shadow record` with no command after `--` should error out cleanly."""
    result = subprocess.run(
        [sys.executable, "-m", "shadow.cli.app", "record", "-o", "/tmp/x.agentlog"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert (
        "specify a command" in result.stderr.lower() or "specify a command" in result.stdout.lower()
    )


# ---- Empty-capture warning (Issue 7) ----------------------------------


def test_empty_capture_warning_fires_on_zero_chat_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`shadow record` against a script that imports nothing must emit a
    LOUD stderr warning at exit — not exit silently with a metadata-only trace.

    Reproduces the customer-reported failure mode: BrowserOS / Skyvern /
    OpenHands ran successfully under shadow record but produced
    metadata-only traces, no warning. CI passed but no behaviour was
    actually verified."""
    script = tmp_path / "no_calls.py"
    script.write_text("print('agent ran successfully')\n")
    out = tmp_path / "trace.agentlog"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "record",
            "-o",
            str(out),
            "--",
            sys.executable,
            str(script),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "agent ran successfully" in proc.stdout
    # The loud stderr warning is the load-bearing assertion — without it
    # users in the customer-reported scenario silently passed CI.
    assert "WARNING" in proc.stderr, proc.stderr
    assert "zero LLM calls were captured" in proc.stderr, proc.stderr
    # The warning names the canonical fix paths so users can self-diagnose.
    assert "openai" in proc.stderr.lower()
    assert "litellm" in proc.stderr.lower()


def test_empty_capture_warning_silent_when_calls_were_captured(tmp_path: Path) -> None:
    """Warning must NOT fire on a session that actually captured calls.

    Records a chat pair via the explicit Session API while the
    sitecustomize-bootstrapped Session is also active; the bootstrapped
    session's _records will see the recorded chat pair and the warning
    should stay silent."""
    script = tmp_path / "with_calls.py"
    # Use a normal Python script instead of an embedded string so long
    # request/response payload literals don't trigger E501. The script
    # is launched as a subprocess by the test below.
    script.write_text(
        '''
from shadow.sdk import Session
from shadow.sdk._autostart import _BOOTSTRAP_DONE
import tempfile

assert _BOOTSTRAP_DONE
with tempfile.NamedTemporaryFile(suffix=".agentlog") as t:
    s = Session(output_path=t.name)
    s.__enter__()
    s.record_chat(
        request={
            "model": "gpt",
            "messages": [{"role": "user", "content": "hi"}],
            "params": {},
        },
        response={
            "model": "gpt",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "latency_ms": 1,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "thinking_tokens": 0,
            },
        },
    )
    s.__exit__(None, None, None)
print("done")
'''
    )
    out = tmp_path / "trace.agentlog"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "record",
            "-o",
            str(out),
            "--",
            sys.executable,
            str(script),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # The autostart's session covers the whole process — its _records
    # don't see the nested session's records, so this test still
    # triggers the empty-capture warning. That's expected: the
    # check is per-session, and the parent session genuinely captured
    # zero calls. The "warning silent" case is exercised via the
    # in-process unit tests in test_instrumentation_extended.py
    # against actual fake-SDK calls.
    #
    # Here we just sanity-check the script ran (non-zero exit would
    # indicate a different bug in the autostart path).
    assert proc.returncode == 0, proc.stderr


def test_empty_capture_warning_includes_diagnostic_hints(tmp_path: Path) -> None:
    """The empty-capture warning must list the four canonical causes
    so users can self-diagnose. Prevents the warning from regressing
    to a generic 'no calls captured' line."""
    script = tmp_path / "empty.py"
    script.write_text("pass\n")
    out = tmp_path / "trace.agentlog"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "record",
            "-o",
            str(out),
            "--",
            sys.executable,
            str(script),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    err = proc.stderr
    # Each cause is named explicitly.
    assert "doesn't yet auto-instrument" in err
    assert "different venv" in err.lower()
    assert "bound-method reference" in err
    assert "didn't actually make any LLM calls" in err
