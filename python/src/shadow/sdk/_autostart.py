"""Zero-config session bootstrap.

Importing this module on Python interpreter startup wires a
`shadow.sdk.Session` around the entire process if
`SHADOW_SESSION_OUTPUT` is set in the environment. Used by
`shadow record -- <cmd>` to record an agent's LLM calls without
requiring any code change to the agent itself.

The bootstrap path is:

1. `shadow record -- python my_agent.py` prepends
   `shadow/sdk/_bootstrap/` to the child process's `PYTHONPATH`
   and sets `SHADOW_SESSION_OUTPUT=<file>.agentlog`.
2. On Python interpreter startup, Python finds
   `shadow/sdk/_bootstrap/sitecustomize.py` first and auto-imports
   it (standard CPython behaviour — see `site.py`).
3. That sitecustomize does `import shadow.sdk._autostart`, which
   runs this module.
4. If the env var is set, we construct a `Session`, enter it, and
   register an `atexit` handler to exit it cleanly on process
   shutdown.

The Session already monkey-patches `anthropic.*` and `openai.*`
call sites on `__enter__`, so every LLM call from the agent gets
recorded with zero code change. If neither SDK is installed, the
Session's instrumentor is a no-op and this still works.

## Guard rails

- **No-op when the env var is unset.** Safe to import from any
  sitecustomize unconditionally.
- **Idempotent.** A module-level flag prevents double-starting even
  if both `sitecustomize` and `usercustomize` hit us.
- **Never raises.** Any failure here would hard-crash the user's
  agent before their code runs. Exceptions are caught and printed
  to stderr with a `hint:` line; the agent continues without
  recording.
"""

from __future__ import annotations

import atexit
import os
import sys

_BOOTSTRAP_DONE = False


def _already_bootstrapped() -> bool:
    """True if a previous import of this module already started a Session."""
    return _BOOTSTRAP_DONE


def _start_session_from_env() -> None:
    """Start a `Session` if `SHADOW_SESSION_OUTPUT` is set.

    Kept as a plain function (not __init__ side effect) so tests can
    invoke it deterministically. Real bootstrap happens at module-import
    time via the `if __name__` guard below.
    """
    global _BOOTSTRAP_DONE
    if _BOOTSTRAP_DONE:
        return

    out_path = os.environ.get("SHADOW_SESSION_OUTPUT")
    if not out_path:
        return  # nothing to do; this is the common case on regular runs

    # Deliberately lazy — importing `shadow.sdk.Session` at sitecustomize
    # time pulls in pyyaml, redact, and the PyO3 _core. That's fine at
    # `shadow record` startup (we want recording), but noisy on every
    # python invocation if the user leaves us in their PYTHONPATH.
    # The env-var gate above is the only check that runs on the hot path.
    try:
        from shadow.sdk.session import Session
    except Exception as e:  # pragma: no cover — should be very rare
        print(
            f"shadow: autostart could not import Session ({e}); recording disabled.\n"
            "hint: `pip install shadow-diff` should have installed the SDK; "
            "check that your PYTHONPATH isn't shadowing it.",
            file=sys.stderr,
        )
        return

    tags = _tags_from_env()
    session = Session(output_path=out_path, tags=tags)
    try:
        session.__enter__()
    except Exception as e:  # pragma: no cover
        print(
            f"shadow: autostart Session.__enter__ failed ({e}); recording disabled.\n"
            "hint: check that the output directory is writable.",
            file=sys.stderr,
        )
        return

    def _flush() -> None:
        # `atexit` handlers run after the user's script completes,
        # including on unhandled exceptions. Swallow any error here —
        # we never want shadow's bookkeeping to mask the user's traceback.
        try:
            session.__exit__(None, None, None)
        except Exception as e:  # pragma: no cover
            print(
                f"shadow: autostart session flush failed ({e}); "
                f"trace at {out_path} may be incomplete.",
                file=sys.stderr,
            )

    atexit.register(_flush)
    _BOOTSTRAP_DONE = True


def _tags_from_env() -> dict[str, str]:
    """Parse `SHADOW_SESSION_TAGS=key1=v1,key2=v2` into a tag dict.

    Safe on malformed input — silently drops bad entries.
    """
    raw = os.environ.get("SHADOW_SESSION_TAGS", "")
    if not raw:
        return {}
    out: dict[str, str] = {}
    for entry in raw.split(","):
        if "=" not in entry:
            continue
        k, _, v = entry.partition("=")
        k, v = k.strip(), v.strip()
        if k:
            out[k] = v
    return out


# Fire immediately on import. sitecustomize imports this module, which
# triggers the Session bootstrap before the user's script runs.
try:
    _start_session_from_env()
except Exception as e:  # pragma: no cover
    # Triple-belt: nothing in this file should ever escape to the user's
    # process. If it does, print a hint and continue.
    print(
        f"shadow: autostart crashed ({e}); continuing without recording.\n"
        "hint: please file a bug at https://github.com/manav8498/Shadow/issues.",
        file=sys.stderr,
    )
