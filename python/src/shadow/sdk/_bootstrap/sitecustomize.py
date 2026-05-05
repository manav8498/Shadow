"""sitecustomize shim prepended to PYTHONPATH by `shadow record`.

On Python interpreter startup, the `site` module auto-imports the
first `sitecustomize` it finds on `sys.path`. `shadow record`
prepends the directory containing THIS file to the child's
PYTHONPATH, so this sitecustomize wins over any stdlib or
distribution-provided sitecustomize.

We do exactly one thing: import `shadow.sdk._autostart`, which
starts a `Session` if `SHADOW_SESSION_OUTPUT` is set in the
environment and does nothing otherwise. That indirection keeps
this file trivial so we don't need to worry about breaking user
environments even if our zero-config logic grows complex later.

If a user has their own sitecustomize on PYTHONPATH, prepending
this one shadows it. That's a known trade-off; `shadow record`
only activates this shim for the duration of its subprocess,
so it doesn't affect the user's Python install globally.

## Failure mode: shadow-diff missing from the target venv

The most common silent breakage is this: a user runs
`shadow record -- python my_agent.py` against an agent in a
DIFFERENT virtualenv (the parent shell has shadow-diff installed,
the child venv doesn't). The child Python imports our
sitecustomize, but `import shadow.sdk._autostart` fails because
shadow-diff isn't in the child's site-packages — and the agent
runs without any recording.

We detect that exact case (`SHADOW_SESSION_OUTPUT` set + import
fails) and fail LOUDLY: write a metadata-only stub trace at the
output path so the user sees something landed, plus an
unmistakable stderr warning. Without `SHADOW_SESSION_OUTPUT`
set, we stay silent — this shim is meant to be inert when
`shadow record` isn't actually wrapping us.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

# `datetime.UTC` is a Python 3.11+ alias; sitecustomize may run under
# whatever Python the target venv is built on — including Python 3.9
# from /usr/bin on macOS, or older system Pythons on RHEL/Ubuntu LTS.
# `timezone.utc` works on every Python 3.x.
_UTC = timezone.utc  # noqa: UP017 — must support Python <3.11

try:
    import shadow.sdk._autostart  # noqa: F401
except Exception as e:  # pragma: no cover — exercised only outside the parent venv
    out_path = os.environ.get("SHADOW_SESSION_OUTPUT", "")
    # No-op when shadow record isn't wrapping us. The shim should be
    # inert on regular Python invocations even if it gets stuck on
    # someone's PYTHONPATH.
    if out_path:
        py_exe = sys.executable
        # Loud stderr warning — the user MUST see this. Phrasing names
        # the most likely cause first (venv mismatch) so they can
        # diagnose without reading source.
        sys.stderr.write(
            "shadow: WARNING — recording is DISABLED.\n"
            f"  shadow-diff is not importable from this Python:\n"
            f"    {py_exe}\n"
            f"  Reason: {e!r}\n"
            "  Likely cause: the target venv doesn't have shadow-diff installed.\n"
            "  Fix:        pip install shadow-diff   (in the venv your agent uses)\n"
            "  After install, re-run the same `shadow record -- ...` command.\n"
            f"  A metadata-only stub trace will be written to {out_path}\n"
            "  so you can see this run was attempted.\n"
        )
        # Best-effort: write a metadata-only stub trace so downstream
        # tooling sees a file (not a missing artifact). The stub uses
        # a non-content-hash placeholder id with a recognisable prefix
        # so a stub is distinguishable from a real shadow trace.
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            ts = datetime.now(_UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            stub = {
                "version": "0.1",
                "id": "sha256:" + ("0" * 64),
                "kind": "metadata",
                "ts": ts,
                "parent": None,
                "meta": {
                    "shadow_record_error": "sdk_unavailable_in_target_venv",
                    "python_executable": py_exe,
                    "import_error": repr(e),
                },
                "payload": {
                    "sdk": {"name": "shadow", "version": "stub"},
                    "tags": {"shadow_record_status": "disabled"},
                },
            }
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(stub, ensure_ascii=False) + "\n")
        except Exception:
            # Even the stub failed — nothing left to do. The stderr
            # warning above is the canonical surface.
            pass
