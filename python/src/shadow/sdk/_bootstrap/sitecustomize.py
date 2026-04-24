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
"""

import contextlib

# Never break the user's Python startup. If shadow isn't importable
# for any reason, the agent still runs — just without recording.
with contextlib.suppress(Exception):  # pragma: no cover
    import shadow.sdk._autostart  # noqa: F401
